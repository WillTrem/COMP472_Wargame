from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random

import requests
import math

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""

    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""

    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


##############################################################################################################


@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################


@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""

    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = "?"
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = "?"
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    def iter_diagonal(self) -> Iterable[Coord]:
        """ "Iterates over diagonal Coords."""
        yield Coord(self.row - 1, self.col - 1)
        yield Coord(self.row - 1, self.col + 1)
        yield Coord(self.row + 1, self.col - 1)
        yield Coord(self.row + 1, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 2:
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################


@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""

    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 4:
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################


@dataclass(slots=True)
class Options:
    """Representation of the game options."""

    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = None
    randomize_moves: bool = True
    broker: str | None = None


##############################################################################################################


@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""

    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################


@dataclass(slots=True)
class Game:
    """Representation of the game state."""

    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    file: file | None = None

    def write_initial_state_to_file(self):
        """Writes the initial parameters and state of the game to the output file"""
        self.file = open(
            f"gameTrace-{self.options.alpha_beta}-{self.options.max_time}-{self.options.max_turns}.txt",
            "w+",
        )
        self.file.write("---PARAMETERS---\n")
        self.file.write(f"Timeout value: {self.options.max_time} seconds\n")
        self.file.write(f"Alpha-beta: {self.options.alpha_beta}\n")
        # TODO: make it look better
        self.file.write(f"Play mode: {self.options.game_type.name}\n")
        # TODO: print heuristic if one of the players is AI

        self.file.write("\n---INITIAL BOARD CONFIG---\n")
        self.file.write(self.to_string())
        self.file.write("\n---GAME TRACE---\n")

    def write_turn_to_file(self, move: CoordPair):
        """Writes the current turn to the output file"""
        self.file.write(f"\nTurn #{self.turns_played + 1}\n")
        self.file.write(f"Player: {self.next_player.name}\n")
        self.file.write(f"Action taken: {move.to_string()}\n")
        # TODO: Write AI time for the action
        # TODO: Write AI heuristic score of the resulting board
        self.file.write(
            f"New board configuration: \n{self.board_to_string()}\n")
        # TODO: AI cumulative information about the game so far

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(
            player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(
            player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(
            player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md),
                 Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1),
                 Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md),
                 Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2),
                 Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(
            Coord(md - 1, md - 1), Unit(player=Player.Attacker,
                                        type=UnitType.Firewall)
        )

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def count_units(self, player: Player, unit_type: UnitType) -> int:
        """Count the number of units of a specific type for a given player."""
        mid_cell = Coord(2, 2)
        grid = mid_cell.iter_range(2)
        count = 0
        for cell in grid:
            if (
                self.get(cell) is not None
                and self.get(cell).type == unit_type
                and self.get(cell).player == player
            ):
                count += 1
        return count

    def heuristic_e0(self) -> int:
        """Heuristic function e0 based on the provided formula."""
        attacker_vp = 3 * self.count_units(Player.Attacker, UnitType.Virus)
        attacker_tp = 3 * self.count_units(Player.Attacker, UnitType.Tech)
        attacker_fp = 3 * self.count_units(Player.Attacker, UnitType.Firewall)
        attacker_pp = 3 * self.count_units(Player.Attacker, UnitType.Program)
        attacker_aip = 9999 * self.count_units(Player.Attacker, UnitType.AI)

        defender_vp = 3 * self.count_units(Player.Defender, UnitType.Virus)
        defender_tp = 3 * self.count_units(Player.Defender, UnitType.Tech)
        defender_fp = 3 * self.count_units(Player.Defender, UnitType.Firewall)
        defender_pp = 3 * self.count_units(Player.Defender, UnitType.Program)
        defender_aip = 9999 * self.count_units(Player.Defender, UnitType.AI)

        return (
            attacker_vp + attacker_tp + attacker_fp + attacker_pp + attacker_aip
        ) - (defender_vp + defender_tp + defender_fp + defender_pp + defender_aip)

    # Heuristic e1 - This heuristic takes into consideration the relative proximity of the viruses/programs to the defender's AI
    # We will give a higher score to the viruses since they do more damage, and since they can move freely!
    def heuristic_e1(self) -> float:
        ai_position = self.ai_position()
        virus_positions = self.virus_position()
        program_positions = self.prog_position()

        virus_number = self.count_units(Player.Attacker, UnitType.Virus)
        program_number = self.count_units(Player.Attacker, UnitType.Program)

        virus_reach = 0
        program_reach = 0

        if virus_number > 0:
            for virus in virus_positions:
                virus_reach += self.reach(ai_position, virus)

        if program_number > 0:
            for program in program_positions:
                program_reach += self.reach(ai_position, program)

        # player unit returns an iterable array of tuples, the second arg in tuples is type unit, fetch health like that
        defender_units = self.player_units(Player.Defender)
        defender_health = 0

        for unit in defender_units:
            if unit[1].type == UnitType.AI:
                defender_health += 50 * unit[1].health
            else:
                defender_health += unit[1].health

        score = (
            (5 * virus_number) / virus_reach
            + (2 * program_number) / program_reach
            - 0.001 * defender_health
        )
        return score

    # Heuristic e2 - Takes into account the health of all units, weighted by importance
    def heuristic_e2(self) -> int:
        defender_units = self.player_units(Player.Defender)
        attacker_units = self.player_units(Player.Attacker)

        defender_health = 0
        attacker_health = 0

        for unit in defender_units:
            if unit[1].type == UnitType.AI:
                defender_health += 250 * unit[1].health
            elif unit[1].type == UnitType.Tech:
                defender_health += 25 * unit[1].health
            elif unit[1].type == UnitType.Firewall:
                defender_health += 10 * unit[1].health
            else:
                defender_health += unit[1].health

        for unit in attacker_units:
            if unit[1].type == UnitType.AI:
                attacker_health += 250 * unit[1].health
            elif unit[1].type == UnitType.Virus:
                attacker_health += 25 * unit[1].health
            elif unit[1].type == UnitType.Program:
                attacker_health += 10 * unit[1].health
            else:
                attacker_health += unit[1].health

        return 0.001 * attacker_health - 0.01 * defender_health

    def evaluate(self, eval_func=None):
        if eval_func == "e1":
            return self.heuristic_e1()
        elif eval_func == "e2":
            return self.heuristic_e2()
        else:
            return self.heuristic_e0()

    def minimax(self, game, depth, maximizing_player):
        if (
            depth == 0 or game.is_finished()
        ):  # If the search depth is 0 or the game is finished, return an evaluation score
            return game.evaluate()

        if maximizing_player:
            max_eval = float("-inf")
            # For the maximizing player (attacker), find the move with the highest score.
            for child in game.get_children_nodes(Player.Attacker):
                # Recursively call minimax with the defender's perspective (maximizing_player=False)
                eval = self.minimax(child, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float("inf")
            # For the minimizing player (defender), find the move with the lowest score.
            for child in game.get_children_nodes(Player.Defender):
                # Recursively call minimax with the attacker's perspective (maximizing_player=True)
                eval = self.minimax(child, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def alpha_beta(self, depth: int, alpha: float, beta: float):

        if depth == 0 or self.is_finished():
            return (self.evaluate(), None)

        # Generate all possible moves
        children = self.get_children_nodes(self.next_player)
        # TODO: find branching factor
        if self.next_player == Player.Attacker:  # We try to Maximize the Attacker's score
            maxValue = -(math.inf)
            maxMove = None
            # Iterate over all the children
            for (game_copy, move) in children:
                if maxMove is None:
                    maxMove = move
                value = game_copy.alpha_beta(depth-1, alpha, beta)[0]
                maxValue = max(value, maxValue)
                if value > maxValue:
                    maxValue = value
                    maxMove = move
                alpha = max(alpha, maxValue)
                if beta <= alpha:
                    break  # Alpha Pruning
            return (maxValue, maxMove)
        else:  # We try to Minimize the Attacker's score
            minValue = math.inf
            minMove = None
            # Iterate over all the children
            for (game_copy, move) in children:
                if minMove is None:
                    minMove = move
                value = game_copy.alpha_beta(depth-1, alpha, beta)[0]
                if value <= minValue:
                    minValue = value
                    minMove = move
                beta = min(beta, minValue)
                if beta <= alpha:
                    break  # Beta Pruning
            return (minValue, minMove)

    def get_children_nodes(self, player: Player):
        children = []
        for (coord, unit) in self.player_units(player):
            adjacentCoords = Coord.iter_adjacent(coord)
            # Iterates over the possible moves of the current unit to validate them
            for adjacentCoord in adjacentCoords:
                game_copy = self.clone()
                move = CoordPair(coord, adjacentCoord)
                if game_copy.perform_move(move)[0]:
                    game_copy.next_turn()
                    children.append((game_copy, move))
            # Considering the possiblity of self-destruct
            selfdestruct_move = CoordPair(coord, coord)
            if game_copy.perform_move(selfdestruct_move)[0]:
                game_copy.next_turn()
                children.append((game_copy, selfdestruct_move))
        return children

    # Method to return the defender's AI's position in the grid system
    def ai_position(self) -> Coord:
        mid_position = Coord(2, 2)
        grid = mid_position.iter_range(2)

        for coord in grid:
            unit = self.get(coord)
            if unit and unit.player == Player.Defender and unit.type == UnitType.AI:
                position = coord
                return position

    # Method to return the attacker's program positions in the grid system

    def prog_position(self) -> Iterable[Coord]:
        mid_position = Coord(2, 2)
        grid = mid_position.iter_range(2)

        for coord in grid:
            unit = self.get(coord)
            if (
                unit
                and unit.player == Player.Attacker
                and unit.type == UnitType.Program
            ):
                yield coord

    # Method to return the attacker's virus positions in the grid system
    def virus_position(self) -> Iterable[Coord]:
        mid_position = Coord(2, 2)
        grid = mid_position.iter_range(2)

        for coord in grid:
            unit = self.get(coord)
            if unit and unit.type == UnitType.Virus:
                yield coord

    # The reach returns how easy it is for a unit at coordinate t to get to a unit at coordinate s
    def reach(self, coord_s: Coord, coord_t: Coord) -> int:
        reach = 0
        y_reach = coord_s.row - coord_t.row
        x_reach = coord_s.col - coord_t.col
        unit_up = Coord(coord_t.row - 1, coord_t.col)
        unit_down = Coord(coord_t.row + 1, coord_t.col)
        unit_left = Coord(coord_t.row, coord_t.col - 1)
        unit_right = Coord(coord_t.row, coord_t.col + 1)

        # If you must go in a certain direction and that direction is blocked 2 ways, increase the reach (you need more moves to get there)
        # These checks assume the units blocking you are your own so that you can move them
        if y_reach < 0 and x_reach < 0 and self.get(unit_up) and self.get(unit_left):
            reach += 1
        elif y_reach < 0 and x_reach > 0 and self.get(unit_up) and self.get(unit_right):
            reach += 1
        elif (
            y_reach > 0 and x_reach < 0 and self.get(
                unit_down) and self.get(unit_left)
        ):
            reach += 1
        elif (
            y_reach > 0 and x_reach > 0 and self.get(
                unit_down) and self.get(unit_right)
        ):
            reach += 1

        # If any of the adjacent coordinates are enemy units, increase the reach by 1. This means you might have to defeat them first to get
        # to the AI (like Techs) or that you must go around them
        if (
            (self.get(unit_right) and self.get(
                unit_right).player == Player.Defender)
            or (self.get(unit_left) and self.get(unit_left).player == Player.Defender)
            or (self.get(unit_up) and self.get(unit_up).player == Player.Defender)
            or (self.get(unit_down) and self.get(unit_down).player == Player.Defender)
        ):
            reach += 1

        reach += abs(y_reach) + abs(x_reach) + 1
        return reach

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            # print("Invalid coordinate input")
            return False

        # Generating the unit from the source coordinate, generating an array of all adjacent coordinates for possible moves
        unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)
        adjacentCoords = Coord.iter_adjacent(coords.src)

        # Checking that the source coordinate is indeed a unit
        if unit is None or unit.player != self.next_player:
            # print(
            #     "There is no unit in your source coordinate or you do not own that unit."
            # )
            return False

        # Checking that the destination coordinate is an adjacent square
        if coords.dst not in adjacentCoords and coords.dst != coords.src:
            # print(
            #     "Your destination coordinate is not adjacent to the source coordinate."
            # )
            return False

        # No movement restrictions on Viruses or Tech
        if (
            unit.type not in [UnitType.Virus,
                              UnitType.Tech] and dst_unit is None
        ):  # Those units may still repair or attack in those coordinates
            # Verifying if the unit is engaged in combat
            for adjacentCoord in adjacentCoords:
                adjacentUnit = self.get(adjacentCoord)
                if adjacentUnit is not None and adjacentUnit.player != unit.player:
                    # print("That unit is engaged in combat and cannot flee.")
                    return False

            # Validating move depending on the Player type
            if (
                unit.player == Player.Attacker
            ):  # Attacker's AI, Firewall and Program can only go left or up
                if coords.dst.col > coords.src.col or coords.dst.row > coords.src.row:
                    # print("That unit may only go up or left.")
                    return False
            else:  # Defender's AI, Firewall and Program can only go right or down
                if coords.dst.col < coords.src.col or coords.dst.row < coords.src.row:
                    # print("That unit may only go down or right.")
                    return False

        # If the move is a repair, checking that a tech is not repairing a virus
        if (
            dst_unit is not None
            and dst_unit.player == unit.player
            and not dst_unit == unit
        ):  # Checking if the move is a repair
            if unit is UnitType.Tech and dst_unit is UnitType.Virus:
                # print("Your tech cannot repair your virus.")
                return False

            if (
                dst_unit.health >= 9
            ):  # Checking if the destination unit is alrdy at full health
                # print("That unit is at full health. You can't repair it.")
                return False

        return True

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        # Checking first if the move is valid
        if self.is_valid_move(coords):
            # Generating the source unit
            src_unit = self.get(coords.src)
            dst_unit = self.get(coords.dst)
            adjacentCoords = Coord.iter_adjacent(coords.src)
            diagonalCoords = Coord.iter_diagonal(
                coords.src
            )  # Generating adjacent but diagonal coordinates for self-destruct

            # Empty destination -> Perform the movement, stop the method
            if not dst_unit:
                self.set(coords.dst, src_unit)
                self.set(coords.src, None)

            # Src and Dst coordinates are the same -> Perform self-destruct
            elif coords.src == coords.dst:
                # Source unit is removed from the board
                self.mod_health(coords.src, -src_unit.health)

                # Damage all adjacent units by 2
                for adjacentCoord in adjacentCoords:
                    adjacentUnit = self.get(adjacentCoord)
                    if adjacentUnit is not None:
                        adjacentUnit.mod_health(-2)

                for diagonalCoord in diagonalCoords:
                    diagonalUnit = self.get(diagonalCoord)
                    if diagonalUnit is not None:
                        diagonalUnit.mod_health(-2)

            # Enemy unit at destination -> Perform the attack
            elif dst_unit and dst_unit.player != src_unit.player:
                t_dmg = src_unit.damage_amount(dst_unit)
                s_dmg = dst_unit.damage_amount(src_unit)
                self.mod_health(coords.src, -s_dmg)
                self.mod_health(coords.dst, -t_dmg)

            # Team unit at destination -> Perform the repair
            elif dst_unit and dst_unit.player == src_unit.player:
                repair = src_unit.repair_amount(dst_unit)
                self.mod_health(coords.dst, repair)

            return (True, "")

        return (False, "invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def board_to_string(self) -> str:
        """Text representation of the game board only"""
        dim = self.options.dim
        coord = Coord()
        output = "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(f"Player {self.next_player.name}, enter your move: ")
            coords = CoordPair.from_string(s)
            if (
                coords is not None
                and self.is_valid_coord(coords.src)
                and self.is_valid_coord(coords.dst)
            ):
                return coords
            else:
                print("Invalid coordinates! Try again.")

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end="")
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end="")
                    print(result)
                    self.write_turn_to_file(mv)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end="")
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if (
            self.options.max_turns is not None
            and self.turns_played >= self.options.max_turns
        ):
            return Player.Defender
        elif self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for src, _ in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        # (score, move) = self.random_move()  # Removed avg_depth
        (score, move) = self.alpha_beta(self.options.min_depth, -
                                        (math.inf), math.inf)  # Removed avg_depth
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Suggested move: {move} with score of {score} ")

        print(f"Heuristic score: {self.evaluate()}")
        # print(f"Average recursive depth: {avg_depth:0.1f}") we don't need this
        print(f"Evals per depth: ", end="")
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end="")
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(
                f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")

        best_move = None
        max_eval = float("-inf")

        # for child in game.get_children_nodes(Player.Attacker):
        #     eval = self.minimax(child, game.options.max_depth, False)
        #     if eval > max_eval:
        #         max_eval = eval
        #         best_move = move
        # return best_move
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played,
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if (
                r.status_code == 200
                and r.json()["success"]
                and r.json()["data"] == data
            ):
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}"
                )
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {"Accept": "application/json"}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()["success"]:
                data = r.json()["data"]
                if data is not None:
                    if data["turn"] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data["from"]["row"], data["from"]["col"]),
                            Coord(data["to"]["row"], data["to"]["col"]),
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}"
                )
        except Exception as error:
            print(f"Broker error: {error}")
        return None


##############################################################################################################


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog="ai_wargame", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--max_depth", type=int, help="maximum search depth")
    parser.add_argument("--max_time", type=float, help="maximum search time")
    parser.add_argument(
        "--game_type",
        type=str,
        default="manual",
        help="game type: auto|attacker|defender|manual",
    )
    parser.add_argument("--broker", type=str, help="play via a game broker")
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    if options.max_turns is None:
        while True:
            try:
                max_turns = int(input("Enter the maximum number of turns: "))
                if max_turns > 0:
                    options.max_turns = max_turns
                    break
                else:
                    print("Please enter a positive number of turns.")
            except ValueError:
                print("Please enter a valid integer for the number of turns.")

    # create a new game
    game = Game(options=options)

    game.write_initial_state_to_file()

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            game.file.write(f"\n{winner.name} wins in {game.turns_played}")
            game.file.close()
            break

        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif (
            game.options.game_type == GameType.AttackerVsComp
            and game.next_player == Player.Attacker
        ):
            game.human_turn()
        elif (
            game.options.game_type == GameType.CompVsDefender
            and game.next_player == Player.Defender
        ):
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)


##############################################################################################################


if __name__ == "__main__":
    main()
