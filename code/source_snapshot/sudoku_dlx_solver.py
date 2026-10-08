"""Sudoku solver using Knuth's Algorithm X with Dancing Links (DLX).

The exact-cover model has 324 constraints:

* 81 cell constraints: every cell has one digit.
* 81 row-digit constraints: every row contains each digit once.
* 81 column-digit constraints: every column contains each digit once.
* 81 box-digit constraints: every 3x3 box contains each digit once.

Each candidate assignment (row, col, digit) covers exactly four constraints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


GRID_SIZE = 9
BOX_SIZE = 3
CELL_COUNT = GRID_SIZE * GRID_SIZE
CONSTRAINT_COUNT = 4 * CELL_COUNT


@dataclass(frozen=True)
class SolveResult:
    status: str
    solution: list[list[int]] | None
    nodes: int
    solutions_found: int


class Column:
    def __init__(self, name: int | str) -> None:
        self.name = name
        self.size = 0
        self.left: Column | Node = self
        self.right: Column | Node = self
        self.up: Column | Node = self
        self.down: Column | Node = self
        self.column = self


class Node:
    def __init__(self, column: Column, row_id: tuple[int, int, int]) -> None:
        self.column = column
        self.row_id = row_id
        self.left: Column | Node = self
        self.right: Column | Node = self
        self.up: Column | Node = self
        self.down: Column | Node = self


class DancingLinks:
    def __init__(self, column_count: int) -> None:
        self.root = Column("root")
        self.columns = [Column(i) for i in range(column_count)]

        prev: Column | Node = self.root
        for column in self.columns:
            column.left = prev
            column.right = self.root
            prev.right = column
            self.root.left = column
            prev = column

    def add_row(self, row_id: tuple[int, int, int], column_ids: Sequence[int]) -> None:
        first: Node | None = None
        prev: Node | None = None

        for column_id in column_ids:
            column = self.columns[column_id]
            node = Node(column, row_id)

            node.down = column
            node.up = column.up
            column.up.down = node
            column.up = node
            column.size += 1

            if first is None:
                first = node
            if prev is not None:
                node.left = prev
                node.right = first
                prev.right = node
                first.left = node
            prev = node

    def cover(self, column: Column) -> None:
        column.right.left = column.left
        column.left.right = column.right

        row = column.down
        while row is not column:
            node = row.right
            while node is not row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down

    def uncover(self, column: Column) -> None:
        row = column.up
        while row is not column:
            node = row.left
            while node is not row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up

        column.right.left = column
        column.left.right = column

    def choose_column(self) -> Column:
        best = self.root.right
        column = best.right
        while column is not self.root:
            if column.size < best.size:
                best = column
                if best.size == 0:
                    break
            column = column.right
        return best


def box_index(row: int, col: int) -> int:
    return (row // BOX_SIZE) * BOX_SIZE + (col // BOX_SIZE)


def constraint_ids(row: int, col: int, digit: int) -> tuple[int, int, int, int]:
    digit_idx = digit - 1
    cell = row * GRID_SIZE + col
    row_digit = CELL_COUNT + row * GRID_SIZE + digit_idx
    col_digit = 2 * CELL_COUNT + col * GRID_SIZE + digit_idx
    box_digit = 3 * CELL_COUNT + box_index(row, col) * GRID_SIZE + digit_idx
    return cell, row_digit, col_digit, box_digit


def parse_puzzle(puzzle: str | Sequence[int]) -> list[int]:
    if isinstance(puzzle, str):
        chars = [ch for ch in puzzle if ch in ".0123456789"]
        if len(chars) != CELL_COUNT:
            raise ValueError(f"Expected 81 puzzle cells, got {len(chars)}")
        return [0 if ch in ".0" else int(ch) for ch in chars]

    values = list(puzzle)
    if len(values) != CELL_COUNT:
        raise ValueError(f"Expected 81 puzzle cells, got {len(values)}")
    if any(value < 0 or value > 9 for value in values):
        raise ValueError("Puzzle values must be integers in 0..9")
    return [int(value) for value in values]


def build_links(values: Sequence[int]) -> DancingLinks:
    links = DancingLinks(CONSTRAINT_COUNT)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            clue = values[row * GRID_SIZE + col]
            digits = (clue,) if clue else range(1, GRID_SIZE + 1)
            for digit in digits:
                links.add_row((row, col, digit), constraint_ids(row, col, digit))
    return links


def rows_to_grid(rows: Iterable[Node]) -> list[list[int]]:
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for node in rows:
        row, col, digit = node.row_id
        grid[row][col] = digit
    return grid


def solve_dlx(
    puzzle: str | Sequence[int],
    *,
    max_nodes: int = 0,
    max_solutions: int = 1,
) -> SolveResult:
    values = parse_puzzle(puzzle)
    links = build_links(values)
    partial: list[Node] = []
    first_solution: list[list[int]] | None = None
    nodes = 0
    solutions_found = 0
    limit_hit = False

    def search() -> bool:
        nonlocal first_solution, limit_hit, nodes, solutions_found

        if links.root.right is links.root:
            solutions_found += 1
            if first_solution is None:
                first_solution = rows_to_grid(partial)
            return solutions_found >= max_solutions

        if max_nodes and nodes >= max_nodes:
            limit_hit = True
            return True

        column = links.choose_column()
        if column.size == 0:
            return False

        links.cover(column)
        row = column.down
        while row is not column:
            nodes += 1
            partial.append(row)

            node = row.right
            while node is not row:
                links.cover(node.column)
                node = node.right

            stop = search()

            node = row.left
            while node is not row:
                links.uncover(node.column)
                node = node.left
            partial.pop()

            if stop:
                links.uncover(column)
                return True
            row = row.down

        links.uncover(column)
        return False

    search()

    if limit_hit:
        status = "limit"
    elif first_solution is not None:
        status = "solved"
    else:
        status = "unsat"
    return SolveResult(status, first_solution, nodes, solutions_found)


def compact_grid(grid: Sequence[Sequence[int]]) -> str:
    return "".join(str(value) for row in grid for value in row)


def format_grid(grid: Sequence[Sequence[int]]) -> str:
    lines = []
    for row_idx, row in enumerate(grid):
        if row_idx and row_idx % BOX_SIZE == 0:
            lines.append("------+-------+------")
        parts = []
        for col_idx, value in enumerate(row):
            if col_idx and col_idx % BOX_SIZE == 0:
                parts.append("|")
            parts.append(str(value))
        lines.append(" ".join(parts))
    return "\n".join(lines)


def iter_stdin_puzzles() -> Iterator[str]:
    while True:
        try:
            line = input()
        except EOFError:
            return
        line = line.strip()
        if line:
            yield line


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve Sudoku puzzles with DLX.")
    parser.add_argument(
        "puzzle",
        nargs="?",
        help="81-character puzzle string; use 0 or . for blanks. Reads stdin if omitted.",
    )
    parser.add_argument("--max-nodes", type=int, default=0, help="0 means no search limit.")
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=1,
        help="Search up to this many solutions; use 2 to test uniqueness.",
    )
    parser.add_argument("--compact", action="store_true", help="Print one 81-digit line.")
    args = parser.parse_args()

    puzzles = [args.puzzle] if args.puzzle else list(iter_stdin_puzzles())
    exit_code = 0
    for idx, puzzle in enumerate(puzzles):
        result = solve_dlx(
            puzzle,
            max_nodes=args.max_nodes,
            max_solutions=max(1, args.max_solutions),
        )
        if len(puzzles) > 1:
            print(f"# puzzle {idx + 1}: {result.status}, nodes={result.nodes}")
        else:
            print(f"status={result.status} nodes={result.nodes} solutions={result.solutions_found}")

        if result.solution is None:
            exit_code = 1
            continue
        if args.compact:
            print(compact_grid(result.solution))
        else:
            print(format_grid(result.solution))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
