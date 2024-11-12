"""This module contains the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()
BLD = ROOT.joinpath("bld").resolve()
PUBLIC = BLD.joinpath("public").resolve()
TESTS = ROOT.joinpath("tests").resolve()


__all__ = ["BLD", "SRC", "TESTS"]
