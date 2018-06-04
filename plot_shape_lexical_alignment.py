#!/usr/bin/env python3

"""
Creates a plot of the frequencies of words used for a given shape.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import logging
import os
import sys
import typing
from collections import Counter
from enum import Enum, unique
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tangrams_analysis import utterances
import tangrams_analysis.session_data

def add_session_col(indir : str, df : pd.DataFrame):
	session = parse_dir_session_name(indir)
	df["DYAD"] = session

def parse_dir_session_name(dirpath: str) -> str:
	return os.path.basename(dirpath)

def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a plot of the frequencies of words used for a given shape.")
	result.add_argument("session_dir", metavar="PATH", help="The directory under which the dyad files are to be found.")
	return result


def __main(args):
	session_dir = args.session_dir
	print("Will look for sessions underneath \"{}\".".format(session_dir), file=sys.stderr)
	session_data = tangrams_analysis.session_data.walk_session_data((session_dir,))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())