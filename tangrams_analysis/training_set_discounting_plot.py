#!/usr/bin/env python3

"""
Plots the results of using different training set discount values.

Use with e.g. "find ~/Documents/Projects/Tangrams/output -iname "Training-discount*\bothspkr.tsv" -exec ./training_set_discounting_plot.py {} +"
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import os
import re
import sys
import typing

import pandas as pd

MULTIVALUE_DELIM_PATTERN = re.compile("\\s*,\\s*")
RESULTS_FILE_ENCODING = "utf-8"
RESULTS_FILE_CSV_DIALECT = csv.excel_tab


def __parse_sequence(input_str: str) -> typing.Tuple[str, ...]:
	values = MULTIVALUE_DELIM_PATTERN.split(input_str)
	return tuple(sys.intern(value) for value in values)


def __parse_set(input_str: str) -> typing.FrozenSet[str]:
	values = MULTIVALUE_DELIM_PATTERN.split(input_str)
	return frozenset(sys.intern(value) for value in values)


__RESULTS_FILE_CONVERTERS = {"REFERRING_TOKENS": __parse_sequence, "REFERRRING_TOKEN_TYPES": __parse_set,
							 "OOV_TYPES": __parse_set}
__RESULTS_FILE_DTYPES = {"DYAD": "category", "SHAPE": "category", "ONLY_INSTRUCTOR": bool}


def read_results_file(inpath: str) -> pd.DataFrame:
	print("Reading \"{}\".".format(inpath), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=RESULTS_FILE_ENCODING, memory_map=True, converters=__RESULTS_FILE_CONVERTERS,
						 dtype=__RESULTS_FILE_DTYPES)
	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Visualizes usage of vocabulary across sessions as a heatmap.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The files to process.")

	return result


def __dyad_id(infile: str) -> str:
	session_dir = os.path.dirname(infile)
	return os.path.split(session_dir)[1]


def __main(args):
	inpaths = args.inpaths
	print("Will read {} file(s).".format(len(inpaths)), file=sys.stderr)
	cv_results = pd.concat((read_results_file(inpath) for inpath in inpaths))
	print(
		"Read {} cross-validation round(s) from {} file(s) with {} column(s).".format(cv_results.shape[0], len(inpaths),
																					  cv_results.shape[1]),
		file=sys.stderr)
	cv_results.pivot_table("RANK", "")


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
