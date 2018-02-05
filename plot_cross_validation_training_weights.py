#!/usr/bin/env python3

"""
Creates a scatter plot of cross-validation rankings for different update training weights.

Use with e.g. "find ~/Documents/Projects/Tangrams/Data/output/ -iname "*bothspkr*.tsv" -exec ./plot_cross_validation_training_weights.py {} +"
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import sys

import pandas as pd

from tangrams_analysis import natural_keys

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
__RESULTS_FILE_DTYPES = {"DYAD": "category", "IS_TARGET": bool, "IS_OOV": bool,
						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}


def read_results_file(inpath: str, encoding: str) -> pd.DataFrame:
	print("Reading \"{}\" using encoding \"{}\".".format(inpath, encoding), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=encoding, memory_map=True, dtype=__RESULTS_FILE_DTYPES)
	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Writes only certain dyads from a word score file.")
	result.add_argument("infiles", metavar="FILE", nargs='+',
						help="The cross-validation results files to process.")
	result.add_argument("-e", "--encoding", metavar="CODEC", default="utf-8",
						help="The input file encoding.")
	return result


def __main(args):
	infiles = args.infiles
	encoding = args.encoding
	print("Will read {} cross-validation results file(s) using encoding \"{}\".".format(len(infiles), encoding),
		  file=sys.stderr)
	cv_results = pd.concat((read_results_file(infile, encoding) for infile in infiles))
	print("Read results for {} iteration(s) of {}-fold cross-validation (one for each dyad).".format(
		cv_results["CROSS_VALIDATION_ITER"].nunique(), cv_results["DYAD"].nunique()), file=sys.stderr)
	print("Dyads present: {}".format(sorted(cv_results["DYAD"].unique(), key=natural_keys)), file=sys.stderr)
	print("Unique discounting values: {}".format(sorted(cv_results["DISCOUNT"].unique())), file=sys.stderr)
	print("Unique updating weights: {}".format(sorted(cv_results["UPDATE_WEIGHT"].unique())), file=sys.stderr)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
