#!/usr/bin/env python3

"""
Writes only certain dyads from a word score file.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import sys

import pandas as pd

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
__RESULTS_FILE_DTYPES = {"DYAD": "category", "WORD": "category", "IS_TARGET": bool, "IS_OOV": bool,
						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}


def read_results_file(inpath: str, encoding: str) -> pd.DataFrame:
	print("Reading \"{}\" using encoding \"{}\".".format(inpath, encoding), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=encoding, memory_map=True, dtype=__RESULTS_FILE_DTYPES)
	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Learns a measure of referential salience of classifiers used based on the context of their corresponding words in dialogue.")
	result.add_argument("infiles", metavar="FILE", nargs='+',
						help="The cross-validation results files to process.")
	result.add_argument("-e", "--encoding", metavar="CODEC", default="utf-8",
						help="The input file encoding.")
	result.add_argument("-p", "--pattern", metavar="REGEX", required=True,
						help="A regular expression to match the desired dyad IDs.")
	return result


def __main(args):
	infiles = args.infiles
	encoding = args.encoding
	print("Will read {} cross-validation results file(s) using encoding \"{}\".".format(len(infiles), encoding),
		  file=sys.stderr)
	word_scores = pd.concat((read_results_file(infile, encoding) for infile in infiles))
	dyad_id_pattern = args.pattern
	filtered_scores = word_scores[word_scores["DYAD"].str.match(dyad_id_pattern)]
	print("Writing data for dyad(s): {}".format(sorted(filtered_scores["DYAD"].unique())), file=sys.stderr)
	filtered_scores.to_csv(sys.stdout, sep=RESULTS_FILE_CSV_DIALECT.delimiter, encoding=encoding, index=False)
	print("Finished writing {} row(s).".format(filtered_scores.shape[0]), file=sys.stderr)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
