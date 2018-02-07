#!/usr/bin/env python3

"""
Creates a scatter plot of cross-validation rankings for different update training weights.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import os
import re
import sys
from typing import Iterable, Pattern

import pandas as pd
# Matplotlib for additional customization
import seaborn as sns

from tangrams_analysis import natural_keys

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
__RESULTS_FILE_DTYPES = {"DYAD": "category", "IS_TARGET": bool, "IS_OOV": bool,
						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}


def read_results_files(inpaths: Iterable[str], pattern: Pattern, encoding: str) -> pd.DataFrame:
	dfs = []
	for inpath in inpaths:
		if os.path.isdir(inpath):
			for root, dirnames, filenames in os.walk(inpath, followlinks=True):
				for filename in filenames:
					if pattern.match(filename):
						abs_path = os.path.join(root, filename)
						dfs.append(read_results_file(abs_path, encoding))
		else:
			# Don't check if the file matches the pattern for directly-specified files
			dfs.append(read_results_file(inpath, encoding))
	return pd.concat(dfs)


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
	result.add_argument("-p", "--pattern", metavar="REGEX", type=re.compile, default=re.compile(".+\.tsv"),
						help="A regular expression to match the desired files.")
	result.add_argument("-o", "--outfile", metavar="OUTFILE", required=True,
						help="The path to write the plot graphics to.")
	return result


def __main(args):
	infiles = args.infiles
	pattern = args.pattern
	encoding = args.encoding
	print("Will search {} paths(s) for files matching \"{}\", to be read using encoding \"{}\".".format(len(infiles),
																										pattern.pattern,
																										encoding),
		  file=sys.stderr)
	cv_results = read_results_files(infiles, pattern, encoding)
	print("Read results for {} iteration(s) of {}-fold cross-validation (one for each dyad).".format(
		cv_results["CROSS_VALIDATION_ITER"].nunique(), cv_results["DYAD"].nunique()), file=sys.stderr)
	print("Dyads present: {}".format(sorted(cv_results["DYAD"].unique(), key=natural_keys)), file=sys.stderr)
	print("Unique discounting values: {}".format(sorted(cv_results["DISCOUNT"].unique())), file=sys.stderr)
	print("Unique updating weights: {}".format(sorted(cv_results["UPDATE_WEIGHT"].unique())), file=sys.stderr)
	# groups = cv_results.groupby(("ROUND", "UPDATE_WEIGHT"))
	# print(groups["RANK"].mean())

	sns.set_style("whitegrid")
	sns_plot = sns.lmplot(x="ROUND", y="RANK", hue="UPDATE_WEIGHT", data=cv_results)
	# https://stackoverflow.com/a/39482402/1391325
	# fig = sns_plot.get_figure()
	outfile = args.outfile
	ext = os.path.splitext(outfile)[1][1:]
	print("Writing to \"{}\" as format \"{}\".".format(outfile, ext), file=sys.stderr)
	sns_plot.savefig(outfile, format=ext, dpi=1000)


# fig.savefig(...)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
