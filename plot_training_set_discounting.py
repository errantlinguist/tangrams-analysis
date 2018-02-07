#!/usr/bin/env python3

"""
Plots the results of using different training set discount values.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import os
import re
import sys
from typing import Iterable, Pattern

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tangrams_analysis import natural_keys

MULTIVALUE_DELIM_PATTERN = re.compile("\\s*,\\s*")
RESULTS_FILE_ENCODING = "utf-8"
RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
__RESULTS_FILE_DTYPES = {"DYAD": "category", "IS_TARGET": bool, "IS_OOV": bool,
						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}


def plot_ranks(discount_mean_ranks: pd.DataFrame) -> sns.axisgrid.FacetGrid:
	# https://stackoverflow.com/a/47407428/1391325
	# Use lmplot to plot scatter points
	result = sns.lmplot(x="BACKGROUND_DATA_WORD_TOKEN_COUNT", y="RR", hue="DYAD", data=discount_mean_ranks,
					   fit_reg=False)
	# Use regplot to plot the regression line for the whole points
	sns.regplot(x="BACKGROUND_DATA_WORD_TOKEN_COUNT", y="RR", data=discount_mean_ranks, scatter=False,
				ax=result.axes[0, 0])
	result.set_axis_labels("Training set size", "MRR")
	return result


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
		description="Plots the results of using different training set discount values.")
	result.add_argument("infiles", metavar="INPATH", nargs='+',
						help="The files to process.")
	result.add_argument("-e", "--encoding", metavar="CODEC", default="utf-8",
						help="The input file encoding.")
	result.add_argument("-p", "--pattern", metavar="REGEX", type=re.compile, default=re.compile(".+\.tsv"),
						help="A regular expression to match the desired files.")
	result.add_argument("-o", "--outfile", metavar="OUTFILE",
						help="The path to write the plot graphics to.")
	return result


def __parse_format(path: str, default="png") -> str:
	_, ext = os.path.splitext(path)
	# Strip leading "." of extension
	return ext[1:] if ext else default


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
	cv_results["RR"] = 1.0 / cv_results["RANK"]

	# noinspection PyUnresolvedReferences
	discount_results = cv_results.groupby(["DYAD", "BACKGROUND_DATA_WORD_TOKEN_COUNT"], as_index=False)
	discount_mean_ranks = discount_results.agg({"RANK": "mean", "RR": "mean"})
	print("Plotting.", file=sys.stderr)
	sns.set_style("whitegrid")
	graph = plot_ranks(discount_mean_ranks)

	outfile = args.outfile
	if outfile:
		output_format = __parse_format(outfile)
		print("Writing to \"{}\" as format \"{}\".".format(outfile, output_format), file=sys.stderr)
		graph.savefig(outfile, format=output_format, dpi=1000)
	else:
		plt.show()


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
