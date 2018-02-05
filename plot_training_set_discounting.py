#!/usr/bin/env python3

"""
Plots the results of using different training set discount values.

Use with e.g. "find ~/Documents/Projects/Tangrams/Data/output/tangrams-morediscounting -iname "Training-discount*bothspkr.tsv" -exec ./plot_training_set_discounting.py {} +"
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import re
import sys
import typing
from numbers import Integral
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MULTIVALUE_DELIM_PATTERN = re.compile("\\s*,\\s*")
RESULTS_FILE_ENCODING = "utf-8"
RESULTS_FILE_CSV_DIALECT = csv.excel_tab

__LONGFLOAT_ONE = np.longfloat(1)


def __parse_sequence(input_str: str) -> typing.Tuple[str, ...]:
	values = MULTIVALUE_DELIM_PATTERN.split(input_str)
	return tuple(sys.intern(value) for value in values)


def __parse_set(input_str: str) -> typing.FrozenSet[str]:
	values = MULTIVALUE_DELIM_PATTERN.split(input_str)
	return frozenset(sys.intern(value) for value in values)


__RESULTS_FILE_CONVERTERS = {"REFERRING_TOKENS": __parse_sequence, "REFERRING_TOKEN_TYPES": __parse_set,
							 "OOV_TYPES": __parse_set}
__RESULTS_FILE_DTYPES = {"DYAD": "category", "SHAPE": "category", "ONLY_INSTRUCTOR": bool}


def anonymize_dyad_ids(df: pd.DataFrame, dyad_ids: Sequence[str]):
	dyad_id_idxs = dict((dyad_id, idx) for (idx, dyad_id) in enumerate(dyad_ids, start=1))
	df["DYAD"] = df["DYAD"].transform(lambda dyad_id: dyad_id_idxs[dyad_id])


def create_training_set_size_series(df: pd.DataFrame, dyad_ids: Sequence[str]) -> pd.Series:
	max_training_set_size = len(dyad_ids) - 1
	return df["TRAINING_SET_SIZE_DISCOUNT"].transform(
		lambda discount_value: (max_training_set_size - discount_value) / np.longfloat(max_training_set_size))


def plot_ranks(discount_mean_ranks: pd.DataFrame) -> sns.axisgrid.FacetGrid:
	sns.set()
	# https://stackoverflow.com/a/47407428/1391325
	# Use lmplot to plot scatter points
	graph = sns.lmplot(x="TRAINING_SET_SIZE", y="RR", hue="DYAD", data=discount_mean_ranks, fit_reg=False)
	# Use regplot to plot the regression line for the whole points
	sns.regplot(x="TRAINING_SET_SIZE", y="RR", data=discount_mean_ranks, scatter=False, ax=graph.axes[0, 0])
	graph.set_axis_labels("Training set size", "MRR")
	return graph


def read_results_file(inpath: str) -> pd.DataFrame:
	print("Reading \"{}\".".format(inpath), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=RESULTS_FILE_ENCODING, memory_map=True, converters=__RESULTS_FILE_CONVERTERS,
						 dtype=__RESULTS_FILE_DTYPES)
	return result


def rr(rank: Integral) -> np.longfloat:
	return __LONGFLOAT_ONE / np.longfloat(rank)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Plots the results of using different training set discount values.")
	result.add_argument("infiles", metavar="INPATH", nargs='+',
						help="The files to process.")
	result.add_argument("-o", "--outfile", metavar="OUTFILE",
						help="The path to write the plot graphics to.")
	return result


def __main(args):
	infiles = args.infiles
	print("Will read {} file(s).".format(len(infiles)), file=sys.stderr)
	cv_results = pd.concat((read_results_file(infile) for infile in infiles))
	print(
		"Read {} cross-validation round(s) from {} file(s) with {} column(s).".format(cv_results.shape[0], len(infiles),
																					  cv_results.shape[1]),
		file=sys.stderr)
	# noinspection PyUnresolvedReferences
	dyad_ids = tuple(sorted(cv_results["DYAD"].unique()))
	# noinspection PyTypeChecker
	anonymize_dyad_ids(cv_results, dyad_ids)
	# noinspection PyTypeChecker,PyUnresolvedReferences
	cv_results["TRAINING_SET_SIZE"] = create_training_set_size_series(cv_results, dyad_ids)
	# noinspection PyUnresolvedReferences
	cv_results["RR"] = cv_results["RANK"].transform(rr)

	# noinspection PyUnresolvedReferences
	discount_results = cv_results.groupby(["DYAD", "TRAINING_SET_SIZE"], as_index=False)
	discount_mean_ranks = discount_results.agg({"RANK": "mean", "RR": "mean"})
	print("Plotting.", file=sys.stderr)
	graph = plot_ranks(discount_mean_ranks)
	print(type(graph))

	outfile = args.outfile
	if outfile:
		print("Writing to \"{}\".".format(outfile), file=sys.stderr)
		graph.savefig(outfile, format='eps', dpi=1000)
	else:
		plt.show()


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
