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

# Matplotlib for additional customization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import cm

from tangrams_analysis import natural_keys

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
#__RESULTS_FILE_DTYPES = {"DYAD": "category", "IS_TARGET": bool, "IS_OOV": bool,
#						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}
__RESULTS_FILE_DTYPES = {"cond" : "category", "sess" : "category", "round" : int, "rank" :int, "mrr" : float, "accuracy" : int}

def plot_heatmap(cv_results: pd.DataFrame):
	# print(cv_results["BACKGROUND_DATA_WORD_TOKEN_COUNT"].dtype)
	# print(cv_results["INTERACTION_DATA_WORD_TOKEN_COUNT"].dtype)
	# https://stackoverflow.com/a/28652153/1391325
	# dataset_size_table = cv_results.pivot_table(
	#	values="RR",
	#		index="BACKGROUND_DATA_WORD_TOKEN_COUNT",
	#		columns="INTERACTION_DATA_WORD_TOKEN_COUNT", aggfunc='mean').fillna(0.0)
	#	print(dataset_size_table)
	# Draw a heatmap with the numeric values in each cell
	# f, ax = plt.subplots(figsize=(9, 6))
	# with sns.axes_style("white"):
	# result = sns.heatmap(dataset_size_table, cmap=cm.magma_r)
	# result = sns.kdeplot(dataset_size_table, cmap=cm.magma_r, kind='hex', stat_func="mean")
	means = cv_results.groupby(("BACKGROUND_DATA_WORD_TOKEN_COUNT", "INTERACTION_DATA_WORD_TOKEN_COUNT"))["RANK"].mean()
	print(means)
	result = sns.heatmap(means, cmap=cm.magma_r)
	# result = plt.hexbin(cv_results["BACKGROUND_DATA_WORD_TOKEN_COUNT"], cv_results["INTERACTION_DATA_WORD_TOKEN_COUNT"], cv_results["RR"])
	# result.set(xlabel='Background', ylabel='Interaction')
	# result.set(xlabel='x', ylabel='y')
	# result.invert_yaxis()
	return result


def plot_rounds(cv_results: pd.DataFrame, ci=95) -> sns.axisgrid.FacetGrid:
	#fig = sns.lmplot(x="ROUND", y="RR", hue="UPDATE_WEIGHT", data=cv_results, scatter_kws={"s": 10}, markers=["o", "x"])
	#hue_group_kws = {"marker" : ["^", "v"], 'color': ['C0', 'k'], "ls": ["-", "--"]}
	#hue_kws = {"marker": ["^", "v"], "line_kws" : [{"linestyle" : "-"}, {"linestyle" : "--"}]}
	# result = sns.lmplot("round", "RR", cv_results, hue="cond")
	result = sns.FacetGrid(cv_results, hue="cond", sharex=True, sharey=True, legend_out=True, palette="colorblind")

	# Draw the regression plot on each facet
	#regplot_kws = dict(
	#	x_estimator=x_estimator, x_bins=x_bins, x_ci=x_ci,
	#	scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units,
	#	order=order, logistic=logistic, lowess=lowess, robust=robust,
	#	logx=logx, x_partial=x_partial, y_partial=y_partial, truncate=truncate,
	#	x_jitter=x_jitter, y_jitter=y_jitter,
	#	scatter_kws=scatter_kws, line_kws=line_kws,
	#)
	#confounding_var = "sess"
	#print(cv_results)
	#result.map_dataframe(sns.pointplot,  "round", "RR", scatter_kws={"s": 10}, robust=False, ci=ci)
	result.map_dataframe(sns.regplot, "round", "RR", scatter_kws={"s": 10}, n_boot=1000, robust=False, ci=ci, x_estimator=np.mean)

	# https://stackoverflow.com/a/47407428/1391325
	# Use lmplot to plot scatter points
	#result = sns.lmplot(x="ROUND", y="RR", hue="UPDATE_WEIGHT", data=cv_results,
	#				   fit_reg=False)
	# Use regplot to plot the regression line for the whole points
	#sns.regplot(x="ROUND", y="RR", data=cv_results, scatter=False,
	#			ax=result.axes[0, 0])
	#result.set_axis_labels("Round", "RR")
	result.set_axis_labels("Round", "RR")
	result.add_legend(title="Condition")
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
		description="Writes only certain dyads from a word score file.")
	result.add_argument("infiles", metavar="FILE", nargs='+',
						help="The cross-validation results files to process.")
	result.add_argument("-e", "--encoding", metavar="CODEC", default="utf-8",
						help="The input file encoding.")
	result.add_argument("-p", "--pattern", metavar="REGEX", type=re.compile, default=re.compile(".+\.[ct]sv"),
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
	print("Read results for {} dyads.".format(cv_results["sess"].nunique()), file=sys.stderr)
	#print("Dyads present: {}".format(sorted(cv_results["DYAD"].unique(), key=natural_keys)), file=sys.stderr)
	#print("Unique discounting values: {}".format(sorted(cv_results["DISCOUNT"].unique())), file=sys.stderr)
	#print("Unique updating weights: {}".format(sorted(cv_results["UPDATE_WEIGHT"].unique())), file=sys.stderr)
	cv_results["RR"] = 1.0 / cv_results["rank"]
	# groups = cv_results.groupby(("ROUND", "UPDATE_WEIGHT"))
	# print(groups["RANK"].mean())

	with sns.plotting_context("paper"):
		sns.set(style="whitegrid", font="Times New Roman")
		#fig = sns.lmplot(x="ROUND", y="RR", hue="UPDATE_WEIGHT", data=cv_results, scatter_kws={"s": 10}, markers=["o", "x"])
		fig = plot_rounds(cv_results)
	# cv_results["DATA_RATIO"] = cv_results["INTERACTION_DATA_WORD_TOKEN_COUNT"] / cv_results["BACKGROUND_DATA_WORD_TOKEN_COUNT"]
	# fig = sns.lmplot(x="DATA_RATIO", y="RR", hue="UPDATE_WEIGHT", data=cv_results)

	# sns_plot = plot_heatmap(cv_results)
	# https://stackoverflow.com/a/39482402/1391325
	# fig = sns_plot.get_figure()

	outfile = args.outfile
	if outfile:
		output_format = __parse_format(outfile)
		print("Writing to \"{}\" as format \"{}\".".format(outfile, output_format), file=sys.stderr)
		fig.savefig(outfile, format=output_format, dpi=1000)
	else:
		plt.show()


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
