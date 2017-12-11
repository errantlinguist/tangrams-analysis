#!/usr/bin/env python3
"""
A script for estimating how well the combined decisions of the classifiers used for ranking referents in a given tangrams round matches the gold standard.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"
__since__ = "2017-12-06"

import argparse
import csv
import sys
from numbers import Integral, Real
from typing import Iterable, Tuple

import pandas as pd

INFILE_CSV_DIALECT = csv.excel_tab
INFILE_ENCODING = 'utf-8'

_INFILE_DTYPES = {"DYAD": "category", "ENTITY": "category", "WORD": "category", "IS_OOV": bool, "SHAPE": "category",
				  "TARGET": bool}


def create_word_metric_row(word: str, confidences: pd.DataFrame) -> Tuple[str, Real, Real, Real, Real, Integral]:
	positive_ex_count = confidences.loc[(confidences["WORD"] == word) & confidences["TARGET"] == True].shape[0]
	negative_ex_count = confidences.loc[(confidences["WORD"] == word) & confidences["TARGET"] != True].shape[0]
	positive_obs_weight = float(negative_ex_count) / positive_ex_count
	word_model_ref_scores = confidences.loc[(confidences["WORD"] == word)]
	probs_of_being_correct = word_model_ref_scores.apply(lambda row: __prob_of_being_correct(row, positive_obs_weight),
														 axis=1)
	perplexity = word_model_perplexity(probs_of_being_correct)
	observation_count = word_model_ref_scores.shape[0]
	return word, probs_of_being_correct.mean(), probs_of_being_correct.var(), probs_of_being_correct.std(), perplexity, observation_count,


def read_ref_confidence_results_file(inpath: str) -> pd.DataFrame:
	return pd.read_csv(inpath, encoding=INFILE_ENCODING, dialect=INFILE_CSV_DIALECT, sep=INFILE_CSV_DIALECT.delimiter,
					   float_precision="round_trip", memory_map=True, dtype=_INFILE_DTYPES)


def read_ref_confidence_results_files(inpaths: Iterable[str]) -> pd.DataFrame:
	dfs = (read_ref_confidence_results_file(inpath) for inpath in inpaths)
	# noinspection PyTypeChecker
	return pd.concat(dfs)


def word_model_perplexity(probs_of_being_correct: pd.Series) -> Real:
	log_probs = probs_of_being_correct.transform("log2")
	log_prob_sum = log_probs.sum()
	normalized_log_prob_sum = - 1 / len(log_probs) * log_prob_sum
	return 2 ** normalized_log_prob_sum


def __prob_of_being_correct(row: pd.Series, positive_obs_weight: Real) -> Real:
	is_target = row["TARGET"]
	confidence_score = row["CONFIDENCE"]
	return confidence_score * positive_obs_weight if is_target else 1.0 - confidence_score


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Estimates the goodness of words-as-classifiers models.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The referent confidence result file(s) to process.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Will read {} path(s).".format(len(inpaths)), file=sys.stderr)
	confidences = read_ref_confidence_results_files(inpaths)
	print("Read {} dataframe row(s).".format(confidences.shape[0]), file=sys.stderr)
	words = confidences["WORD"].unique()
	print("Found {} unique word(s) (token type(s)).".format(len(words)), file=sys.stderr)
	word_metrics = pd.DataFrame(
		columns=("WORD", "MEAN_PROB", "PROB_VARIANCE", "PROB_STDEV", "PERPLEXITY", "OBSERVATION_COUNT"),
		data=(create_word_metric_row(word, confidences) for word in words))
	word_metrics.sort_values(by="PERPLEXITY", inplace=True)
	word_metrics.to_csv(sys.stdout, encoding=INFILE_ENCODING, sep=INFILE_CSV_DIALECT.delimiter, index=False)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
