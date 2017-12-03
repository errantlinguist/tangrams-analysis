#!/usr/bin/env python3

"""
Correlates keyword TF-IDF score with the cross-validation ranks for rounds in which the given keywords were used.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import re
import sys
from typing import Sequence, Tuple

import pandas as pd

CV_RESULTS_FILE_CSV_DIALECT = csv.excel_tab
CV_RESULTS_FILE_ENCODING = "utf-8"
KEYWORDS_FILE_CSV_DIALECT = csv.excel_tab
KEYWORDS_FILE_ENCODING = "utf-8"


class TokenSequenceFactory(object):
	_TOKEN_DELIMITER_PATTERN = re.compile("\\s+")

	def __init__(self):
		self.token_seq_singletons = {}

	def __call__(self, token_str: str) -> Tuple[str]:
		content = self._TOKEN_DELIMITER_PATTERN.split(token_str)
		if content:
			try:
				result = self.token_seq_singletons[content]
			except KeyError:
				result = tuple(sys.intern(token) for token in content)
				self.token_seq_singletons[result] = result
		else:
			result = None

		return result


__TOKEN_SEQ_FACTORY = TokenSequenceFactory()
__CV_RESULTS_FILE_CONVERTERS = {"REFERRING_TOKENS": __TOKEN_SEQ_FACTORY, "REFERRING_TOKEN_TYPES": __TOKEN_SEQ_FACTORY}


def keyword_score(tokens: Sequence[str]):
	pass


def read_csv_results_file(infile: str) -> pd.DataFrame:
	return pd.read_csv(infile, dialect=CV_RESULTS_FILE_CSV_DIALECT, sep=CV_RESULTS_FILE_CSV_DIALECT.delimiter,
					   float_precision="round_trip",
					   encoding=CV_RESULTS_FILE_ENCODING, memory_map=True, converters=__CV_RESULTS_FILE_CONVERTERS)


def read_keyword_scores(keywords_file: str) -> pd.DataFrame:
	return pd.read_csv(keywords_file, dialect=KEYWORDS_FILE_CSV_DIALECT, sep=KEYWORDS_FILE_CSV_DIALECT.delimiter,
					   float_precision="round_trip",
					   encoding=KEYWORDS_FILE_ENCODING, memory_map=True)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Correlates keyword TF-IDF score with the cross-validation ranks for rounds in which the given keywords were used.")
	result.add_argument("infile", metavar="PATH",
						help="The cross-validation result file to read.")
	result.add_argument("-k", "--keywords", metavar="PATH",
						help="The keyword TF-IDF score file.")
	return result


def __main(args):
	infile = args.infile
	print("Reading cross-validation ranks from \"{}\".".format(infile), file=sys.stderr)
	cv_results = read_csv_results_file(infile)
	print("Cross-validation results dataframe shape: {}".format(cv_results.shape), file=sys.stderr)
	cv_sessions = frozenset(cv_results["DYAD"].unique())
	print("Read results for {} session(s).".format(len(cv_sessions)), file=sys.stderr)

	keywords_file = args.keywords
	print("Reading keyword scores from \"{}\".".format(keywords_file), file=sys.stderr)
	session_keyword_scores = read_keyword_scores(keywords_file)
	keyword_sessions = frozenset(session_keyword_scores["SESSION"].unique())
	print("Read keyword scores for {} session(s).".format(len(keyword_sessions)), file=sys.stderr)
	if keyword_sessions == cv_sessions:
		pass
	# TODO: Finish
	else:
		raise ValueError("Set of sessions for keywords is not equal to that for cross-validation results.")


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
