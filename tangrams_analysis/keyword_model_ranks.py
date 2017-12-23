#!/usr/bin/env python3

"""
Correlates keyword TF-IDF score with the cross-validation ranks for rounds in which the given keywords were used.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import logging
import os.path
import re
import sys
from enum import Enum, unique
from numbers import Number
from typing import Any, Iterable, Iterator, Tuple

import pandas as pd
from nltk.util import ngrams as nltk_ngrams

import session_data as sd
import utterances

UTTERANCE_DYAD_ID = "DYAD"


@unique
class CrossValidationResultsDataColumn(Enum):
	DYAD_ID = "DYAD"
	ROUND_ID = "ROUND"
	TOKEN_SEQS = "TOKEN_SEQS"


class KeywordScorer(object):

	def __init__(self, session_keyword_scores: pd.DataFrame):
		self.session_keyword_scores = session_keyword_scores
		self.score_cache = {}

	def __call__(self, row: pd.Series):
		dyad_id = row[CrossValidationResultsDataColumn.DYAD_ID.value]
		token_seqs = row[CrossValidationResultsDataColumn.TOKEN_SEQS.value]
		token_seq_scores = (self.__fetch_scores(seq, dyad_id) for seq in token_seqs)

	# tokens = row["REFERRING_TOKENS"]
	# TODO: Finish
	# return self.__score_keywords(tokens, dyad_id)

	def __calculate_score(self, ngram: Tuple[str, ...], dyad_id: str) -> Number:
		dyad_keyword_scores = self.session_keyword_scores.loc[
			((self.session_keyword_scores["SESSION"] == dyad_id) & (self.session_keyword_scores["NGRAM"] == ngram))]
		assert dyad_keyword_scores.shape[0] < 2
		if dyad_keyword_scores.shape[0] < 1:
			logging.warning("No score for ngram: %s; Session: %s", ngram, dyad_id)
			result = 0
		else:
			result = dyad_keyword_scores.iloc[0]
		return result

	def __fetch_scores(self, tokens: Tuple[str, ...], dyad_id: str) -> Iterator[Number]:
		ngrams_by_length = (nltk_ngrams(tokens, n) for n in range(1, len(tokens)))
		ngrams = (ngram for length_list in ngrams_by_length for ngram in length_list)
		return (self.__fetch_score(ngram, dyad_id) for ngram in ngrams)

	def __fetch_score(self, ngram: Tuple[str, ...], dyad_id: str) -> Number:
		key = (ngram, dyad_id)
		try:
			result = self.score_cache[key]
		except KeyError:
			result = self.__calculate_score(ngram, dyad_id)
			self.score_cache[key] = result
		return result


class TokenSequenceFactory(object):
	_TOKEN_DELIMITER_PATTERN = re.compile("(?:,\\s*?)|(?:\\s+)")

	def __init__(self):
		self.token_seq_singletons = {}

	def __call__(self, token_str: str) -> Tuple[str, ...]:
		content = tuple(self._TOKEN_DELIMITER_PATTERN.split(token_str))
		if content:
			try:
				result = self.token_seq_singletons[content]
			except KeyError:
				result = tuple(sys.intern(token) for token in content)
				self.token_seq_singletons[result] = result
		else:
			result = None

		return result


class RoundUtteranceTokenSequenceJoiner(object):
	def __init__(self, session_utts: pd.DataFrame, only_instructor: bool = True):
		self.session_utts = session_utts
		self.__session_dyad_token_seq_getter = self.__dyad_round_token_seqs_only_instructor if only_instructor else self.__dyad_round_token_seqs

	def __call__(self, row: pd.Series) -> Tuple[Any, ...]:
		dyad_to_match = row[CrossValidationResultsDataColumn.DYAD_ID.value]
		round_to_match = row[CrossValidationResultsDataColumn.ROUND_ID.value]
		dyad_round_token_seqs = self.__session_dyad_token_seq_getter(dyad_to_match, round_to_match)
		logging.debug("Found %d utterance(s) for round %d of session \"%s\".", len(dyad_round_token_seqs),
					  round_to_match,
					  dyad_to_match)
		return tuple(seq for seq in dyad_round_token_seqs if seq)

	def __dyad_round_token_seqs(self, dyad_to_match: str, round_to_match: Any) -> pd.Series:
		return self.session_utts.loc[
			(self.session_utts[UTTERANCE_DYAD_ID] == dyad_to_match) & (
					self.session_utts[
						utterances.UtteranceTabularDataColumn.ROUND_ID.value] == round_to_match), utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value]

	def __dyad_round_token_seqs_only_instructor(self, dyad_to_match: str, round_to_match: Any) -> pd.Series:
		return self.session_utts.loc[
			(self.session_utts[UTTERANCE_DYAD_ID] == dyad_to_match) & (
					self.session_utts[utterances.UtteranceTabularDataColumn.ROUND_ID.value] == round_to_match) & (
					self.session_utts[
						"DIALOGUE_ROLE"] == "INSTRUCTOR"), utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value]


_TOKEN_SEQ_FACTORY = TokenSequenceFactory()


class TabularDataFileDatum(object):
	def __init__(self, csv_dialect, encoding: str, converters):
		self.csv_dialect = csv_dialect
		self.encoding = encoding
		self.converters = converters

	def read_df(self, inpath: str):
		return pd.read_csv(inpath, dialect=self.csv_dialect, sep=self.csv_dialect.delimiter,
						   float_precision="round_trip",
						   encoding=self.encoding, memory_map=True, converters=self.converters)


@unique
class TabularDataFile(Enum):
	CV_RESULT = TabularDataFileDatum(csv.excel_tab, "utf-8", {"REFERRING_TOKENS": _TOKEN_SEQ_FACTORY,
															  "REFERRING_TOKEN_TYPES": _TOKEN_SEQ_FACTORY})
	KEYWORDS = TabularDataFileDatum(csv.excel_tab, "utf-8", {"NGRAM": _TOKEN_SEQ_FACTORY})
	UTTS = TabularDataFileDatum(csv.excel_tab, "utf-8",
								{utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value: _TOKEN_SEQ_FACTORY})


def read_utts_dfs(session_paths: Iterable[str]) -> Iterator[pd.DataFrame]:
	print("Reading utterances from {}.".format(session_paths), file=sys.stderr)
	session_dirs = tuple(sd.walk_session_dirs(session_paths))
	session_parent_dirs = (os.path.dirname(session_dir) for session_dir in session_dirs)
	common_session_id_prefix = os.path.commonpath(session_parent_dirs) + os.path.sep
	logging.debug("Common parent path for all sessions is \"%s\".", common_session_id_prefix)
	print("Reading utterances for {} session(s).".format(len(session_dirs)), file=sys.stderr)
	return (__read_utts_df(session_dir, common_session_id_prefix) for session_dir in session_dirs)


def __read_utts_df(session_dir: str, common_session_id_prefix: str) -> pd.DataFrame:
	dyad_id = session_dir[len(common_session_id_prefix):] if session_dir.startswith(
		common_session_id_prefix) else session_dir
	utts_file = os.path.join(session_dir, "utts.tsv")
	result = TabularDataFile.UTTS.value.read_df(utts_file)
	result[UTTERANCE_DYAD_ID] = dyad_id
	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Correlates keyword TF-IDF score with the cross-validation ranks for rounds in which the given keywords were used.")
	result.add_argument("sessions", metavar="PATH", nargs='+',
						help="Paths(s) containing session directories to use for getting utterance information.")
	result.add_argument("-r", "--results", metavar="PATH", required=True,
						help="The cross-validation results file to read.")
	result.add_argument("-k", "--keywords", metavar="PATH", required=True,
						help="The keyword TF-IDF score file.")
	return result


def __main(args):
	cv_results_infile = args.results
	print("Reading cross-validation ranks from \"{}\".".format(cv_results_infile), file=sys.stderr)
	cv_results = TabularDataFile.CV_RESULT.value.read_df(cv_results_infile)
	logging.debug("Cross-validation results dataframe shape: %s", cv_results.shape)
	cv_sessions = frozenset(cv_results["DYAD"].unique())
	print("Read results for {} session(s).".format(len(cv_sessions)), file=sys.stderr)

	keywords_file = args.keywords
	print("Reading keyword scores from \"{}\".".format(keywords_file), file=sys.stderr)
	session_keyword_scores = TabularDataFile.KEYWORDS.value.read_df(keywords_file)
	keyword_sessions = frozenset(session_keyword_scores["SESSION"].unique())
	print("Read keyword scores for {} session(s).".format(len(keyword_sessions)), file=sys.stderr)
	if keyword_sessions != cv_sessions:
		raise ValueError("Set of sessions for keywords is not equal to that for cross-validation results.")
	else:
		session_paths = args.sessions
		session_utts = pd.concat(read_utts_dfs(session_paths))
		# noinspection PyUnresolvedReferences
		logging.debug("Session utterances dataframe shape: %s", session_utts.shape)
		# noinspection PyUnresolvedReferences
		utt_session_set = frozenset(session_utts[UTTERANCE_DYAD_ID].unique())
		if utt_session_set != cv_sessions:
			raise ValueError("Set of sessions for utterances is not equal to that for cross-validation results.")
		else:
			# noinspection PyUnresolvedReferences
			session_utts[utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value] = session_utts[
				utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value].transform(lambda token_seq: tuple(
				token for token in token_seq if utterances.is_semantically_relevant_token(token)))
			# noinspection PyTypeChecker
			cv_results[CrossValidationResultsDataColumn.TOKEN_SEQS.value] = cv_results.apply(
				RoundUtteranceTokenSequenceJoiner(session_utts), axis=1)
			cv_results["TF-IDF"] = cv_results.apply(KeywordScorer(session_keyword_scores), axis=1)
			print(cv_results["TOKEN_SEQS"])


# TODO: Finish


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
