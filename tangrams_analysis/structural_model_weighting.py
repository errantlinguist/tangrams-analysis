#!/usr/bin/env python3

"""
Learns a measure of referential salience of classifiers used based on the context of their corresponding words in dialogue.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import os.path
import re
import sys
import typing
from enum import Enum, unique
from numbers import Real
from typing import Any, Iterable, Iterator, List, Mapping, Tuple

import numpy as np
import pandas as pd

import utterances

MULTIVALUE_DELIM_PATTERN = re.compile("\\s*,\\s*")

UTTS_FILENAME = "utts.tsv"

RESULTS_FILE_ENCODING = "utf-8"
RESULTS_FILE_CSV_DIALECT = csv.excel_tab


def __parse_sequence(input_str: str) -> typing.Tuple[str, ...]:
	values = MULTIVALUE_DELIM_PATTERN.split(input_str)
	return tuple(sys.intern(value) for value in values)


__RESULTS_FILE_CONVERTERS = {"REFERRING_TOKENS": __parse_sequence,
							 "OOV_TOKENS": __parse_sequence}
__RESULTS_FILE_DTYPES = {"DYAD": "category", "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}

UTTERANCE_DYAD_ID_COL_NAME = "DYAD"


@unique
class CrossValidationResultsDataColumn(Enum):
	DYAD_ID = "DYAD"
	ROUND_ID = "ROUND"
	UTTERANCES = "UTTERANCES"
	ONLY_INSTRUCTOR = "ONLY_INSTRUCTOR"
	SHAPE = "SHAPE"
	SIZE = "SIZE"
	# EDGE_COUNT = "EDGE_COUNT"
	RED = "RED"
	GREEN = "GREEN"
	BLUE = "BLUE"
	POSITION_X = "POSITION_X"
	POSITION_Y = "POSITION_Y"


class DyadRoundUtteranceFactory(object):

	@staticmethod
	def __create_utterance_from_df_row(row: pd.Series) -> utterances.Utterance:
		return utterances.Utterance(row[utterances.UtteranceTabularDataColumn.DIALOGUE_ROLE.value],
									row[utterances.UtteranceTabularDataColumn.START_TIME.value],
									row[utterances.UtteranceTabularDataColumn.END_TIME.value],
									row[utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value])

	@staticmethod
	def __find_utt_rows(dyad: str, game_round: Any, only_instr: bool, utts: pd.DataFrame) -> pd.DataFrame:
		if only_instr:
			result = utts.loc[(utts[UTTERANCE_DYAD_ID_COL_NAME] == dyad) & (
					utts[utterances.UtteranceTabularDataColumn.ROUND_ID.value] == game_round) & (
									  utts[
										  utterances.UtteranceTabularDataColumn.ROUND_ID.value] == game_round) & (
									  utts[
										  utterances.UtteranceTabularDataColumn.DIALOGUE_ROLE.value] == "INSTRUCTOR")]
		else:
			result = utts.loc[(utts[UTTERANCE_DYAD_ID_COL_NAME] == dyad) & (
					utts[utterances.UtteranceTabularDataColumn.ROUND_ID.value] == game_round) & (
									  utts[
										  utterances.UtteranceTabularDataColumn.ROUND_ID.value] == game_round)]
		return result

	def __init__(self, utts: pd.DataFrame):
		self.utts = utts
		self.utts.sort_values([UTTERANCE_DYAD_ID_COL_NAME, utterances.UtteranceTabularDataColumn.ROUND_ID.value,
							   utterances.UtteranceTabularDataColumn.START_TIME.value,
							   utterances.UtteranceTabularDataColumn.END_TIME.value], inplace=True)

	def __call__(self, row: pd.Series) -> Tuple[utterances.Utterance, ...]:
		dyad = row[CrossValidationResultsDataColumn.DYAD_ID.value]
		game_round = row[CrossValidationResultsDataColumn.ROUND_ID.value]
		only_instr = row[CrossValidationResultsDataColumn.ONLY_INSTRUCTOR.value]
		round_utt_rows = self.__find_utt_rows(dyad, game_round, only_instr, self.utts)
		assert round_utt_rows.shape[0] > 0
		utts = round_utt_rows.apply(self.__create_utterance_from_df_row, axis=1)
		return tuple(utt for utt in utts if utt.content)


class InputDatapointSequenceFactory(object):

	@staticmethod
	def __create_weighted_token_seq(tokens: Iterable[str]) -> List[Tuple[str, int]]:
		result = []
		token_iter = iter(tokens)
		current_token = next(token_iter)
		current_token_count = 1
		for next_token in token_iter:
			if next_token == current_token:
				current_token_count += 1
			else:
				result.append((current_token, current_token_count))
				current_token = next_token
				current_token_count = 1
		# Put the tail element in the list
		result.append((current_token, current_token_count))
		return result

	def __init__(self, word_feature_idxs: Mapping[str, int], dialogue_role_values: Mapping[str, Real]):
		self.word_feature_idxs = word_feature_idxs
		self.dialogue_role_values = dialogue_role_values
		self.cardinality = len(self.word_feature_idxs) + 1
		self.dialoge_role_feature_idx = len(self.word_feature_idxs)

	def __call__(self, game_round_ranking_result: Mapping[str, Any]) -> np.array:
		# shape = game_round_ranking_result[CrossValidationResultsDataColumn.SHAPE.value]
		# size = game_round_ranking_result[CrossValidationResultsDataColumn.SIZE.value]
		# red = game_round_ranking_result[CrossValidationResultsDataColumn.RED.value]
		# green = game_round_ranking_result[CrossValidationResultsDataColumn.GREEN.value]
		# blue = game_round_ranking_result[CrossValidationResultsDataColumn.BLUE.value]
		# pos_x = game_round_ranking_result[CrossValidationResultsDataColumn.POSITION_X.value]
		# pos_y = game_round_ranking_result[CrossValidationResultsDataColumn.POSITION_Y.value]
		# mid_x = distance_from_central_value(pos_x)
		# mid_y = distance_from_central_value(pos_y)
		utts = game_round_ranking_result[CrossValidationResultsDataColumn.UTTERANCES.value]
		rows = tuple(row for utt in utts for row in self.__create_word_rows(utt))
		return np.array(rows)

	def __create_word_rows(self, utt: utterances.Utterance) -> Iterator[np.array]:
		dialogue_role = utt.speaker_id
		weighted_tokens = self.__create_weighted_token_seq(utt.content)
		return (self.__create_word_row(token, weight, dialogue_role) for (token, weight) in weighted_tokens)

	def __create_word_row(self, word: str, weight: Real, dialogue_role: str) -> np.array:
		result = np.zeros(self.cardinality)
		word_feature_idx = self.word_feature_idxs[word]
		result[word_feature_idx] = weight
		diag_role_value = self.dialogue_role_values[dialogue_role]
		result[self.dialoge_role_feature_idx] = diag_role_value
		return result


def distance_from_extrema(value: float) -> float:
	"""
	Calculates the distance from the extrema for a value in the range of 0.0 to 1.0.
	:param value:  A value between 0.0 and 1.0 (inclusive).
	:return: The distance from the extrema (i.e. 0.5).
	"""
	assert value >= 0.0
	assert value <= 1.0
	result = 1.0 - abs(0.5 - value) * 2.0
	assert result >= 0.0
	assert result <= 1.0
	return result


def read_results_file(inpath: str) -> pd.DataFrame:
	print("Reading \"{}\".".format(inpath), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=RESULTS_FILE_ENCODING, memory_map=True, converters=__RESULTS_FILE_CONVERTERS,
						 dtype=__RESULTS_FILE_DTYPES)
	return result


def read_utts_file(inpath: str, utt_reader: utterances.UtteranceTabularDataReader) -> pd.DataFrame:
	dyad_id = __dyad_id(inpath)
	print("Reading utt file at \"{}\" as dyad \"{}\".".format(inpath, dyad_id), file=sys.stderr)
	result = utt_reader(inpath)
	result[UTTERANCE_DYAD_ID_COL_NAME] = dyad_id
	return result


def walk_session_uttfile_paths(inpaths: Iterable[str]) -> Iterator[str]:
	for inpath in inpaths:
		for dirpath, _, filenames in os.walk(inpath, followlinks=True):
			if UTTS_FILENAME in filenames:
				yield os.path.join(dirpath, UTTS_FILENAME)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Learns a measure of referential salience of classifiers used based on the context of their corresponding words in dialogue.")
	result.add_argument("infiles", metavar="FILE", nargs='+',
						help="The cross-validation results files to process.")
	result.add_argument("-s", "--sessions", metavar="PATH", nargs="+", required=True,
						help="Paths containing the session utterance data to read.")
	return result


def __dyad_id(infile: str) -> str:
	session_dir = os.path.dirname(infile)
	return os.path.split(session_dir)[1]


def __main(args):
	infiles = args.infiles
	print("Will read {} cross-validation results file(s).".format(len(infiles)), file=sys.stderr)
	cv_results = pd.concat((read_results_file(infile) for infile in infiles))
	# noinspection PyUnresolvedReferences
	cv_results_dyads = frozenset(cv_results[CrossValidationResultsDataColumn.DYAD_ID.value].unique())
	print("Read {} cross-validation results for {} dyad(s).".format(cv_results.shape[0], len(cv_results_dyads)),
		  file=sys.stderr)

	session_paths = args.sessions
	print("Will read session utterance data from {}.".format(session_paths), file=sys.stderr)
	uttfile_paths = tuple(walk_session_uttfile_paths(session_paths))
	utt_reader = utterances.UtteranceTabularDataReader()
	utts = pd.concat((read_utts_file(inpath, utt_reader) for inpath in uttfile_paths))
	# noinspection PyUnresolvedReferences
	utts_dyads = frozenset(utts[UTTERANCE_DYAD_ID_COL_NAME].unique())
	print("Read {} unique utterance(s) from {} file(s) for {} dyad(s), with {} column(s).".format(utts.shape[0],
																								  len(uttfile_paths),
																								  len(utts_dyads),
																								  utts.shape[1]),
		  file=sys.stderr)
	if cv_results_dyads != utts_dyads:
		raise ValueError(
			"Dyad ID sets for CV results and utterance data do not match.\nCV results: {}\nUtterance data: {}".format(
				sorted(cv_results_dyads), sorted(utts_dyads)))
	else:
		# noinspection PyTypeChecker
		round_token_seq_finder = DyadRoundUtteranceFactory(utts)
		# noinspection PyUnresolvedReferences
		cv_results[CrossValidationResultsDataColumn.UTTERANCES.value] = cv_results.apply(round_token_seq_finder, axis=1)
		vocab = frozenset(
			token for token_seq in utts[utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value] for token in token_seq)
		print("Using a vocabulary of size {}.".format(len(vocab)), file=sys.stderr)
		dialogue_role_feature_values = dict((dialogue_role, float(idx)) for (idx, dialogue_role) in enumerate(
			utts[utterances.UtteranceTabularDataColumn.DIALOGUE_ROLE.value].unique()))
		print("Found {} dialogue role(s): {}".format(len(dialogue_role_feature_values),
													 sorted(dialogue_role_feature_values.keys())), file=sys.stderr)
		input_seq_factory = InputDatapointSequenceFactory(dict((word, idx) for (idx, word) in enumerate(vocab)),
														  dialogue_role_feature_values)
		for game_round_ranking_result in cv_results.itertuples(index=False):
			# noinspection PyProtectedMember
			input_seq = input_seq_factory(game_round_ranking_result._asdict())
			print(input_seq)

		# TODO: Create output features: one feature per word class, the value thereof being the referential salience, i.e. ((STDEV of probability of "true" for all referents in round being classified) * number of times classifier has been observed in training)
		# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
		# fix random seed for reproducibility
		np.random.seed(7)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
