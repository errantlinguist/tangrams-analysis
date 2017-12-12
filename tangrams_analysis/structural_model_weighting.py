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
from typing import Any, Iterable, Iterator, Sequence, Tuple

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
	TOKEN_SEQS = "TOKEN_SEQS"
	ONLY_INSTRUCTOR = "ONLY_INSTRUCTOR"


class DyadRoundTokenSequenceFinder(object):

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

	def __call__(self, row: pd.Series) -> Tuple[Sequence[str], ...]:
		dyad = row[CrossValidationResultsDataColumn.DYAD_ID.value]
		game_round = row[CrossValidationResultsDataColumn.ROUND_ID.value]
		only_instr = row[CrossValidationResultsDataColumn.ONLY_INSTRUCTOR.value]
		round_utt_rows = self.__find_utt_rows(dyad, game_round, only_instr, self.utts)
		assert round_utt_rows.shape[0] > 0

		return tuple(token_seq for token_seq in round_utt_rows[utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value] if token_seq)


def read_results_file(inpath: str) -> pd.DataFrame:
	print("Reading \"{}\".".format(inpath), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=RESULTS_FILE_ENCODING, memory_map=True, converters=__RESULTS_FILE_CONVERTERS,
						 dtype=__RESULTS_FILE_DTYPES)
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


def __read_utts_file(inpath: str, utt_reader: utterances.UtteranceTabularDataReader) -> pd.DataFrame:
	dyad_id = __dyad_id(inpath)
	print("Reading utt file at \"{}\" as dyad \"{}\".".format(inpath, dyad_id), file=sys.stderr)
	result = utt_reader(inpath)
	result[UTTERANCE_DYAD_ID_COL_NAME] = dyad_id
	return result


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
	utts = pd.concat((__read_utts_file(inpath, utt_reader) for inpath in uttfile_paths))
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
		round_token_seq_finder = DyadRoundTokenSequenceFinder(utts)
		# noinspection PyUnresolvedReferences
		cv_results[CrossValidationResultsDataColumn.TOKEN_SEQS.value] = cv_results.apply(round_token_seq_finder, axis=1)
		print(cv_results[CrossValidationResultsDataColumn.TOKEN_SEQS.value])


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
