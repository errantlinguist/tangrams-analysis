#!/usr/bin/env python3

"""
Creates a heatmap of vocabulary used in different sessions.

Use with e.g. "find ~/Documents/Projects/Tangrams/Data/Ready/ -iname "utts.tsv" -exec ./vocabulary_heatmap.py {} +"
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import os
import sys
from collections import Counter
from enum import Enum, unique

import pandas as pd

import utterances


@unique
class SessionVocabularyCountDataColumn(Enum):
	DYAD_ID = "DYAD"
	TOKEN_TYPE = "TOKEN_TYPE"
	TOKEN_COUNT = "TOKEN_COUNT"


__SESSION_WORD_COUNT_DF_COLS = (
	SessionVocabularyCountDataColumn.DYAD_ID.value, SessionVocabularyCountDataColumn.TOKEN_TYPE.value,
	SessionVocabularyCountDataColumn.TOKEN_COUNT.value)
# noinspection PyTypeChecker
assert len(__SESSION_WORD_COUNT_DF_COLS) == len(SessionVocabularyCountDataColumn)


def create_session_word_count_df(utts: pd.DataFrame) -> pd.DataFrame:
	session_utts = utts.groupby(SessionVocabularyCountDataColumn.DYAD_ID.value)

	rows = []
	for session, utts in session_utts:
		token_type_counts = Counter()
		for row in utts.itertuples(index=False):
			# noinspection PyProtectedMember
			row_as_dict = row._asdict()
			token_seq = row_as_dict[utterances.UtteranceTabularDataColumn.TOKEN_SEQ.value]
			token_type_counts.update(token_seq)

		session_rows = ((session, token_type, count) for token_type, count in token_type_counts.items())
		rows.extend(session_rows)

	return pd.DataFrame(data=rows, columns=__SESSION_WORD_COUNT_DF_COLS)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Visualizes usage of vocabulary across sessions as a heatmap.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The files to process.")

	return result


def __dyad_id(infile: str) -> str:
	session_dir = os.path.dirname(infile)
	return os.path.split(session_dir)[1]


def __read_utts_file(inpath: str, utt_reader: utterances.UtteranceTabularDataReader) -> pd.DataFrame:
	dyad_id = __dyad_id(inpath)
	print("Reading utt file at \"{}\" as dyad \"{}\"".format(inpath, dyad_id), file=sys.stderr)
	result = utt_reader(inpath)
	result[SessionVocabularyCountDataColumn.DYAD_ID.value] = dyad_id
	return result


def __main(args):
	inpaths = args.inpaths
	print("Will read {} file(s).".format(len(inpaths)), file=sys.stderr)
	utt_reader = utterances.UtteranceTabularDataReader()
	utts = pd.concat((__read_utts_file(inpath, utt_reader) for inpath in inpaths))
	print("Read {} unique utterance(s) from {} file(s) with {} column(s).".format(utts.shape[0], len(inpaths),
																				  utts.shape[1]), file=sys.stderr)
	session_word_count_df = create_session_word_count_df(utts)
	session_word_count_df.sort_values(
		[SessionVocabularyCountDataColumn.DYAD_ID.value, SessionVocabularyCountDataColumn.TOKEN_TYPE.value],
		inplace=True)
	print(session_word_count_df)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
