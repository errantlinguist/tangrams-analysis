#!/usr/bin/env python3


"""
Writes mean round count data in tabular format.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import statistics
import sys
from typing import Iterable, Sequence

import pandas as pd

from tangrams_analysis import game_utterances
from tangrams_analysis import session_data as sd


class RoundCounter(object):

	@staticmethod
	def __count_utts(session_datum: sd.SessionData,
					 session_round_utt_df_factory: game_utterances.SessionGameRoundUtteranceSequenceFactory) -> pd.Series:
		round_utt_df = session_round_utt_df_factory(session_datum)
		# Just use the "true" referent rows for extra safety in case something weird happens with the counts otherwise
		round_utt_df = round_utt_df.loc[round_utt_df["REFERENT"] == True]
		utt_counts = round_utt_df[session_round_utt_df_factory.UTTERANCE_SEQUENCE_COL_NAME].transform(
			lambda utts: len(utts))
		return utt_counts

	def __init__(self):
		pass

	def __call__(self, session_data: Iterable[sd.SessionData]) -> Sequence[int]:
		result = []
		session_round_utt_df_factory = game_utterances.SessionGameRoundUtteranceSequenceFactory()
		for session_datum in session_data:
			round_utt_df = session_round_utt_df_factory(session_datum)
			round_count = round_utt_df[game_utterances.EventColumn.ROUND_ID.value].nunique()
			result.append(round_count)
		return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Writes mean round count data in tabular format.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The paths to search for session data.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = sd.walk_session_data(inpaths)
	counter = RoundCounter()
	counts = counter((session_datum for (_, session_datum) in infile_session_data))

	# https://pythonconquerstheuniverse.wordpress.com/2011/05/08/newline-conversion-in-python-3/
	writer = csv.writer(sys.stdout, dialect=csv.excel_tab, lineterminator="\n")
	writer.writerow(("DESC", "VALUE"))
	writer.writerow(("Total dialogues", len(counts)))
	writer.writerow(("Mean rounds per dialogue", statistics.mean(counts)))
	writer.writerow(("Median rounds per dialogue", statistics.median(counts)))
	writer.writerow(("Stdev", statistics.stdev(counts)))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
