#!/usr/bin/env python3


"""
Writes the n-grams used to refer to particular shapes in tabular format
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import collections
import csv
import sys
from typing import Counter, DefaultDict, Iterable, Tuple

from tangrams_analysis import game_utterances
from tangrams_analysis import session_data as sd
from tangrams_analysis import utterances


def count_shape_ngrams(session_data: Iterable[sd.SessionData]) -> DefaultDict[str, Counter[str]]:
	result = collections.defaultdict(collections.Counter)

	session_round_utt_df_factory = game_utterances.SessionGameRoundUtteranceSequenceFactory()
	for session_datum in session_data:
		round_utt_df = session_round_utt_df_factory(session_datum)
		ngrams = round_utt_df[session_round_utt_df_factory.UTTERANCE_SEQUENCE_COL_NAME].transform(__utt_unigrams)
		round_utt_df["NGRAMS"] = ngrams

		referent_utts = round_utt_df.loc[round_utt_df["REFERENT"] == True]
		for row in referent_utts.itertuples(index=False):
			shape = row.SHAPE
			ngram_counts = result[shape]
			ngrams = row.NGRAMS
			ngram_counts.update(ngrams)

	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Writes the n-grams used to refer to particular shapes in tabular format.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The paths to search for session data.")
	return result


def __utt_unigrams(utts: Iterable[utterances.Utterance]) -> Tuple[str, ...]:
	return tuple(token for utt in utts for token in utt.content)


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = sd.walk_session_data(inpaths)
	shape_ngram_counts = count_shape_ngrams(session_datum for (_, session_datum) in infile_session_data)
	# https://pythonconquerstheuniverse.wordpress.com/2011/05/08/newline-conversion-in-python-3/
	writer = csv.writer(sys.stdout, dialect=csv.excel_tab, lineterminator="\n")
	writer.writerow(("SHAPE", "NGRAM", "COUNT"))
	for shape, ngram_counts in sorted(shape_ngram_counts.items(), key=lambda item: item[0]):
		for ngram, count in sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True):
			writer.writerow((shape, ngram, count))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
