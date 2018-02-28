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
from typing import Counter, DefaultDict, Iterable, Iterator, Sequence, Tuple

import nltk

from tangrams_analysis import game_utterances
from tangrams_analysis import session_data as sd
from tangrams_analysis import utterances


class ShapeNgramCounter(object):

	def __init__(self, min_order: int, max_order: int):
		if min_order > max_order:
			raise ValueError("Minimum n-gram order ({}) is greater than maximum ({}).".format(min_order, max_order))
		self.min_order = min_order
		self.max_order = max_order

	def __call__(self, session_data: Iterable[sd.SessionData]) -> DefaultDict[
		str, Counter[str]]:
		result = collections.defaultdict(collections.Counter)

		session_round_utt_df_factory = game_utterances.SessionGameRoundUtteranceSequenceFactory()
		for session_datum in session_data:
			round_utt_df = session_round_utt_df_factory(session_datum)
			ngrams = round_utt_df[session_round_utt_df_factory.UTTERANCE_SEQUENCE_COL_NAME].transform(
				lambda utts: self.__utt_ngrams(utts))
			round_utt_df["NGRAMS"] = ngrams

			referent_utts = round_utt_df.loc[round_utt_df["REFERENT"] == True]
			for row in referent_utts.itertuples(index=False):
				shape = row.SHAPE
				ngram_counts = result[shape]
				ngrams = row.NGRAMS
				ngram_counts.update(ngrams)

		return result

	def __ngrams(self, utt: utterances.Utterance) -> Iterator[Sequence[str]]:
		tokens = utt.content
		ngram_orders = range(self.min_order, self.max_order + 1)
		ngrams_by_order = (nltk.ngrams(tokens, ngram_order) for ngram_order in ngram_orders)
		return (ngram for ngram_iter in ngrams_by_order for ngram in ngram_iter)

	def __utt_ngrams(self, utts: Iterable[utterances.Utterance]) -> Tuple[Sequence[str], ...]:
		utt_ngram_iters = (self.__ngrams(utt) for utt in utts)
		return tuple(ngram for utt_ngram_iter in utt_ngram_iters for ngram in utt_ngram_iter)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Writes the n-grams used to refer to particular shapes in tabular format.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The paths to search for session data.")
	result.add_argument("--min", metavar="N", type=int, default=1, help="The minimum n-gram order to extract.",
						required=False)
	result.add_argument("--max", metavar="N", type=int, default=2, help="The maximum n-gram order to extract.",
						required=False)
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = sd.walk_session_data(inpaths)
	min_order = args.min
	max_order = args.max
	print("Extracting n-grams of order {} to {}.".format(min_order, max_order), file=sys.stderr)
	shape_ngram_counter = ShapeNgramCounter(min_order, max_order)
	shape_ngram_counts = shape_ngram_counter((session_datum for (_, session_datum) in infile_session_data))
	# https://pythonconquerstheuniverse.wordpress.com/2011/05/08/newline-conversion-in-python-3/
	writer = csv.writer(sys.stdout, dialect=csv.excel_tab, lineterminator="\n")
	writer.writerow(("SHAPE", "NGRAM", "ORDER", "COUNT"))
	for shape, ngram_counts in sorted(shape_ngram_counts.items(), key=lambda item: item[0]):
		for ngram, count in sorted(sorted(ngram_counts.items(), key=lambda item: item[0]), key=lambda item: item[1],
								   reverse=True):
			ngram_repr = " ".join(ngram)
			order = len(ngram)
			writer.writerow((shape, ngram_repr, order, count))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
