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
import sys

import numpy as np
import pandas as pd
from typing import Sequence, Set, Tuple
import random

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

__RESULTS_FILE_DTYPES = {"DYAD": "category", "WORD": "category", "IS_TARGET": bool, "IS_OOV": bool,
						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}


def read_results_file(inpath: str, encoding: str) -> pd.DataFrame:
	print("Reading \"{}\".".format(inpath), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=encoding, memory_map=True,
						 dtype=__RESULTS_FILE_DTYPES)
	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Learns a measure of referential salience of classifiers used based on the context of their corresponding words in dialogue.")
	result.add_argument("infiles", metavar="FILE", nargs='+',
						help="The cross-validation results files to process.")
	result.add_argument("-e", "--encoding", metavar="CODEC", default="utf-8",
						help="The input file encoding.")
	result.add_argument("-s", "--random-seed", dest="random_seed", metavar="SEED", type=int, default=7,
						help="The random seed to use.")
	return result

def create_input_output_dfs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	input_df = df.loc[:, ["DYAD", "ROUND", "TOKEN_SEQ_ORDINALITY", "WORD", "IS_INSTRUCTOR", "IS_OOV"]]
	output_df = df.loc[:, ["DYAD", "ROUND", "TOKEN_SEQ_ORDINALITY", "PROBABILITY"]]
	return input_df, output_df

def __main(args):
	random_seed = args.random_seed
	print("Setting random seed to {}.".format(random_seed), file=sys.stderr)
	# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
	# fix random seed for reproducibility
	random.seed(random_seed)
	np.random.seed(random_seed)

	infiles = args.infiles
	encoding = args.encoding
	print("Will read {} cross-validation results file(s) using encoding \"{}\".".format(len(infiles), encoding),
		  file=sys.stderr)
	cv_results = pd.concat((read_results_file(infile, encoding) for infile in infiles))
	# noinspection PyUnresolvedReferences
	dyad_ids = tuple(sorted(frozenset(cv_results["DYAD"].unique())))
	print("Read {} cross-validation results for {} dyad(s).".format(cv_results.shape[0], len(dyad_ids)),
		  file=sys.stderr)
	# noinspection PyUnresolvedReferences
	vocab = frozenset(cv_results["WORD"].unique())
	print("Using a vocabulary of size {}.".format(len(vocab)), file=sys.stderr)



	test_set_dyad_ids = frozenset(random.sample(dyad_ids, 3))
	print("Dyads used for testing: {}".format(sorted(test_set_dyad_ids)), file=sys.stderr)

	testing_df = cv_results.loc[cv_results["DYAD"].isin(test_set_dyad_ids)]
	print("{} rows in test set.".format(testing_df.shape[0]), file=sys.stderr)
	training_df = cv_results.loc[~cv_results["DYAD"].isin(test_set_dyad_ids)]
	print("{} rows in training set.".format(training_df.shape[0]), file=sys.stderr)

	# truncate and pad input sequences
	max_review_length = 500
	#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
	embedding_vector_length = 32
	model = Sequential()
	word_embeddings = Embedding(len(vocab), embedding_vector_length, input_length=max_review_length)
	model.add(word_embeddings)
	#model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	output_dim = 1
	model.add(LSTM(output_dim))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	#https://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/

	#input_datapoint_features = ("WORD", "IS_INSTRUCTOR", "IS_OOV", "SHAPE", "RED", "GREEN", "BLUE", "POSITION_X", "POSITION_Y", "MID_X", "MID_Y")
	input_datapoint_features = ("WORD", "IS_INSTRUCTOR", "IS_OOV")
	output_datapoint_features = ("PROBABILITY",)




	# TODO: Create output features: one feature per word class, the value thereof being the referential salience, i.e. ((STDEV of probability of "true" for all referents in round being classified) * number of times classifier has been observed in training)



if __name__ == "__main__":
	__main(__create_argparser().parse_args())
