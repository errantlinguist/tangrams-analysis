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
import keras.preprocessing.sequence

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
__RESULTS_FILE_DTYPES = {"DYAD": "category", "WORD": "category", "IS_TARGET": bool, "IS_OOV": bool,
				 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}

def read_results_file(inpath: str, encoding: str) -> pd.DataFrame:
	print("Reading \"{}\" using encoding \"{}\".".format(inpath, encoding), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=encoding, memory_map=True, dtype=__RESULTS_FILE_DTYPES)
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


def create_token_seq_start_datapoint(sequence: pd.DataFrame) -> pd.Series:
	# Sanity check
	assert len(sequence["POSITION_X"].unique()) == 1
	first_token = sequence.loc[sequence["TOKEN_SEQ_ORDINALITY"].idxmin()]
	result = pd.Series(first_token, copy=True)
	result["TOKEN_SEQ_ORDINALITY"] = result["TOKEN_SEQ_ORDINALITY"] - 1
	assert result["TOKEN_SEQ_ORDINALITY"] != first_token["TOKEN_SEQ_ORDINALITY"]
	print(result)
	return result

def create_token_sequences(df : pd.DataFrame) -> Tuple[np.array, np.array]:
	"""
	Creates a sequence of sequences of tokens, each representing an utterance, each of which thus causes an "interruption" in the chain
	so that e.g. the first token of one utterance is not learned as dependent on the last token of the utterance preceding it.
	:param df: The DataFrame to process.
	:return:
	"""
	# https://stackoverflow.com/a/47815400/1391325
	df.sort_values("TOKEN_SEQ_ORDINALITY", inplace=True)
	sequences = df.groupby(("CROSS_VALIDATION_ITER", "DYAD", "ROUND", "UTT_START_TIME", "UTT_END_TIME", "ENTITY"))
	#return sequences['WORD', 'PROBABILITY'].apply(lambda group : group.values.tolist()).values
	words = sequences['WORD'].apply(lambda group : group.values).values
	scores = sequences["PROBABILITY"].apply(lambda group : group.values).values
	assert len(words) == len(scores)
	return words, scores


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

	print("Reading token sequences.", file=sys.stderr)
	words, scores = create_token_sequences(cv_results)
	print(words)
	max_seq_len = max(len(seq) for seq in words)
	#max_seq_len= max(len(seq) for seq in sequences)
	print("Found {} token sequences, with a maximum sequence length of {}.".format(len(words), max_seq_len), file=sys.stderr)
	print("Padding sequences to a maximum sequence length of {}.".format(max_seq_len),
		  file=sys.stderr)
	#padded_seqs = keras.preprocessing.sequence.pad_sequences(words)
	#print("Padded sequence count: {}".format(len(padded_seqs)))

	#longest_seq = max(sequences, key=len)
	#for seq in sorted(sequences, key=len, reverse=True):
	#	print(seq)

	#seq_lengths = tuple(len(seq) for seq in sequences)
	#longest_seq_idx = np.amax(seq_lengths)
	#longest_seq = sequences[longest_seq_idx]
	#print(longest_seq)
	#for seq in sequences:
	#	print(len(seq))

	#test_set_dyad_ids = frozenset(random.sample(dyad_ids, 3))
	#print("Dyads used for testing: {}".format(sorted(test_set_dyad_ids)), file=sys.stderr)
	#testing_df = cv_results.loc[cv_results["DYAD"].isin(test_set_dyad_ids)]
	#print("{} rows in test set.".format(testing_df.shape[0]), file=sys.stderr)
	#training_df = cv_results.loc[~cv_results["DYAD"].isin(test_set_dyad_ids)]
	#print("{} rows in training set.".format(training_df.shape[0]), file=sys.stderr)

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
	#print(model.summary())

	# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
	#from keras.preprocessing.text import Tokenizer
	# create the tokenizer
	#t = Tokenizer()
	# fit the tokenizer on the documents
	#t.fit_on_texts(docs)

	#https://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/

	#input_datapoint_features = ("WORD", "IS_INSTRUCTOR", "IS_OOV", "SHAPE", "RED", "GREEN", "BLUE", "POSITION_X", "POSITION_Y", "MID_X", "MID_Y")
	input_datapoint_features = ("WORD", "IS_INSTRUCTOR", "IS_OOV")
	output_datapoint_features = ("PROBABILITY",)




	# TODO: Create output features: one feature per word class, the value thereof being the referential salience, i.e. ((STDEV of probability of "true" for all referents in round being classified) * number of times classifier has been observed in training)



if __name__ == "__main__":
	__main(__create_argparser().parse_args())
