#!/usr/bin/env python3

"""
Learns a measure of referential salience of classifiers used based on the context of their corresponding words in dialogue.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import itertools
import math
import random
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
__RESULTS_FILE_DTYPES = {"DYAD": "category", "WORD": "category", "IS_TARGET": bool, "IS_OOV": bool,
						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}


class OneHotTokenEncodedTokenSequenceFactory(object):

	def __init__(self, onehot_encoder: OneHotEncoder, max_len: int, padding_integer_label: int):
		self.onehot_encoder = onehot_encoder
		self.__max_len = max_len
		self.__max_len_divisor = float(max_len)
		self.padding_integer_label = padding_integer_label

	def __call__(self, df: pd.DataFrame) -> Tuple[List[np.array], List[np.array]]:
		"""
		Creates a sequence of sequences of tokens, each representing an utterance, each of which thus causes an "interruption" in the chain
		so that e.g. the first token of one utterance is not learned as dependent on the last token of the utterance preceding it.
		:param df: The DataFrame to process.
		:return: Paired lists of 2D numpy arrays, each representing a sequence of datapoints which represents an utterance and the corresponding scores to predict.
		"""
		# https://stackoverflow.com/a/47815400/1391325
		df.sort_values("TOKEN_SEQ_ORDINALITY", inplace=True)
		sequences = df.groupby(("CROSS_VALIDATION_ITER", "DYAD", "ROUND", "UTT_START_TIME", "UTT_END_TIME", "ENTITY"),
							   as_index=False)

		# binary encode <https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/>
		# First fit the one-hot encoder on all data before processing each utterance group individually
		integer_labels = df["WORD_LABEL"].values
		reshaped_integer_labels = integer_labels.reshape(len(integer_labels), 1)
		self.onehot_encoder.fit(reshaped_integer_labels)
		# If the integer is not found after fitting, this method will throw a ValueError exception
		onehot_padding_integer_label_array = self.__create_onehot_label_array(self.padding_integer_label)
		df["ONEHOT_WORD_LABEL"] = df["WORD_LABEL"].transform(self.__create_onehot_label_array)

		word_onehot_encoded_labels = []
		word_scores = []
		max_len_divisor = float(self.__max_len)
		split_seq_scores = sequences.apply(self.__split_row_values)
		for onehot_encoded_labels, scores in split_seq_scores:
			word_onehot_encoded_labels.extend(onehot_encoded_labels)
			word_scores.extend(scores)
		assert max(len(seq) for seq in word_onehot_encoded_labels) <= self.__max_len
		return word_onehot_encoded_labels, word_scores

	def __create_onehot_label_array(self, integer_label: int) -> np.array:
		return self.onehot_encoder.transform(integer_label)

	def __create_onehot_label_arrays(self, values: np.array) -> np.array:
		reshaped_integer_labels = values.reshape(len(values), 1)
		return self.onehot_encoder.transform(reshaped_integer_labels)

	def __split_row_values(self, df: pd.DataFrame) -> Tuple[np.array, np.array]:
		onehot_encoded_label_arrays = df["ONEHOT_WORD_LABEL"].values
		score_values = df["PROBABILITY"].values
		row_count = len(onehot_encoded_label_arrays)
		assert row_count == len(score_values)

		partition_count = math.ceil(row_count / self.__max_len_divisor)
		split_onehot_encoded_labels = np.array_split(onehot_encoded_label_arrays, partition_count)
		split_score_values = np.array_split(score_values, partition_count)
		return split_onehot_encoded_labels, split_score_values


def are_all_entities_represented(df: pd.DataFrame, entity_ids) -> bool:
	"""
	Checks if all entities are represented for each individual token in each utterance in the dataframe.

	:param df: The dataframe to check.
	:param entity_ids: A collection of all unique entity IDs.
	:return: true iff for each token in each utterance there is one row for each entity ID.
	"""
	utt_toks = df.groupby(
		("CROSS_VALIDATION_ITER", "DYAD", "ROUND", "UTT_START_TIME", "UTT_END_TIME", "WORD", "TOKEN_SEQ_ORDINALITY"),
		as_index=False)
	print("Found {} utterance tokens for all cross-validations.".format(len(utt_toks)), file=sys.stderr)
	# Check if there is a row for each entity (possible referent) for each token
	return all(utt_toks.apply(lambda group: is_collection_equivalent(group["ENTITY"], entity_ids)))


def create_input_output_dfs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	input_df = df.loc[:, ["DYAD", "ROUND", "TOKEN_SEQ_ORDINALITY", "WORD", "IS_INSTRUCTOR", "IS_OOV"]]
	output_df = df.loc[:, ["DYAD", "ROUND", "TOKEN_SEQ_ORDINALITY", "PROBABILITY"]]
	return input_df, output_df


def create_token_sequences(df: pd.DataFrame, max_len: int) -> List[np.array]:
	"""
	Creates a sequence of sequences of tokens, each representing an utterance, each of which thus causes an "interruption" in the chain
	so that e.g. the first token of one utterance is not learned as dependent on the last token of the utterance preceding it.
	:param df: The DataFrame to process.
	:param max_len: The maximum sequence length.
	:return: A list of 2D numpy arrays, each representing a sequence of datapoints which represents an utterance.
	"""
	# https://stackoverflow.com/a/47815400/1391325
	df.sort_values("TOKEN_SEQ_ORDINALITY", inplace=True)
	sequences = df.groupby(("CROSS_VALIDATION_ITER", "DYAD", "ROUND", "UTT_START_TIME", "UTT_END_TIME", "ENTITY"),
						   as_index=False)
	result = []
	max_len_divisor = float(max_len)
	split_seqs = sequences.apply(lambda group: split_row_values(group[["WORD_LABEL", "PROBABILITY"]], max_len_divisor))
	for seqs in split_seqs:
		result.extend(seqs)
	assert max(len(seq) for seq in result) <= max_len
	return result


def find_target_ref_rows(df: pd.DataFrame) -> pd.DataFrame:
	result = df.loc[df["IS_TARGET"] == True]
	result_row_count = result.shape[0]
	complement_row_count = df.loc[~df.index.isin(result.index)].shape[0]
	assert result_row_count + complement_row_count == df.shape[0]
	print("Found {} nontarget rows and {} target rows. Ratio: {}".format(result_row_count, complement_row_count,
																		 complement_row_count / float(
																			 result_row_count)), file=sys.stderr)
	return result


def is_collection_equivalent(c1, c2) -> bool:
	c1_len = len(c1)
	c2_len = len(c2)
	return c1_len == c2_len and all(elem in c2 for elem in c1)


def read_results_file(inpath: str, encoding: str) -> pd.DataFrame:
	print("Reading \"{}\" using encoding \"{}\".".format(inpath, encoding), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=encoding, memory_map=True, dtype=__RESULTS_FILE_DTYPES)
	return result


def split_row_values(df: pd.DataFrame, max_len: float) -> pd.DataFrame:
	partition_count = math.ceil(df.shape[0] / max_len)
	return np.array_split(df.values, partition_count)


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
	orig_row_count = cv_results.shape[0]
	print("Read {} cross-validation results for {} dyad(s).".format(orig_row_count, len(dyad_ids)),
		  file=sys.stderr)
	entity_ids = frozenset(cv_results["ENTITY"].unique())
	print("Found {} unique entity IDs.".format(len(entity_ids)), file=sys.stderr)
	assert are_all_entities_represented(cv_results, entity_ids)
	assert cv_results.shape[0] == orig_row_count
	cv_results = find_target_ref_rows(cv_results)

	all_words = tuple(itertools.chain(("__PADDING__",), cv_results["WORD"].values))
	print("Converting {} vocabulary entries to integer labels.".format(len(all_words)), file=sys.stderr)
	# integer encode <https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/>
	label_encoder = LabelEncoder()
	label_encoder.fit(all_words)
	cv_results["WORD_LABEL"] = label_encoder.transform(cv_results["WORD"])

	max_seq_len = 4
	print("Splitting token sequences.", file=sys.stderr)
	onehot_encoder = OneHotEncoder(sparse=False)
	token_seq_factory = OneHotTokenEncodedTokenSequenceFactory(onehot_encoder, max_seq_len)
	word_onehot_encoded_labels, word_scores = token_seq_factory(cv_results)
	print("Split data into {} token sequences with a maximum sequence length of {}.".format(
		len(word_onehot_encoded_labels),
		max_seq_len),
		file=sys.stderr)

	# TODO: Stopped here

	# truncate and pad input sequences
	max_review_length = 500
	# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
	embedding_vector_length = 32
	model = Sequential()
	word_embeddings = Embedding(len(vocab), embedding_vector_length, input_length=max_review_length)
	model.add(word_embeddings)
	# model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	output_dim = 1
	model.add(LSTM(output_dim))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print(model.summary())

	# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
	# from keras.preprocessing.text import Tokenizer
	# create the tokenizer
	# t = Tokenizer()
	# fit the tokenizer on the documents
	# t.fit_on_texts(docs)

	# https://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/

	# input_datapoint_features = ("WORD", "IS_INSTRUCTOR", "IS_OOV", "SHAPE", "RED", "GREEN", "BLUE", "POSITION_X", "POSITION_Y", "MID_X", "MID_Y")
	input_datapoint_features = ("WORD", "IS_INSTRUCTOR", "IS_OOV")
	output_datapoint_features = ("PROBABILITY",)


# TODO: Create output features: one feature per word class, the value thereof being the referential salience, i.e. ((STDEV of probability of "true" for all referents in round being classified) * number of times classifier has been observed in training)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
