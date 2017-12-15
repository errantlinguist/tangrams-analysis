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


class TokenSequenceFactory(object):

	def __init__(self, max_len: int):
		self.__max_len = max_len
		self.__max_len_divisor = float(max_len)

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
		# sequences.apply(lambda seq : self.create_word_forms(seq["ONEHOT_WORD_LABEL"].values))

		word_seqs = []
		score_seqs = []
		split_seq_scores = sequences.apply(self.__split_row_values)
		for word_seq, score_seq in split_seq_scores:
			word_seqs.extend(word_seq)
			score_seqs.extend(score_seq)
		assert max(len(seq) for seq in word_seqs) <= self.__max_len
		return word_seqs, score_seqs

	def __split_row_values(self, df: pd.DataFrame) -> Tuple[np.array, np.array]:
		seq_words = df["WORD"].values
		seq_scores = df["PROBABILITY"].values

		partition_count = math.ceil(len(seq_words) / self.__max_len_divisor)
		split_seq_words = np.array_split(seq_words, partition_count)
		split_seq_scores = np.array_split(seq_scores, partition_count)
		return split_seq_words, split_seq_scores


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


def pad_sequence(word_score_seq: Tuple[np.array, np.array], min_length: int) -> Tuple[np.array, np.array]:
	word_seq, score_seq = word_score_seq
	word_count = len(word_seq)
	assert word_count == len(score_seq)
	length_diff = min_length - word_count
	if length_diff > 0:
		# NOTE: creating an intermediate tuple is necessary
		padding_words = np.full(length_diff, "__PADDING__")
		padded_word_seq = np.concatenate((padding_words, word_seq), axis=0)
		assert len(padded_word_seq) == min_length
		padding_scores = np.full(length_diff, 0.0)
		padded_score_seq = np.concatenate((padding_scores, score_seq), axis=0)
		assert len(padded_score_seq) == min_length
		result = padded_word_seq, padded_score_seq
	else:
		result = word_seq, score_seq

	return result


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

	desired_seq_len = 4
	print("Splitting token sequences.", file=sys.stderr)
	token_seq_factory = TokenSequenceFactory(desired_seq_len)
	word_seqs, score_seqs = token_seq_factory(cv_results)
	print("Split data into {} token sequences with a maximum sequence length of {}.".format(len(word_seqs),
																							desired_seq_len),
		  file=sys.stderr)

	all_words = tuple(itertools.chain(("__PADDING__",), cv_results["WORD"].values))
	print("Converting {} vocabulary entries to integer labels.".format(len(all_words)), file=sys.stderr)
	# integer encode <https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/>
	label_encoder = LabelEncoder()
	label_encoder.fit(all_words)
	cv_results["WORD_LABEL"] = label_encoder.transform(cv_results["WORD"])
	onehot_encoder = OneHotEncoder(sparse=False)
	padding_integer_label = label_encoder.transform(["__PADDING__"])[0]

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
