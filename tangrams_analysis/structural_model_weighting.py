#!/usr/bin/env python3

"""
Learns a measure of referential salience of classifiers used based on the context of their corresponding words in dialogue.

Uses tabular data, where each line represents the results of classification using one words-as-classifier model; See class "se.kth.speech.coin.tangrams.wac.logistic.WordProbabiltyScoreSequenceWriter" from the project "tangrams-wac" <https://github.com/errantlinguist/tangrams-wac>.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import itertools
import random
import sys
from typing import Iterator, List, Mapping, Tuple

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

class SequenceFeatureVectorFactory(object):

	def __init__(self,vocab_idxs : Mapping[str, int]):
		self.__vocab_idxs = vocab_idxs
		self.__feature_count = len(self.__vocab_idxs) + 3

	def create_datapoint_feature_array(self, row: pd.Series) -> List[float]:
		word_features = [0.0] * len(self.__vocab_idxs)
		# The features representing each individual vocabulary word are at the beginning of the feature vector
		word_features[self.__vocab_idxs[row["WORD"]]] = 1.0
		is_instructor = 1.0 if row["IS_INSTRUCTOR"] else 0.0
		is_oov = 1.0 if row["IS_OOV"] else 0.0
		#is_target = 1.0 if row["IS_TARGET"] else 0.0
		score = row["PROBABILITY"]
		other_features = list((is_instructor, is_oov, score))
		return word_features + other_features


	def __call__(self, df : pd.DataFrame) -> Iterator[List[float]]:
		# noinspection PyProtectedMember
		return (self.create_datapoint_feature_array(row._asdict()) for row in df.itertuples(index=False))


def find_target_ref_rows(df: pd.DataFrame) -> pd.DataFrame:
	result = df.loc[df["IS_TARGET"] == True]
	result_row_count = result.shape[0]
	complement_row_count = df.loc[~df.index.isin(result.index)].shape[0]
	assert result_row_count + complement_row_count == df.shape[0]
	print("Found {} nontarget rows and {} target rows. Ratio: {}".format(complement_row_count, result_row_count,
																		 complement_row_count / float(
																			 result_row_count)), file=sys.stderr)
	return result


def read_results_file(inpath: str, encoding: str) -> pd.DataFrame:
	print("Reading \"{}\" using encoding \"{}\".".format(inpath, encoding), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=encoding, memory_map=True, dtype=__RESULTS_FILE_DTYPES)
	return result

def split_training_testing(df: pd.DataFrame, test_set_size : int) -> Tuple[pd.DataFrame, pd.DataFrame]:
	dyad_ids = df["DYAD"].unique()
	training_set_size = len(dyad_ids) - test_set_size
	if training_set_size < 1:
		raise ValueError("Desired test set size is {} but only {} dyads found.".format(test_set_size, len(dyad_ids)))
	else:
		training_set_dyads = frozenset(np.random.choice(dyad_ids, training_set_size))
		print("Training set dyads: {}".format(sorted(training_set_dyads)), file=sys.stderr)
		training_set_idxs = df["DYAD"].isin(training_set_dyads)
		training_set = df.loc[training_set_idxs]
		test_set = df.loc[~training_set_idxs]
		test_set_dyads = frozenset(test_set["DYAD"].unique())
		print("Test set dyads: {}".format(sorted(test_set_dyads)), file=sys.stderr)

		assert not frozenset(training_set["DYAD"].unique()).intersection(frozenset(test_set_dyads))
		return training_set, test_set


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
	print("Read {} cross-validation results for {} dyad(s).".format(cv_results.shape[0],
																	cv_results["DYAD"].nunique()),
		  file=sys.stderr)

	cv_results = find_target_ref_rows(cv_results)
	vocab = tuple(sorted(cv_results["WORD"].unique()))
	print("Fitting one-hot encoder for vocabulary of size {}.".format(len(vocab)), file=sys.stderr)
	# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
	print("Creating vocab dictionary for one-hot label encoding.", file=sys.stderr)
	vocab_idxs = dict((word, idx) for (idx, word) in enumerate(vocab))

	training_df, test_df = split_training_testing(cv_results, 1)

	print("Splitting token sequences.", file=sys.stderr)
	# https://stackoverflow.com/a/47815400/1391325
	cv_results.sort_values("TOKEN_SEQ_ORDINALITY", inplace=True)
	sequence_groups = cv_results.groupby(("CROSS_VALIDATION_ITER", "DYAD", "SPLIT_SEQ_NO", "UTT_START_TIME", "UTT_END_TIME", "ENTITY"),
							   as_index=False)
	seq_feature_factory = SequenceFeatureVectorFactory(vocab_idxs)
	matrix = np.array(tuple(tuple(seq_feature_factory(seq)) for _, seq in sequence_groups))
	print("Created a data matrix of shape {}.".format(matrix.shape), file=sys.stderr)

	x = matrix[:,:,:-1]
	#print(x)
	y = matrix[:,:,-1]
	#print(y)

	model = Sequential()
	#word_embeddings = Embedding(len(vocab), embedding_vector_length, input_length=max_review_length)
	#model.add(word_embeddings)
	# model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	output_dim = 1
	model.add(LSTM(output_dim))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	# https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
	# from keras.preprocessing.text import Tokenizer
	# create the tokenizer
	# t = Tokenizer()
	# fit the tokenizer on the documents
	# t.fit_on_texts(docs)

	# https://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/


# TODO: Create output features: one feature per word class, the value thereof being the referential salience, i.e. ((STDEV of probability of "true" for all referents in round being classified) * number of times classifier has been observed in training)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
