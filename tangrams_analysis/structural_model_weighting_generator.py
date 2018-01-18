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
import multiprocessing
import random
import sys
from collections import defaultdict
from typing import DefaultDict, List, Sequence, Tuple

import keras.preprocessing.sequence
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

RESULTS_FILE_CSV_DIALECT = csv.excel_tab

# NOTE: "category" dtype doesn't work with pandas-0.21.0 but does with pandas-0.21.1
__RESULTS_FILE_DTYPES = {"DYAD": "category", "ENTITY": "category", "IS_TARGET": bool, "IS_OOV": bool,
						 "IS_INSTRUCTOR": bool, "SHAPE": "category", "ONLY_INSTRUCTOR": bool, "WEIGHT_BY_FREQ": bool}


class DataGeneratorFactory(object):

	@staticmethod
	def __group_by_seq_len(seq_xy: pd.Series) -> DefaultDict[int, List[Tuple[np.matrix, np.ndarray]]]:
		result = defaultdict(list)
		for xy in seq_xy:
			seq_len = xy[0].shape[0]
			result[seq_len].append(xy)
		return result

	def __init__(self, onehot_encoder: OneHotEncoder):
		self.onehot_encoder = onehot_encoder

	@property
	def input_feature_count(self) -> int:
		word_features = self.onehot_encoder.n_values_[0]
		return word_features + 1

	@property
	def output_feature_count(self) -> int:
		return 1

	def __call__(self, df: pd.DataFrame) -> "TokenSequenceSequence":
		sequence_groups = df.groupby(
			("CROSS_VALIDATION_ITER", "DYAD", "ROUND", "UTT_START_TIME", "UTT_END_TIME", "ENTITY"), sort=False)
		print("Generating data for {} entity token sequence(s).".format(len(sequence_groups)), file=sys.stderr)
		seq_xy = sequence_groups.apply(self.__create_seq_xy)
		len_dict = self.__group_by_seq_len(seq_xy)
		print("Created {} batches, one for each unique sequence length.".format(len(len_dict)), file=sys.stderr)
		seq_batches_by_len = tuple(len_dict.values())
		return TokenSequenceSequence(seq_batches_by_len)

	def __create_datapoint_x(self, row: pd.Series) -> Tuple[np.ndarray,]:
		# word_features = [0.0] * len(self.__vocab_idxs)
		# The features representing each individual vocabulary word are at the beginning of the feature vector
		# word_features[self.__vocab_idxs[row["WORD"]]] = 1.0
		# word_label = self.label_encoder.transform(row["WORD"])
		word_label = row["WORD_LABEL"]
		# print("Word label: {}".format(word_label), file=sys.stderr)
		# "OneHotEncoder.transform(..)" returns a matrix even if only a single value is passed to it, so get just the first (and only) row
		word_features = self.onehot_encoder.transform(word_label)[0]
		# print("Word features: {}".format(word_features), file=sys.stderr)
		# The word label for the one-hot encoding is that with the same index as the column that has a "1" value, i.e. the highest value in the vector of one-hot encoding values
		# inverse_label = np.argmax(word_features)
		# assert inverse_label == word_label
		# inverse_word = self.label_encoder.inverse_transform([inverse_label])
		# print("Inverse word label: {}".format(inverse_label), file=sys.stderr)
		is_instructor = 1.0 if row["IS_INSTRUCTOR"] else 0.0
		# is_target = 1.0 if row["IS_TARGET"] else 0.0
		other_features = np.array((is_instructor,))
		# result = word_features + other_features
		result = np.concatenate((word_features, other_features))
		# print("Created a vector of {} features.".format(len(result)), file=sys.stderr)
		# NOTE: Returning a tuple is a hack in order to return an instance of "np.ndarray" from "DataFrame.apply()"
		return result,

	def __create_seq_x_matrix(self, seq_df: pd.DataFrame) -> np.matrix:
		# NOTE: The returned tuples have to be unpacked outside of the "apply(..)" function
		vectors = seq_df.apply(self.__create_datapoint_x, axis=1)
		return np.matrix(tuple(vector[0] for vector in vectors))

	def __create_seq_xy(self, seq_df: pd.DataFrame) -> Tuple[np.matrix, np.ndarray]:
		x = self.__create_seq_x_matrix(seq_df)
		y = seq_df["PROBABILITY"].values
		return x, y


class TokenSequenceSequence(keras.utils.Sequence):
	"""
	A sequence (i.e. less confusingly a dataset) of token sequences, each of which is for a given distinct entity, i.e. possible referent.
	"""

	def __init__(self, seq_batches_by_len: Sequence[Sequence[Tuple[np.matrix, np.ndarray]]]):
		self.seq_batches_by_len = seq_batches_by_len

	def __len__(self) -> int:
		return len(self.seq_batches_by_len)

	def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
		# print("Getting batch idx {}.".format(idx), file=sys.stderr)
		batch = self.seq_batches_by_len[idx]
		seq_x = tuple(x for x, y in batch)
		x = np.asarray(seq_x)
		# print("X shape: {}".format(x.shape), file=sys.stderr)
		seq_y = tuple(y for x, y in batch)
		if any(len(y.shape) > 1 for y in seq_y):
			raise ValueError("Output feature vectors with a dimensionality greater than 1 are not supported.")
		y = np.asarray(tuple(y[0] for y in seq_y))
		# y = np.asarray(seq_y)
		# print("Y shape: {}".format(y.shape), file=sys.stderr)
		return x, y


def create_model(input_feature_count: int, output_feature_count: int) -> Sequential:
	result = Sequential()
	# word_embeddings = Embedding(len(vocab), embedding_vector_length, input_length=max_review_length)
	# model.add(word_embeddings)
	# model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	# input shape is a pair of (timesteps, features) <https://stackoverflow.com/a/44583784/1391325>
	input_shape = (None, input_feature_count)
	print("Input shape: {}".format(input_shape), file=sys.stderr)
	units = output_feature_count
	print("Units: {}".format(units), file=sys.stderr)
	lstm = LSTM(input_shape=input_shape, units=units, dropout=0.1, recurrent_dropout=0.1)
	# lstm = LSTM(batch_input_shape = training_x.shape, stateful = True, units=len(training_y.shape))
	result.add(lstm)
	result.add(Dense(units, activation='sigmoid'))
	result.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(result.summary())
	return result


def find_target_ref_rows(df: pd.DataFrame) -> pd.DataFrame:
	result = df.loc[df["IS_TARGET"] == True]
	result_row_count = result.shape[0]
	complement_row_count = df.loc[~df.index.isin(result.index)].shape[0]
	assert result_row_count + complement_row_count == df.shape[0]
	print("Found {} non-target rows and {} target rows. Ratio: {}".format(complement_row_count, result_row_count,
																		  complement_row_count / float(
																			  result_row_count)), file=sys.stderr)
	return result


def read_results_file(inpath: str, encoding: str) -> pd.DataFrame:
	print("Reading \"{}\" using encoding \"{}\".".format(inpath, encoding), file=sys.stderr)
	result = pd.read_csv(inpath, dialect=RESULTS_FILE_CSV_DIALECT, sep=RESULTS_FILE_CSV_DIALECT.delimiter,
						 float_precision="round_trip",
						 encoding=encoding, memory_map=True, dtype=__RESULTS_FILE_DTYPES)
	return result


def split_training_testing(df: pd.DataFrame, test_set_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
	dyad_ids = df["DYAD"].unique()
	training_set_size = len(dyad_ids) - test_set_size
	if training_set_size < 1:
		raise ValueError("Desired test set size is {} but only {} dyads found.".format(test_set_size, len(dyad_ids)))
	else:
		training_set_dyads = frozenset(np.random.choice(dyad_ids, training_set_size, replace=False))
		assert len(training_set_dyads) == training_set_size
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

	# Create vocab before splitting training and testing DFs so that the word feature set is stable
	print("Fitting one-hot encoder for vocabulary of size {}.".format(cv_results["WORD"].nunique()), file=sys.stderr)
	# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
	# integer encode
	label_encoder = LabelEncoder()
	vocab_labels = label_encoder.fit_transform(cv_results["WORD"])
	cv_results["WORD_LABEL"] = vocab_labels
	# print(vocab_labels)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	vocab_labels = vocab_labels.reshape(len(vocab_labels), 1)
	onehot_encoder.fit(vocab_labels)
	# assert onehot_encoder.n_values_ == len(vocab_words)
	# vocab_onehot_encoded = onehot_encoder.fit_transform(vocab_labels)
	# print(vocab_onehot_encoded)
	# invert first example
	# inverted = label_encoder.inverse_transform([np.argmax(vocab_onehot_encoded[0, :])])
	# print(inverted)

	# https://stackoverflow.com/a/47815400/1391325
	cv_results.sort_values("TOKEN_SEQ_ORDINALITY", inplace=True)
	training_df, test_df = split_training_testing(cv_results, 1)
	# Only train on "true" referents
	training_df = find_target_ref_rows(training_df)

	print("Generating training data token sequences.", file=sys.stderr)
	data_generator_factory = DataGeneratorFactory(onehot_encoder)
	training_data_generator = data_generator_factory(training_df)
	# validation_data_generator = data_generator_factory()

	# https://stackoverflow.com/a/43472000/1391325
	with keras.backend.get_session():
		model = create_model(data_generator_factory.input_feature_count, data_generator_factory.output_feature_count)
		# train LSTM
		epochs = 250
		#workers = max(multiprocessing.cpu_count() - 1, 1)
		print("Training model using {} epoch(s).".format(epochs), file=sys.stderr)
		training_history = model.fit_generator(training_data_generator, epochs=epochs, verbose=1)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
