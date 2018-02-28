#!/usr/bin/env python3

import argparse
import re
import sys
from collections import defaultdict
from decimal import Decimal, Inexact, localcontext
from typing import Callable, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, MutableSequence, \
	Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tangrams_analysis import cross_validation
from tangrams_analysis import game_utterances
from tangrams_analysis import session_data as sd


def __create_regex_disjunction(regexes: Iterable[str]) -> str:
	return "(?:" + ")|(?:".join(regexes) + ")"


DEPENDENT_VARIABLE_COL_NAME_PATTERN = re.compile(__create_regex_disjunction(
	("SHAPE_(?:\\S+)", "EDGE_COUNT", "SIZE", "RED", "GREEN", "BLUE", "HUE", "POSITION_X", "POSITION_Y")))
INDEPENDENT_VARIABLE_COL_NAME = "REFERENT"

ORIGINAL_INDEX_COL_NAME = "OriginalIndex"
OUT_OF_VOCABULARY_TOKEN_LABEL = "__OUT_OF_VOCABULARY__"
'''
See T. Shore and G. Skantze. 2017. "Enhancing reference resolution in dialogue using participant feedback." Grounding Language Understanding 2017
'''
DEFAULT_TOKEN_CLASS_SMOOTHING_FREQ_CUTOFF = 4


class CrossValidator(object):
	def __init__(self, smoothing_freq_cutoff: int):
		# self.parallelizer = parallelizer
		self.smoothing_freq_cutoff = smoothing_freq_cutoff

	def __call__(self, cross_validation_df: cross_validation.CrossValidationDataFrames):
		training_df = cross_validation_df.training
		dependent_var_cols = tuple(
			col for col in training_df.columns if DEPENDENT_VARIABLE_COL_NAME_PATTERN.match(col))
		training_insts_per_observation = Decimal(training_df[game_utterances.EventColumn.ENTITY_ID.value].nunique())
		word_model_trainer = WordModelTrainer(dependent_var_cols, INDEPENDENT_VARIABLE_COL_NAME,
											  lambda token_type_training_insts: smooth(token_type_training_insts,
																					   self.smoothing_freq_cutoff,
																					   training_insts_per_observation))
		print("Training using a total of {} dataframe row(s).".format(len(training_df)), file=sys.stderr)
		word_models = word_model_trainer(training_df)
		print("Trained models for {} token type(s).".format(len(word_models)),
			  file=sys.stderr)

		testing_df = cross_validation_df.testing
		print("Testing using a total of {} dataframe row(s).".format(len(testing_df)), file=sys.stderr)
		token_type_testing_insts = create_token_type_insts(testing_df)
		print("Created testing datasets for {} token type(s).".format(len(token_type_testing_insts)),
			  file=sys.stderr)
		oov_model = word_models[OUT_OF_VOCABULARY_TOKEN_LABEL]
		# 2D array for entity description * word classification probabilities
		last_idx = testing_df[ORIGINAL_INDEX_COL_NAME].max()

		decision_classes = tuple(sorted(
			frozenset(decision_class for classifier in word_models.values() for decision_class in classifier.classes_)))
		decision_class_idxs = dict((decision_class, idx) for (idx, decision_class) in enumerate(decision_classes))
		decision_class_probs = np.zeros((last_idx + 1, len(token_type_testing_insts), len(decision_class_idxs)))
		for col_idx, (token_class, testing_insts) in enumerate(token_type_testing_insts.items()):
			# print("Testing classifier for token type \"{}\".".format(token_class), file=sys.stderr)
			classifier = word_models.get(token_class, oov_model)
			testing_inst_df = pd.DataFrame(testing_insts)
			testing_x = testing_inst_df.loc[:, dependent_var_cols]
			testing_y = testing_inst_df.loc[:, INDEPENDENT_VARIABLE_COL_NAME]
			orig_idxs = testing_inst_df.loc[:, ORIGINAL_INDEX_COL_NAME]

			decision_probs = classifier.predict_log_proba(testing_x)
			for class_idx, decision_class in enumerate(classifier.classes_):
				class_decision_probs = decision_probs[:, class_idx]
				result_matrix_class_idx = decision_class_idxs[decision_class]
				for orig_idx, truth_decision_prob in zip(orig_idxs, class_decision_probs):
					decision_class_probs[orig_idx, col_idx, result_matrix_class_idx] += truth_decision_prob

		print(decision_class_probs)


class WordModelTrainer(object):
	def __init__(self, dependent_var_cols: Sequence[str], independent_var_cols: Sequence[str],
				 smoother: Callable[[MutableMapping[str, MutableSequence[pd.Series]]], None]):
		self.dependent_var_cols = dependent_var_cols
		self.independent_var_cols = independent_var_cols
		self.smoother = smoother

	def __call__(self, training_df: pd.DataFrame) -> Dict[str, LogisticRegression]:
		print("Training using a total of {} dataframe row(s).".format(len(training_df)), file=sys.stderr)
		token_type_training_insts = create_token_type_insts(training_df)
		print("Created training datasets for {} token type(s).".format(len(token_type_training_insts)),
			  file=sys.stderr)
		self.smoother(token_type_training_insts)
		return self.__train_models(token_type_training_insts)

	def __train_models(self, token_type_training_insts: Mapping[str, Iterable[pd.Series]]) -> Dict[
		str, LogisticRegression]:
		word_models = {}
		for (token_type, training_insts) in token_type_training_insts.items():
			training_inst_df = pd.DataFrame(training_insts)
			# print(training_inst_df)
			training_x = training_inst_df.loc[:, self.dependent_var_cols]
			training_y = training_inst_df.loc[:, self.independent_var_cols]
			model = LogisticRegression()
			model.fit(training_x, training_y)
			word_models[token_type] = model
		return word_models


def create_token_observation_series(row: pd.Series) -> List[Tuple[str, pd.Series]]:
	token_observation_template = row.copy()
	# noinspection PyUnresolvedReferences
	token_observation_template.drop(
		game_utterances.SessionGameRoundUtteranceSequenceFactory.UTTERANCE_SEQUENCE_COL_NAME, inplace=True)
	result = []
	utts = row[game_utterances.SessionGameRoundUtteranceSequenceFactory.UTTERANCE_SEQUENCE_COL_NAME]
	for utt in utts:
		tokens = utt.content
		for token in tokens:
			# noinspection PyUnresolvedReferences
			token_observation = token_observation_template.copy()
			result.append((token, token_observation))
	return result


def create_token_type_insts(df: pd.DataFrame) -> DefaultDict[str, List[pd.Series]]:
	df.loc[:, ORIGINAL_INDEX_COL_NAME] = df.index
	token_insts = (token_training_inst for row_training_inst_set in
				   df.apply(create_token_observation_series, axis=1) for token_training_inst in
				   row_training_inst_set)
	result = defaultdict(list)
	for token, inst in token_insts:
		result[token].append(inst)
	# NOTE: simple lists of Series objects are returned rather than complete DataFrame objects so that e.g. the different token types can be smoothed before creating a DataFrame for use in training/classification
	return result


def smooth(token_type_training_insts: MutableMapping[str, MutableSequence[pd.Series]], smoothing_freq_cutoff: int,
		   training_insts_per_observation: Decimal):
	observation_counts = token_type_observation_counts(token_type_training_insts,
													   training_insts_per_observation)
	smoothed_token_types = frozenset(
		token_type for token_type, count in observation_counts if count < smoothing_freq_cutoff)
	for token_type in smoothed_token_types:
		training_insts = token_type_training_insts[token_type]
		del token_type_training_insts[token_type]
		try:
			oov_instances = token_type_training_insts[OUT_OF_VOCABULARY_TOKEN_LABEL]
		except KeyError:
			oov_instances = []

		oov_instances.extend(training_insts)

	smoothed_row_idxs = token_type_training_insts[OUT_OF_VOCABULARY_TOKEN_LABEL]
	print("Token type(s) used for smoothing: {}; {} data point(s) used as out-of-vocabulary instance(s).".format(
		smoothed_token_types, len(smoothed_row_idxs)), file=sys.stderr)


def token_type_observation_counts(token_type_training_insts: Mapping[str, Sequence[pd.Series]],
								  training_insts_per_observation: Decimal) -> Tuple[Tuple[str, int], ...]:
	token_type_training_inst_counts = ((token, len(training_insts)) for (token, training_insts) in
									   token_type_training_insts.items())
	with localcontext() as ctx:
		ctx.traps[Inexact] = True
		result = tuple(
			(token_type, int((Decimal(count) / training_insts_per_observation).to_integral_exact())) for
			token_type, count in
			token_type_training_inst_counts)
	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Cross-validation of reference resolution for tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	result.add_argument("-s", "--smoothing", metavar="COUNT", type=int,
						default=DEFAULT_TOKEN_CLASS_SMOOTHING_FREQ_CUTOFF,
						help="The minimum number of times a word class can be observed without being smoothed.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = tuple(sorted(sd.walk_session_data(inpaths), key=lambda item: item[0]))
	smoothing_freq_cutoff = args.smoothing
	print("Using token types with a frequency less than {} for smoothing.".format(smoothing_freq_cutoff),
		  file=sys.stderr)

	cross_validation_data_frame_factory = cross_validation.CrossValidationDataFrameFactory(
		cross_validation.CachingSessionDataFrameFactory())
	print("Creating cross-validation datasets from {} session(s).".format(len(infile_session_data)), file=sys.stderr)

	# parallelizer = Parallel(n_jobs = -2, backend = "multiprocessing")
	cross_validator = CrossValidator(smoothing_freq_cutoff)
	# NOTE: This must be lazily-generated lest a dataframe be created in-memory for each cross-validation fold
	for cross_validation_df in cross_validation_data_frame_factory(infile_session_data):
		cross_validator(cross_validation_df)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
