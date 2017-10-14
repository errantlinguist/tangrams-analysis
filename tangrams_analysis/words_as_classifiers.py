#!/usr/bin/env python3

import argparse
import itertools
import re
import sys
from collections import namedtuple
from typing import Callable, Iterable, Iterator, Mapping, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

import game_utterances
import session_data as sd
import utterances

CrossValidationDataFrames = namedtuple("CrossValidationDataFrames", ("training", "testing"))


def __create_regex_disjunction(regexes: Iterable[str]):
	return "(?:" + ")|(?:".join(regexes) + ")"


DEPENDENT_VARIABLE_COL_NAME_PATTERN = re.compile(__create_regex_disjunction(
	("SHAPE_(?:\\S+)", "EDGE_COUNT", "SIZE", "RED", "GREEN", "BLUE", "HUE", "POSITION_X", "POSITION_Y")))
INDEPENDENT_VARIABLE_COL_NAME = "REFERENT"

DYAD_ID_COL_NAME = "DYAD"
OUT_OF_VOCABULARY_TOKEN_LABEL = "__OUT_OF_VOCABULARY__"
TOKEN_CLASS_COL_NAME = game_utterances.SessionGameRoundUtteranceFactory.LinguisticDataColumn.TOKEN.value
'''
See T. Shore and G. Skantze. 2017. "Enhancing reference resolution in dialogue using participant feedback." Grounding Language Understanding 2017
'''
TOKEN_CLASS_SMOOTHING_FREQ_CUTOFF = 4

CATEGORICAL_VAR_COL_NAMES = ("SHAPE", "SUBMITTER",
							 game_utterances.SessionGameRoundUtteranceFactory.LinguisticDataColumn.SPEAKER.value)
# NOTE: For some reason, "pandas.get_dummies(..., columns=[col_name_1,...])" works with list objects but not with tuples
CATEGORICAL_DEPENDENT_VAR_COL_NAMES = ["SHAPE"]
assert all(x in CATEGORICAL_VAR_COL_NAMES for x in CATEGORICAL_DEPENDENT_VAR_COL_NAMES)


class CachingSessionDataFrameFactory(object):
	def __init__(self, session_data_frame_factory: Callable[[sd.SessionData], pd.DataFrame]):
		self.session_data_frame_factory = session_data_frame_factory
		self.cache = {}

	def __call__(self, infile: str, session: sd.SessionData) -> pd.DataFrame:
		try:
			session_data_frame = self.cache[infile]
		except KeyError:
			session_data_frame = self.session_data_frame_factory(session)
			session_data_frame[DYAD_ID_COL_NAME] = infile
			self.cache[infile] = session_data_frame
		return session_data_frame


class CrossValidationData(object):
	def __init__(self, testing_data: Tuple[str, sd.SessionData], training_data: Mapping[str, sd.SessionData]):
		self.testing_data = testing_data
		self.training_data = training_data

	@property
	def __key(self):
		return self.testing_data, self.training_data

	def __eq__(self, other):
		return (self is other or (isinstance(other, type(self))
								  and self.__key == other.__key))

	def __hash__(self):
		return hash(self.__key)

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)


class CrossValidationDataFrameFactory(object):
	@staticmethod
	def __categoricize_data(training_feature_df: pd.DataFrame, testing_feature_df: pd.DataFrame):
		for col_name in CATEGORICAL_VAR_COL_NAMES:
			unique_vals = tuple(sorted(frozenset(
				itertools.chain(training_feature_df[col_name].unique(), testing_feature_df[col_name].unique()))))
			training_feature_df[col_name] = pd.Categorical(training_feature_df[col_name], categories=unique_vals,
														   ordered=False)
			testing_feature_df[col_name] = pd.Categorical(testing_feature_df[col_name], categories=unique_vals,
														  ordered=False)

	def __init__(self, session_data_frame_factory: Callable[[str, sd.SessionData], pd.DataFrame]):
		self.session_data_frame_factory = session_data_frame_factory

	def __call__(self, named_session_data=Iterable[Tuple[str, sd.SessionData]]) -> Iterator[CrossValidationDataFrames]:
		for testing_session_name, testing_session_data in named_session_data:
			training_sessions = dict(
				(infile, training_session_data) for (infile, training_session_data) in named_session_data if
				testing_session_data != training_session_data)
			cross_validation_set = CrossValidationData((testing_session_name, testing_session_data),
													   training_sessions)
			yield self.__create_cross_validation_data_frames(cross_validation_set)

	def __create_cross_validation_data_frames(self,
											  cross_validation_data: CrossValidationData) -> CrossValidationDataFrames:
		training_feature_df = pd.concat(self.session_data_frame_factory(infile, session) for (infile, session) in
										cross_validation_data.training_data.items())
		# noinspection PyUnresolvedReferences
		# print("Training data shape: {}".format(training_feature_df.shape), file=sys.stderr)
		# print(training_feature_df)
		testing_feature_df = self.session_data_frame_factory(*cross_validation_data.testing_data)
		# print("Testing data shape: {}".format(testing_feature_df.shape), file=sys.stderr)
		# print(testing_feature_df)

		# noinspection PyTypeChecker
		self.__categoricize_data(training_feature_df, testing_feature_df)
		dummified_training_feature_df = pd.get_dummies(training_feature_df, columns=CATEGORICAL_DEPENDENT_VAR_COL_NAMES)
		dummified_testing_feature_df = pd.get_dummies(testing_feature_df, columns=CATEGORICAL_DEPENDENT_VAR_COL_NAMES)

		return CrossValidationDataFrames(dummified_training_feature_df, dummified_testing_feature_df)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Cross-validation of reference resolution for tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def smooth(df: pd.DataFrame):
	token_classes = df.groupby(TOKEN_CLASS_COL_NAME, as_index=False)
	smoothed_token_classes = frozenset(token_classes.filter(lambda group_df: len(group_df) < TOKEN_CLASS_SMOOTHING_FREQ_CUTOFF)[
		TOKEN_CLASS_COL_NAME].unique())
	print("Token class(es) used for smoothing: {}".format(smoothed_token_classes), file=sys.stderr)
	df.loc[df[TOKEN_CLASS_COL_NAME].isin(smoothed_token_classes), TOKEN_CLASS_COL_NAME] = OUT_OF_VOCABULARY_TOKEN_LABEL
	print("{} data point(s) used as out-of-vocabulary instance(s).".format(
		len(df[df[TOKEN_CLASS_COL_NAME] == OUT_OF_VOCABULARY_TOKEN_LABEL])), file=sys.stderr)


def __cross_validate(cross_validation_df: CrossValidationDataFrames):
	training_df = cross_validation_df.training
	smooth(training_df)

	token_class_training_insts = training_df.groupby(TOKEN_CLASS_COL_NAME, as_index=False)
	for token_class, training_insts in token_class_training_insts:
		# print("Using {} training instance(s) for class \"{}\".".format(len(training_insts), token_class), file=sys.stderr)
		dependent_var_cols = tuple(
			col for col in training_insts.columns if DEPENDENT_VARIABLE_COL_NAME_PATTERN.match(col))
		training_x = training_insts.loc[:, dependent_var_cols]
		training_y = training_insts.loc[:, INDEPENDENT_VARIABLE_COL_NAME]
		# print(training_y.unique())
		model = LogisticRegression()
		model.fit(training_x, training_y)

	testing_df = cross_validation_df.testing
	testing_y = training_df[INDEPENDENT_VARIABLE_COL_NAME]


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = tuple(sorted(sd.walk_session_data(inpaths), key=lambda item: item[0]))

	cross_validation_data_frame_factory = CrossValidationDataFrameFactory(
		CachingSessionDataFrameFactory(game_utterances.SessionGameRoundUtteranceFactory(
			utterances.TokenSequenceFactory())))
	print("Creating cross-validation datasets from {} session(s).".format(len(infile_session_data)), file=sys.stderr)
	# NOTE: This must be lazily-generated lest a dataframe be created in-memory for each cross-validation fold
	for cross_validation_df in cross_validation_data_frame_factory(infile_session_data):
		__cross_validate(cross_validation_df)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
