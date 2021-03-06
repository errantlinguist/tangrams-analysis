"""
Functionalities for cross-validating words-as-classifiers reference resolution (not yet finished!).
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import csv
import itertools
from collections import namedtuple
from typing import Callable, Iterable, Iterator, Mapping, Optional, Tuple

import pandas as pd

from . import game_utterances
from . import iristk
from . import session_data as sd

CATEGORICAL_VAR_COL_NAMES = (
	game_utterances.EventColumn.ENTITY_SHAPE.value, game_utterances.EventColumn.EVENT_SUBMITTER.value)
# NOTE: For some reason, "pandas.get_dummies(..., columns=[col_name_1,...])" works with list objects but not with tuples
CATEGORICAL_DEPENDENT_VAR_COL_NAMES = [game_utterances.EventColumn.ENTITY_SHAPE.value]
assert all(col_name in CATEGORICAL_VAR_COL_NAMES for col_name in CATEGORICAL_DEPENDENT_VAR_COL_NAMES)

RESULTS_FILE_ENCODING = "utf-8"
__RESULTS_FILE_DTYPES = {"Cleaning.DISFLUENCIES": bool, "Cleaning.DUPLICATES": bool, "Cleaning.FILLERS": bool}

CrossValidationDataFrames = namedtuple("CrossValidationDataFrames", ("training", "testing"))


class CachingSessionDataFrameFactory(object):
	def __init__(self, session_data_frame_factory: Optional[Callable[[sd.SessionData], pd.DataFrame]] = None):
		self.session_data_frame_factory = game_utterances.SessionGameRoundUtteranceSequenceFactory() if session_data_frame_factory is None else session_data_frame_factory
		self.cache = {}

	def __call__(self, infile: str, session: sd.SessionData) -> pd.DataFrame:
		try:
			result = self.cache[infile]
		except KeyError:
			result = self.session_data_frame_factory(session)
			result[game_utterances.EventColumn.DYAD_ID.value] = infile
			self.cache[infile] = result
		return result


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
			unique_values = tuple(sorted(frozenset(
				itertools.chain(training_feature_df[col_name].unique(), testing_feature_df[col_name].unique()))))
			training_feature_df[col_name] = pd.Categorical(training_feature_df[col_name], categories=unique_values,
														   ordered=False)
			testing_feature_df[col_name] = pd.Categorical(testing_feature_df[col_name], categories=unique_values,
														  ordered=False)

	def __init__(self, session_data_frame_factory: Optional[Callable[[str, sd.SessionData], pd.DataFrame]]):
		self.session_data_frame_factory = CachingSessionDataFrameFactory() if session_data_frame_factory is None else session_data_frame_factory

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
		testing_feature_df = self.session_data_frame_factory(*cross_validation_data.testing_data)

		# noinspection PyTypeChecker
		self.__categoricize_data(training_feature_df, testing_feature_df)
		dummified_training_feature_df = pd.get_dummies(training_feature_df, columns=CATEGORICAL_DEPENDENT_VAR_COL_NAMES)
		dummified_testing_feature_df = pd.get_dummies(testing_feature_df, columns=CATEGORICAL_DEPENDENT_VAR_COL_NAMES)

		return CrossValidationDataFrames(dummified_training_feature_df, dummified_testing_feature_df)


def read_results_file(inpath: str) -> pd.DataFrame:
	return pd.read_csv(inpath, sep=csv.excel_tab.delimiter, dialect=csv.excel_tab, float_precision="round_trip",
					   encoding=RESULTS_FILE_ENCODING, memory_map=True, parse_dates=["TIME", "EVENT_TIME"],
					   date_parser=iristk.parse_timestamp,
					   dtype=__RESULTS_FILE_DTYPES)
