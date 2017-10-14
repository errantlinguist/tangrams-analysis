import sys
from collections import namedtuple
from typing import Callable, Iterable, Iterator, Mapping, Tuple

import pandas as pd

import session_data as sd

CrossValidationDataFrames = namedtuple("CrossValidationDataFrames", ("training", "testing"))


class CachingSessionDataFrameFactory(object):
	def __init__(self, session_data_frame_factory: Callable[[sd.SessionData], pd.DataFrame]):
		self.session_data_frame_factory = session_data_frame_factory
		self.cache = {}

	def __call__(self, infile: str, session: sd.SessionData) -> pd.DataFrame:
		try:
			session_data_frame = self.cache[infile]
		except KeyError:
			session_data_frame = self.session_data_frame_factory(session)
			session_data_frame["DYAD"] = infile
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
		print("Training data shape: {}".format(training_feature_df.shape), file=sys.stderr)
		# print(training_feature_df)
		testing_feature_df = self.session_data_frame_factory(*cross_validation_data.testing_data)
		print("Testing data shape: {}".format(testing_feature_df.shape), file=sys.stderr)
		# print(testing_feature_df)
		return CrossValidationDataFrames(training_feature_df, testing_feature_df)
