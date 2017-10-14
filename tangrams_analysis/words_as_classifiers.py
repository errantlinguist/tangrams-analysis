#!/usr/bin/env python3

import argparse
import sys
from typing import Iterator, Mapping, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

import game_utterances
import utterances
from session_data import SessionData, walk_session_data


class CrossValidationData(object):
	def __init__(self, testing_data: Tuple[str, SessionData], training_data: Mapping[str, SessionData]):
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


class CrossValidator(object):
	def __init__(self, session_game_round_utt_factory: game_utterances.SessionGameRoundUtteranceFactory):
		self.session_game_round_utt_factory = session_game_round_utt_factory
		self.session_training_data = {}

	def __call__(self, cross_validation_data: CrossValidationData):
		training_feature_df = pd.concat(self.__fetch_session_feature_df(infile, session) for (infile, session) in
										cross_validation_data.training_data.items())
		print("Training data shape: {}".format(training_feature_df.shape), file=sys.stderr)
		testing_feature_df = self.__fetch_session_feature_df(*cross_validation_data.testing_data)
		print("Testing data shape: {}".format(testing_feature_df.shape), file=sys.stderr)

		logistic = LogisticRegression()

	# TODO: Finish

	# logistic.fit(X,y)
	# logistic.predict(iris.data[-1,:]),iris.target[-1])

	def __fetch_session_feature_df(self, infile: str, session: SessionData) -> Iterator[pd.DataFrame]:
		try:
			session_feature_df = self.session_training_data[infile]
		except KeyError:
			session_feature_df = self.session_game_round_utt_factory(session)
			session_feature_df["DYAD"] = infile
			self.session_training_data[infile] = session_feature_df
		return session_feature_df


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Cross-validation of reference resolution for tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = tuple(sorted(walk_session_data(inpaths), key=lambda item: item[0]))

	cross_validator = CrossValidator(game_utterances.SessionGameRoundUtteranceFactory(
		utterances.TokenSequenceFactory()))
	for testing_infile_path, testing_session_data in infile_session_data:
		training_sessions = dict(
			(infile, training_session_data) for (infile, training_session_data) in infile_session_data if
			testing_session_data != training_session_data)
		cross_validation_set = CrossValidationData((testing_infile_path, testing_session_data), training_sessions)
		cross_validator(cross_validation_set)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
