#!/usr/bin/env python3

import argparse
import sys
from typing import ItemsView, Mapping, Tuple

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

	def create_training_data(self, sessions: ItemsView[str, SessionData]):
		for infile, session in sessions:
			game_round_token_df = self.session_game_round_utt_factory(session)
			print(game_round_token_df)
		# for round_id, (game_round, round_utts) in enumerate(session_round_utts.game_round_utts,
		#													start=game_utterances.SessionGameRoundUtteranceFactory.ROUND_ID_OFFSET):
		#	round_instructor_id = session_round_utts.round_instructor_ids[round_id]

	def __call__(self, cross_validation_data: CrossValidationData):
		self.create_training_data(cross_validation_data.training_data.items())
		logistic = LogisticRegression()
		# logistic.fit(X,y)
		# logistic.predict(iris.data[-1,:]),iris.target[-1])
		pass


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
