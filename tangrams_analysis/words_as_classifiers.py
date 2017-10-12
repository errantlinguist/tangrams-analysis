import argparse
import sys

import utterances
import game_utterances
from sklearn.linear_model import LogisticRegression

from session_data import SessionData, walk_session_data
from typing import FrozenSet, Iterable, Tuple

class CrossValidationData(object):
	def __init__(self, testing_data : SessionData, training_data : FrozenSet[SessionData]):
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

def create_training_data(sessions : Iterable[SessionData]):
	for session in sessions:
		utterances.read_segments(session.utts)

def cross_validation(cross_validation_data : CrossValidationData):
	logistic = LogisticRegression()
	#logistic.fit(X,y)
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
	session_data = frozenset(data for (inpath, data) in walk_session_data(inpaths))
	cross_validation_testing_training_data = tuple((CrossValidationData(testing_data, frozenset(
		training_data for training_data in session_data if training_data != testing_data)) for testing_data in
													session_data))
	print("Using {} sessions in total for {}-fold cross-validation.".format(len(session_data), len(
		cross_validation_testing_training_data)), file=sys.stderr)
	for testing_training_data in cross_validation_testing_training_data:
		cross_validation(testing_training_data)






if __name__ == "__main__":
	__main(__create_argparser().parse_args())
