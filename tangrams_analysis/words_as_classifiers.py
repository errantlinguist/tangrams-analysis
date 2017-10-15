#!/usr/bin/env python3

import argparse
import itertools
import re
import sys
from collections import Counter, defaultdict, namedtuple
from decimal import Decimal, Inexact, localcontext
from numbers import Integral
from typing import Callable, Iterable, Iterator, Mapping, Sequence, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

import game_utterances
import session_data as sd
import utterances

CrossValidationDataFrames = namedtuple("CrossValidationDataFrames", ("training", "testing"))
TokenTypeRowIndexMapping = namedtuple("TokenTypeRowIndexMapping", ("row_idxs", "type_counts"))


def __create_regex_disjunction(regexes: Iterable[str]) -> str:
	return "(?:" + ")|(?:".join(regexes) + ")"


DEPENDENT_VARIABLE_COL_NAME_PATTERN = re.compile(__create_regex_disjunction(
	("SHAPE_(?:\\S+)", "EDGE_COUNT", "SIZE", "RED", "GREEN", "BLUE", "HUE", "POSITION_X", "POSITION_Y")))
INDEPENDENT_VARIABLE_COL_NAME = "REFERENT"

DYAD_ID_COL_NAME = "DYAD"
OUT_OF_VOCABULARY_TOKEN_LABEL = "__OUT_OF_VOCABULARY__"
'''
See T. Shore and G. Skantze. 2017. "Enhancing reference resolution in dialogue using participant feedback." Grounding Language Understanding 2017
'''
DEFAULT_TOKEN_CLASS_SMOOTHING_FREQ_CUTOFF = 4

CATEGORICAL_VAR_COL_NAMES = ("SHAPE", "SUBMITTER")
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
		testing_feature_df = self.session_data_frame_factory(*cross_validation_data.testing_data)

		# noinspection PyTypeChecker
		self.__categoricize_data(training_feature_df, testing_feature_df)
		dummified_training_feature_df = pd.get_dummies(training_feature_df, columns=CATEGORICAL_DEPENDENT_VAR_COL_NAMES)
		dummified_testing_feature_df = pd.get_dummies(testing_feature_df, columns=CATEGORICAL_DEPENDENT_VAR_COL_NAMES)

		return CrossValidationDataFrames(dummified_training_feature_df, dummified_testing_feature_df)


class CrossValidator(object):
	@staticmethod
	def __split_dependent_independent_vars(df: pd.DataFrame, dependent_var_cols: Sequence[str]) -> Tuple[
		pd.DataFrame, pd.DataFrame]:
		x = df.loc[:, dependent_var_cols]
		y = df.loc[:, INDEPENDENT_VARIABLE_COL_NAME]
		assert (len(x) == len(y))
		return x, y

	def __init__(self, smoothing_freq_cutoff: Integral):
		# self.parallelizer = parallelizer
		self.smoothing_freq_cutoff = smoothing_freq_cutoff

	def __call__(self, cross_validation_df: CrossValidationDataFrames):
		training_df = cross_validation_df.training
		print("Training using a total of {} dataframe row(s).".format(len(training_df)), file=sys.stderr)
		training_token_type_row_idxs = create_token_type_row_idx_mapping(training_df)
		smooth(training_token_type_row_idxs, self.smoothing_freq_cutoff)
		word_models, dependent_var_cols = self.__train_models(training_token_type_row_idxs.row_idxs, training_df)

		testing_df = cross_validation_df.testing
		print("Testing using a total of {} dataframe row(s).".format(len(testing_df)), file=sys.stderr)
		# testing_df.to_csv("/home/tshore/Downloads/testingoutput.tsv", sep='\t', encoding="utf-8", index_label="Index")
		testing_token_type_row_idxs = create_token_type_row_idx_mapping(testing_df)
		oov_model = word_models[OUT_OF_VOCABULARY_TOKEN_LABEL]
		for token_class, testing_inst_idxs in testing_token_type_row_idxs.row_idxs.items():
			print("Testing classifier for \"{}\".".format(token_class), file=sys.stderr)
			classifier = word_models.get(token_class, oov_model)
			testing_insts = (training_df.loc[inst_idx] for inst_idx in testing_inst_idxs)
			#assert (len(testing_insts) == len(testing_inst_idxs))
			testing_x, testing_y = self.__split_dependent_independent_vars(pd.DataFrame(testing_insts), dependent_var_cols)
			# testing_x.to_csv("/home/tshore/Downloads/testingoutput_{}.tsv".format(token_class), sep='\t',
			#				 encoding="utf-8", index_label="Index")
			# print(len(testing_x))
			# print(testing_x)
			decision_probs = classifier.predict_proba(testing_x)
			# decision_probs = classifier.predict_log_proba(testing_x)
			print(decision_probs)

	def __train_models(self, token_type_row_idxs: Iterable[
		Tuple[str, Sequence[Integral]]], training_df: pd.DataFrame):
		dependent_var_cols = tuple(
			col for col in training_df.columns if DEPENDENT_VARIABLE_COL_NAME_PATTERN.match(col))
		word_models = {}
		for (token_class, training_inst_idxs) in token_type_row_idxs.items():
			training_insts = (training_df.loc[training_inst_idx] for training_inst_idx in training_inst_idxs)
			#print(training_insts)
			training_x, training_y = self.__split_dependent_independent_vars(pd.DataFrame(training_insts), dependent_var_cols)
			model = LogisticRegression()
			model.fit(training_x, training_y)
			word_models[token_class] = model
		return word_models, dependent_var_cols


def create_token_type_row_idx_mapping(df: pd.DataFrame) -> TokenTypeRowIndexMapping:
	# df.to_csv("/home/tshore/Downloads/tokenrowidx.tsv", sep='\t', encoding="utf-8", index_label="Index")
	# There should be an equivalent event for each unique entity in the game
	observation_event_count = len(df["ENTITY"].unique())
	token_type_row_idxs = defaultdict(list)
	type_event_counts = Counter()
	for row in df.itertuples():
		# noinspection PyProtectedMember
		row_dict = row._asdict()
		idx = row.Index
		utts = row_dict[game_utterances.SessionGameRoundUtteranceSequenceFactory.UTTERANCE_SEQUENCE_COL_NAME]
		for utt in utts:
			tokens = utt.content
			for token in tokens:
				token_type_row_idxs[token].append(idx)
				type_event_counts[token] += 1

	with localcontext() as ctx:
		ctx.traps[Inexact] = True
		type_counts = tuple(
			(token_type, int((Decimal(count) / Decimal(observation_event_count)).to_integral_exact())) for
			token_type, count in
			type_event_counts.items())

	return TokenTypeRowIndexMapping(token_type_row_idxs, dict(type_counts))


def smooth(token_type_row_idxs: TokenTypeRowIndexMapping, smoothing_freq_cutoff: Integral):
	smoothed_token_types = frozenset(
		token_type for token_type, count in token_type_row_idxs.type_counts.items() if count < smoothing_freq_cutoff)
	row_idxs = token_type_row_idxs.row_idxs
	for token_type in smoothed_token_types:
		inst_idxs = row_idxs[token_type]
		del row_idxs[token_type]
		row_idxs[OUT_OF_VOCABULARY_TOKEN_LABEL].extend(inst_idxs)

	smoothed_row_idxs = row_idxs[OUT_OF_VOCABULARY_TOKEN_LABEL]
	print("Token class(es) used for smoothing: {}; {} data point(s) used as out-of-vocabulary instance(s).".format(
		smoothed_token_types, len(smoothed_row_idxs)), file=sys.stderr)


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

	cross_validation_data_frame_factory = CrossValidationDataFrameFactory(
		CachingSessionDataFrameFactory(game_utterances.SessionGameRoundUtteranceSequenceFactory(
			utterances.TokenSequenceFactory())))
	print("Creating cross-validation datasets from {} session(s).".format(len(infile_session_data)), file=sys.stderr)

	# parallelizer = Parallel(n_jobs = -2, backend = "multiprocessing")
	cross_validator = CrossValidator(smoothing_freq_cutoff)
	# NOTE: This must be lazily-generated lest a dataframe be created in-memory for each cross-validation fold
	for cross_validation_df in cross_validation_data_frame_factory(infile_session_data):
		cross_validator(cross_validation_df)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
