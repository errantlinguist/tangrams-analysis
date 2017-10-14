import itertools
from numbers import Number
from string import ascii_uppercase
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Sequence, \
	Tuple, TypeVar

import numpy as np
import pandas as pd

import game_events
import session_data as sd
import utterances

N = TypeVar('N', bound=Number)


class SessionGameRoundUtteranceFactory(object):
	ROUND_ID_OFFSET = 1

	__EVENT_SUBMITTER_COL_NAME = "SUBMITTER"
	__EVENT_TIME_COL_NAME = "TIME"
	__UTTERANCE_SEQUENCE_COL_NAME = "UTTERANCES"

	@staticmethod
	def __username_participant_ids(usernames: np.ndarray, initial_participant_username: str) -> Iterator[
		Tuple[str, str]]:
		assert initial_participant_username in usernames
		alphabetically_ordered_usernames = sorted(usernames)
		role_ordered_usernames = sorted(alphabetically_ordered_usernames,
										key=lambda username: -1 if username == initial_participant_username else 0)
		return zip(role_ordered_usernames, ascii_uppercase)

	@classmethod
	def __anonymize_event_submitter_ids(cls, event_df: pd.DataFrame, initial_participant_username: str):
		event_submitter_ids = event_df[cls.__EVENT_SUBMITTER_COL_NAME]
		username_participant_ids = dict(
			cls.__username_participant_ids(event_submitter_ids.unique(), initial_participant_username))
		anonymized_event_submitter_ids = event_submitter_ids.transform(
			lambda submitter_id: username_participant_ids[submitter_id])
		event_df[cls.__EVENT_SUBMITTER_COL_NAME] = anonymized_event_submitter_ids

	def __init__(self, token_seq_factory: Callable[[Iterable[str]], Sequence[str]]):
		self.__token_seq_factory = token_seq_factory

	def __call__(self, session: sd.SessionData) -> pd.DataFrame:
		event_data = game_events.read_events(session)
		source_participant_ids = event_data.source_participant_ids
		seg_utt_factory = utterances.SegmentUtteranceFactory(self.__token_seq_factory,
															 lambda source_id: source_participant_ids[source_id])
		event_df = event_data.events
		self.__anonymize_event_submitter_ids(event_df, event_data.initial_instructor_id)

		event_df.sort_values(self.__EVENT_TIME_COL_NAME, inplace=True)

		# Get the events which describe the referent entity at the time a new turn is submitted
		entity_reference_events = event_df[(event_df["NAME"] == "nextturn.request")]
		# Ensure the chronologically-first event is chosen (should be unimportant because there should be only one turn submission event per round)
		round_first_reference_events = entity_reference_events.groupby("ROUND").first()

		event_colums = tuple(round_first_reference_events.columns.values)
		round_first_reference_event_times = round_first_reference_events[self.__EVENT_TIME_COL_NAME]
		round_first_reference_event_end_times = itertools.chain(
			(value for idx, value in round_first_reference_event_times.iteritems()), (np.inf,))

		segments = utterances.read_segments(session.utts)
		utts = tuple(seg_utt_factory(segments))
		round_utts = tuple(game_round_utterances(round_first_reference_event_end_times, utts)[1])
		round_first_reference_events[self.__UTTERANCE_SEQUENCE_COL_NAME] = round_utts

		token_row_cols = tuple(itertools.chain(event_colums, ("SPEAKER", "TOKEN")))
		round_token_row_iters = (self.__create_token_rows(row, event_colums) for row in
								 round_first_reference_events.itertuples())
		token_row_value_iters = itertools.chain.from_iterable(round_token_row_iters)
		token_rows = (tuple(token_row_value_iter) for token_row_value_iter in token_row_value_iters)
		round_token_df = pd.DataFrame(token_rows, columns=token_row_cols)
		return round_token_df

	@classmethod
	def __create_token_rows(cls, row: NamedTuple, event_features: Sequence[str]) -> Iterator[Iterator[Any]]:
		# noinspection PyProtectedMember
		row_dict = row._asdict()
		event_feature_vals = tuple(row_dict[event_feature] for event_feature in event_features)
		for utt in row_dict[cls.__UTTERANCE_SEQUENCE_COL_NAME]:
			speaker_id = utt.speaker_id
			tokens = utt.content
			for token in tokens:
				linguistic_features = (speaker_id, token)
				yield itertools.chain(event_feature_vals, linguistic_features)


def game_round_utterances(round_start_time_iter: Iterator[N],
						  utts: Iterable[utterances.Utterance]) -> Tuple[Tuple[utterances.Utterance, ...], Iterator[
	Tuple[utterances.Utterance, ...]]]:
	first_round_start_time = next(round_start_time_iter)
	# Get utterances preceding the first round
	pre_game_utts = tuple(utt for utt in utts if utt.start_time < first_round_start_time)
	# TODO: optimize
	game_round_utts = (tuple(utt for utt in utts if utt.start_time < next_round_start_time) for next_round_start_time in
					   round_start_time_iter)
	return pre_game_utts, game_round_utts
