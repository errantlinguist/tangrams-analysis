import itertools
from enum import Enum, unique
from numbers import Number
from string import ascii_uppercase
from typing import Callable, Iterable, Iterator, Sequence, \
	Tuple, TypeVar

import numpy as np
import pandas as pd

import game_events
import session_data as sd
import utterances

N = TypeVar('N', bound=Number)


class SessionGameRoundUtteranceFactory(object):
	@unique
	class EventColumn(Enum):
		ENTITY_ID = "ENTITY"
		EVENT_ID = "EVENT"
		EVENT_NAME = "NAME"
		EVENT_SUBMITTER = "SUBMITTER"
		EVENT_TIME = "TIME"
		ROUND_ID = "ROUND"

	UTTERANCE_SEQUENCE_COL_NAME = "UTTERANCES"

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
		event_submitter_ids = event_df[cls.EventColumn.EVENT_SUBMITTER.value]
		username_participant_ids = dict(
			cls.__username_participant_ids(event_submitter_ids.unique(), initial_participant_username))
		anonymized_event_submitter_ids = event_submitter_ids.transform(
			lambda submitter_id: username_participant_ids[submitter_id])
		event_df[cls.EventColumn.EVENT_SUBMITTER.value] = anonymized_event_submitter_ids

	@classmethod
	def __first_events(cls, df: pd.DataFrame) -> pd.DataFrame:
		min_event_time = df[cls.EventColumn.EVENT_TIME.value].min()
		return df.loc[df[cls.EventColumn.EVENT_TIME.value] == min_event_time]

	def __init__(self, token_seq_factory: Callable[[Iterable[str]], Sequence[str]]):
		self.__token_seq_factory = token_seq_factory

	def __call__(self, session: sd.SessionData) -> pd.DataFrame:
		event_data = game_events.read_events(session)
		source_participant_ids = event_data.source_participant_ids
		seg_utt_factory = utterances.SegmentUtteranceFactory(self.__token_seq_factory,
															 lambda source_id: source_participant_ids[source_id])
		event_df = event_data.events
		self.__anonymize_event_submitter_ids(event_df, event_data.initial_instructor_id)

		event_df.sort_values(
			[self.EventColumn.ROUND_ID.value, self.EventColumn.EVENT_ID.value, self.EventColumn.EVENT_TIME.value,
			 self.EventColumn.ENTITY_ID.value],
			inplace=True)

		# Get the events which describe the referent entity at the time a new turn is submitted
		turn_submission_events = event_df.loc[event_df[self.EventColumn.EVENT_NAME.value] == "nextturn.request"]
		round_time_turn_submission_events = turn_submission_events.groupby(self.EventColumn.ROUND_ID.value,
																		   as_index=False)
		# Ensure the chronologically-first events are chosen (should be unimportant because there should be only one turn submission event per round)
		round_first_turn_submission_events = round_time_turn_submission_events.apply(self.__first_events)
		round_first_turn_submission_event_times = round_first_turn_submission_events.loc[:,
												  self.EventColumn.EVENT_TIME.value]
		round_first_turn_submission_event_end_times = itertools.chain(
			(value for idx, value in round_first_turn_submission_event_times.iteritems()), (np.inf,))

		segments = utterances.read_segments(session.utts)
		utts = tuple(seg_utt_factory(segments))
		round_utts = tuple(game_round_utterances(round_first_turn_submission_event_end_times, utts)[1])
		round_first_turn_submission_events.loc[:, self.UTTERANCE_SEQUENCE_COL_NAME] = round_utts
		round_first_turn_submission_events.drop([self.EventColumn.EVENT_ID.value, self.EventColumn.EVENT_NAME.value], 1,
												inplace=True)

		# Assert that all entities are represented in each round's set of events
		assert len(round_first_turn_submission_events) % len(round_first_turn_submission_events[self.EventColumn.ENTITY_ID.value].unique()) == 0
		return round_first_turn_submission_events


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
