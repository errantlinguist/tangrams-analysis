"""
Functionalities for manipulating game log data with the corresponding transcribed utterances for the game.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

from enum import Enum, unique
from numbers import Integral, Number
from string import ascii_uppercase
from typing import Callable, Iterable, Iterator, Sequence, \
	Tuple, TypeVar

import numpy as np
import pandas as pd

import game_events
import session_data as sd
import utterances

N = TypeVar('N', bound=Number)


@unique
class EventColumn(Enum):
	DYAD_ID = "DYAD"
	ENTITY_ID = "ENTITY"
	ENTITY_SHAPE = "SHAPE"
	EVENT_ID = "EVENT"
	EVENT_NAME = "NAME"
	EVENT_SUBMITTER = "SUBMITTER"
	EVENT_TIME = "TIME"
	ROUND_ID = "ROUND"


class GameRoundUtteranceSequenceFactory(object):
	def __init__(self, df: pd.DataFrame, utts: Iterable[utterances.Utterance]):
		self.df = df
		self.utts = utts
		self.cache = {}

	def __call__(self, row: pd.Series) -> Tuple[utterances.Utterance, ...]:
		round_id = row[EventColumn.ROUND_ID.value]
		try:
			result = self.cache[round_id]
		except KeyError:
			result = tuple(self.__find_round_utts(round_id, row))
			self.cache[round_id] = result
		return result

	def __find_round_utts(self, round_id: Integral, row: pd.Series) -> Iterator[utterances.Utterance]:
		start_time = row[EventColumn.EVENT_TIME.value]
		following_round_events = self.df.loc[(self.df[EventColumn.ROUND_ID.value] > round_id)]
		end_time = np.inf if len(following_round_events) < 1 else following_round_events[
			EventColumn.EVENT_TIME.value].min()
		return (utt for utt in self.utts if start_time <= utt.start_time < end_time)


class SessionGameRoundUtteranceSequenceFactory(object):
	UTTERANCE_SEQUENCE_COL_NAME = "UTTERANCES"

	@staticmethod
	def __first_events(df: pd.DataFrame) -> pd.DataFrame:
		min_event_time = df.loc[:, EventColumn.EVENT_TIME.value].min()
		return df.loc[df[EventColumn.EVENT_TIME.value] == min_event_time]

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
		event_submitter_ids = event_df[EventColumn.EVENT_SUBMITTER.value]
		username_participant_ids = dict(
			cls.__username_participant_ids(event_submitter_ids.unique(), initial_participant_username))
		anonymized_event_submitter_ids = event_submitter_ids.transform(
			lambda submitter_id: username_participant_ids[submitter_id])
		event_df[EventColumn.EVENT_SUBMITTER.value] = anonymized_event_submitter_ids

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
			[EventColumn.ROUND_ID.value, EventColumn.EVENT_ID.value, EventColumn.EVENT_TIME.value,
			 EventColumn.ENTITY_ID.value],
			inplace=True)

		# Get the events which describe the referent entity at the time a new turn is submitted
		turn_submission_events = event_df.loc[event_df[EventColumn.EVENT_NAME.value] == "nextturn.request"]
		round_time_turn_submission_events = turn_submission_events.groupby(EventColumn.ROUND_ID.value,
																		   as_index=False)
		# Ensure the chronologically-first events are chosen (should be unimportant because there should be only one turn submission event per round)
		round_first_turn_submission_events = round_time_turn_submission_events.apply(self.__first_events)

		utts = tuple(seg_utt_factory(utterances.read_segments(session.utts)))
		round_utt_seq_factory = GameRoundUtteranceSequenceFactory(round_first_turn_submission_events, utts)
		round_first_turn_submission_events.loc[:,
		self.UTTERANCE_SEQUENCE_COL_NAME] = round_first_turn_submission_events.apply(round_utt_seq_factory, axis=1)

		round_first_turn_submission_events.drop([EventColumn.EVENT_ID.value, EventColumn.EVENT_NAME.value, "SELECTED"],
												1,
												inplace=True)
		# Assert that all entities are represented in each round's set of events
		assert len(round_first_turn_submission_events) % round_first_turn_submission_events[EventColumn.ENTITY_ID.value].nunique() == 0
		return round_first_turn_submission_events
