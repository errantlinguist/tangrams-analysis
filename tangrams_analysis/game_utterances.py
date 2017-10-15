import itertools
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
	UTTERANCE_SEQUENCE_COL_NAME = "UTTERANCES"

	__EVENT_ID_COL_NAME = "EVENT"
	__EVENT_NAME_COL_NAME = "NAME"
	__EVENT_SUBMITTER_COL_NAME = "SUBMITTER"
	__EVENT_TIME_COL_NAME = "TIME"

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

		event_df.sort_values("ROUND", self.__EVENT_ID_COL_NAME, self.__EVENT_TIME_COL_NAME, "ENTITY", inplace=True)

		# Get the events which describe the referent entity at the time a new turn is submitted
		entity_reference_events = event_df.loc[event_df[self.__EVENT_NAME_COL_NAME] == "nextturn.request"]
		# Ensure the chronologically-first event is chosen (should be unimportant because there should be only one turn submission event per round)
		round_first_reference_events = entity_reference_events.groupby("ROUND", as_index=False).first()
		round_first_reference_event_times = round_first_reference_events.loc[:, self.__EVENT_TIME_COL_NAME]
		round_first_reference_event_end_times = itertools.chain(
			(value for idx, value in round_first_reference_event_times.iteritems()), (np.inf,))

		segments = utterances.read_segments(session.utts)
		utts = tuple(seg_utt_factory(segments))
		round_utts = tuple(game_round_utterances(round_first_reference_event_end_times, utts)[1])
		round_first_reference_events.loc[:, self.UTTERANCE_SEQUENCE_COL_NAME] = round_utts
		round_first_reference_events.drop([self.__EVENT_ID_COL_NAME, self.__EVENT_NAME_COL_NAME], 1, inplace=True)
		return round_first_reference_events


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
