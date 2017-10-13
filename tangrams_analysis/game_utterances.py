import itertools
from numbers import Number
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, \
	Tuple, TypeVar

import numpy as np
import pandas as pd

import game_events
import session_data as sd
import utterances

N = TypeVar('N', bound=Number)


class GameRoundUtterances(object):
	"""
	A class associating game rounds with the dialogues for each.
	"""

	def __init__(self, game_round_utts: Sequence[Tuple[game_events.GameRound, Sequence[utterances.Utterance]]],
				 round_instructor_ids: Mapping[int, str]):
		self.game_round_utts = game_round_utts
		self.round_instructor_ids = round_instructor_ids

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)


class SessionGameRoundUtteranceFactory(object):
	ROUND_ID_OFFSET = 1

	__UTTERANCE_SEQUENCE_COL_NAME = "UTTERANCES"

	def __init__(self, token_seq_factory: Callable[[Iterable[str]], Sequence[str]]):
		self.token_seq_factory = token_seq_factory

	def __call__(self, session: sd.SessionData) -> pd.DataFrame:
		event_data = game_events.read_events(session)
		source_participant_ids = event_data.source_participant_ids
		seg_utt_factory = utterances.SegmentUtteranceFactory(self.token_seq_factory,
															 lambda source_id: source_participant_ids[source_id])
		event_df = event_data.events

		time_col_name = "TIME"
		event_df.sort_values(time_col_name, inplace=True)

		# Get the events which describe the referent entity at the time a new turn is submitted
		entity_reference_events = event_df[(event_df["REFERENT"] == True) & (event_df["NAME"] == "nextturn.request")]
		# Ensure the chronologically-first event is chosen (should be unimportant because there should be only one turn submission event per round)
		round_first_reference_events = entity_reference_events.groupby("ROUND").first()

		event_colums = tuple(round_first_reference_events.columns.values)
		round_first_reference_event_times = round_first_reference_events[time_col_name]
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
	def __create_token_rows(cls, row, event_features: Sequence[str]) -> Iterator[Iterator[Any]]:
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
