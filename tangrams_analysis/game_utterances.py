import itertools
from numbers import Number
from typing import Callable, Iterable, Iterator, Mapping, Sequence, \
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

	def __init__(self, token_seq_factory: Callable[[Iterable[str]], Sequence[str]]):
		self.token_seq_factory = token_seq_factory

	@staticmethod
	def create_token_rows(row) -> Iterator[pd.Series]:
		utts = row.UTTERANCES
		for utt in utts:
			speaker_id = utt.speaker_id
			tokens = utt.content
			for token in tokens:
				#token_row = row.copy(deep=False)
				token_row = pd.Series(row, row._fields)
				#print(token_row)
				token_row.drop("UTTERANCES", inplace=True)

				token_row["SPEAKER"] = speaker_id
				token_row["TOKEN"] = token
				yield token_row

	def __call__(self, session: sd.SessionData) -> pd.DataFrame:
		event_data = game_events.read_events(session)
		source_participant_ids = event_data.source_participant_ids
		seg_utt_factory = utterances.SegmentUtteranceFactory(self.token_seq_factory,
															 lambda source_id: source_participant_ids[source_id])
		event_df = event_data.events
		event_df.sort_values("TIME", inplace=True)

		# Get the events which describe the referent entity at the time a new turn is submitted
		entity_reference_events = event_df[(event_df["REFERENT"] == True) & (event_df["NAME"] == "nextturn.request")]
		# Ensure the chronologically-first event is chosen (should be unimportant because there should be only one turn submission event per round)
		round_first_reference_events = entity_reference_events.groupby("ROUND").first()
		round_first_reference_event_times = round_first_reference_events["TIME"]
		round_first_reference_event_end_times = itertools.chain(
			(value for idx, value in round_first_reference_event_times.iteritems()), (np.inf,))

		segments = utterances.read_segments(session.utts)
		utts = tuple(seg_utt_factory(segments))
		round_utts = tuple(game_round_utterances(round_first_reference_event_end_times, utts)[1])
		round_first_reference_events["UTTERANCES"] = round_utts
		# event_token_df = round_first_reference_events.transform(self.create_token_rows, axis=1)
		# print(event_token_df)
		# print(type(round_first_reference_events))
		# token_rows
		#blah = round_first_reference_events.apply(self.create_token_row_tuple, axis=1)
		#print(blah)
		#token_rows = itertools.chain.from_iterable(self.create_token_rows(row) for row in round_first_reference_events.itertuples())
		#for token_row in token_rows:
		#	print(token_row)

		token_rows = []
		for row in round_first_reference_events.itertuples():
			#print(help(pd.core.frame.Pandas))
			#print(type(row))
			round_token_rows = self.create_token_rows(row)
			token_rows.extend(round_token_rows)

		#print("Token training instances: {}".format(len(token_rows)))
		round_token_df = pd.DataFrame(token_rows)
		#print(round_token_df)

		return round_token_df


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