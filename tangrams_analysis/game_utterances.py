import sys
from numbers import Number
from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Sequence, \
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


def add_round_start_time(group_df: pd.DataFrame) -> pd.DataFrame:
	round_start_time = group_df["TIME"].transform('min')
	group_df["ROUND_START_TIME"] = round_start_time
	return group_df


class SessionGameRoundUtteranceFactory(object):
	ROUND_ID_OFFSET = 1

	__GameRoundUtteranceSequence = TypeVar("__GameRoundUtteranceSequence",
										   bound=Sequence[Tuple[Optional[Tuple[N, N]], Sequence[utterances.Utterance]]])

	@staticmethod
	def __trim_game_round_utterances(
			game_round_utts: __GameRoundUtteranceSequence) -> __GameRoundUtteranceSequence:
		"""
		Trims the first set of utterances if it represents language before the game started.
		:param game_round_utts: The game round utterances to trim.
		:return: The utterance sequence minus the first entry if it represents language before the first game round.
		"""
		if game_round_utts[0][0] is None:
			result = game_round_utts[1:]
		else:
			result = game_round_utts
		return result

	def __init__(self, token_seq_factory: Callable[[Iterable[str]], Sequence[str]]):
		self.token_seq_factory = token_seq_factory

	def __call__(self, session: sd.SessionData) -> GameRoundUtterances:
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
		next_round_first_reference_event_times = round_first_reference_event_times.shift(-1).fillna(np.inf)
		round_timespans = zip(round_first_reference_event_times, next_round_first_reference_event_times)

		segments = utterances.read_segments(session.utts)
		utts = seg_utt_factory(segments)
		round_utts = self.__trim_game_round_utterances(tuple(zip_game_round_utterances(round_timespans, iter(utts))))

		print("Round count : {}".format(round_first_reference_events.shape[0]), file=sys.stderr)
		print("Utterance set count : {}".format(len(round_utts)), file=sys.stderr)
		for utts in round_utts:
			print(utts)
		round_first_reference_events["UTTERANCES"] = round_utts


def zip_game_round_utterances(round_timespan_iter: Iterator[Tuple[N, N]],
							  utt_iter: Iterator[utterances.Utterance]) -> Iterator[
	Tuple[Optional[Tuple[N, N]], List[utterances.Utterance]]]:
	current_round_timespan = None
	current_round_utts = []
	next_round_timespan = next(round_timespan_iter)
	next_round_start_time = next_round_timespan[0]

	try:
		for utt in utt_iter:
			if utt.start_time < next_round_start_time:
				current_round_utts.append(utt)
			else:
				result = current_round_timespan, current_round_utts
				if current_round_timespan is None:
					if current_round_utts:
						yield result
				else:
					yield result

				current_round_timespan = next_round_timespan
				current_round_utts = [utt]
				next_round_timespan = next(round_timespan_iter)
				next_round_start_time = next_round_timespan[0]

		# Return the rest of the rounds with empty utterance lists
		for remaining_round_timespan in round_timespan_iter:
			yield remaining_round_timespan, []

	except StopIteration:
		# There are no more following events; The rest of the utterances must belong to the event directly following this one
		current_round_utts.extend(utt_iter)

	yield current_round_timespan, current_round_utts
