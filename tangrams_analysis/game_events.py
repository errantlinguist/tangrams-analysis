import sys
from typing import Mapping

import pandas as pd

import session_data as sd


class EventData(object):
	def __init__(self, events: pd.DataFrame, source_participant_ids: Mapping[str, str], initial_instructor_id: str):
		self.events = events
		self.source_participant_ids = source_participant_ids
		self.initial_instructor_id = initial_instructor_id

	def __eq__(self, other):
		return (self is other or (isinstance(other, type(self))
								  and self.__key == other.__key))

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	@property
	def __key(self):
		return self.initial_instructor_id, self.source_participant_ids, self.events


def read_events(session_data: sd.SessionData) -> EventData:
	participant_metadata = session_data.read_participant_metadata()
	participant_source_ids = participant_metadata[sd.ParticipantMetadataRow.SOURCE_ID.value]
	interned_source_participant_ids = dict(
		(sys.intern(source_id), sys.intern(participant_id)) for (participant_id, source_id) in
		participant_source_ids.items())

	session_metadata = session_data.read_session_metadata()
	initial_instructor_id = sys.intern(session_metadata[sd.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value])

	event_df = session_data.read_events()
	return EventData(event_df, interned_source_participant_ids, initial_instructor_id)
