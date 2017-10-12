import csv
import decimal
import sys
from collections import defaultdict
from enum import Enum, unique
from typing import Any, Iterable, Iterator, Mapping, MutableSequence, Sequence, Tuple, Union

import session_data

_DECIMAL_INFINITY = decimal.Decimal("Infinity")
_DECIMAL_ONE = decimal.Decimal("1")
_ENTITY_ID_OFFSET = 1
_EVENT_ID_OFFSET = 1


class EntityData(object):
	def __init__(self, col_idxs: Mapping[str, int] = None, row: Sequence[Any] = None):
		self.col_idxs = col_idxs
		self.row = row

	def __bool__(self):
		return bool(self.col_idxs or self.row)

	def __eq__(self, other):
		return (self is other or (isinstance(other, type(self))
								  and self.__key == other.__key))

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	def attr(self, attr_name: str) -> Any:
		attr_value_idx = self.col_idxs[attr_name]
		return self.row[attr_value_idx]

	@property
	def hue(self) -> session_data.DECIMAL_VALUE_TYPE:
		return self.__data_col_attr(session_data.DataColumn.HUE)

	@property
	def is_referent(self) -> bool:
		return self.__data_col_attr(session_data.DataColumn.REFERENT_ENTITY)

	@property
	def is_selected(self) -> bool:
		return self.__data_col_attr(session_data.DataColumn.SELECTED_ENTITY)

	@property
	def shape(self) -> str:
		return self.__data_col_attr(session_data.DataColumn.SHAPE)

	def __data_col_attr(self, col: session_data.DataColumn) -> Any:
		return self.attr(col.value.name)

	@property
	def __key(self):
		return self.col_idxs, self.row


class Event(object):
	@unique
	class Attribute(Enum):
		ID = session_data.DataColumn.EVENT_ID.value
		NAME = session_data.DataColumn.EVENT_NAME.value
		SUBMITTER = session_data.DataColumn.SUBMITTER.value
		TIME = session_data.DataColumn.EVENT_TIME.value
		SCORE = session_data.DataColumn.SCORE.value

	def __init__(self, entities: Sequence[EntityData], attrs: Mapping[Attribute, Any] = None):
		if attrs is None:
			first_entity_desc = next(iter(entities))
			attrs = dict((attr, first_entity_desc.attr(attr.value.name)) for attr in Event.Attribute)
		self.entities = entities
		self.attrs = attrs

	def __eq__(self, other):
		return (self is other or (isinstance(other, type(self))
								  and self.__key == other.__key))

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	def entity(self, entity_id: int) -> EntityData:
		return self.entities[entity_id - _ENTITY_ID_OFFSET]

	def entities_by_id(self):
		return enumerate(self.entities, start=_ENTITY_ID_OFFSET)

	@property
	def event_id(self) -> int:
		return self.attrs[Event.Attribute.ID]

	@property
	def event_time(self) -> session_data.DECIMAL_VALUE_TYPE:
		return self.attrs[Event.Attribute.TIME]

	@property
	def referent_entities(self) -> Iterator[Tuple[int, EntityData]]:
		return ((entity_id, entity) for (entity_id, entity) in self.entities_by_id() if entity.is_referent)

	@property
	def round_id(self) -> int:
		first_entity_desc = next(iter(self.entities))
		return first_entity_desc.attr(GameRound.Attribute.ID.value.name)

	@property
	def score(self) -> int:
		return self.attrs[Event.Attribute.SCORE]

	@property
	def selected_entities(self) -> Iterator[Tuple[int, EntityData]]:
		return ((entity_id, entity) for (entity_id, entity) in self.entities_by_id() if entity.is_selected)

	@property
	def submitter(self) -> str:
		return self.attrs[Event.Attribute.SUBMITTER]

	def score_round_ratio(self) -> decimal.Decimal:
		try:
			round_count = self.round_id - 1
			result = decimal.Decimal(self.score) / decimal.Decimal(round_count)
		except (decimal.InvalidOperation, ZeroDivisionError):
			result = _DECIMAL_ONE
		return result

	def time_score_ratio(self) -> decimal.Decimal:
		try:
			result = decimal.Decimal(self.event_time) / decimal.Decimal(self.score)
		except ZeroDivisionError:
			result = _DECIMAL_INFINITY
		return result

	@property
	def __key(self):
		return self.entities, self.attrs


class EventData(object):
	def __init__(self, events: Iterator[Event], source_participant_ids: Mapping[str, str], initial_instructor_id: str):
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


class EventParticipantIdFactory(object):
	"""
		This is a hack to map the non-anonymized usernames in the event logs to anonymized utterance speaker IDs.
		TODO: Remove this after anonymizing all data
	"""

	def __init__(self, initial_instructor_id: str):
		self.initial_instructor_id = initial_instructor_id

	def __call__(self, event: Event) -> str:
		"""
		:param event: The event to get the participant ID for
		:return: Either "A" or "B"
		"""
		return "A" if event.submitter == self.initial_instructor_id else "B"


class GameRound(object):
	@unique
	class Attribute(Enum):
		ID = session_data.DataColumn.ROUND_ID.value

	def __init__(self, start_time: session_data.DECIMAL_VALUE_TYPE,
				 end_time: Union[session_data.DECIMAL_VALUE_TYPE, None], events: Sequence[Event]):
		self.start_time = start_time
		self.end_time = end_time
		self.events = events

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	def __eq__(self, other):
		return (self is other or (isinstance(other, type(self))
								  and self.__key == other.__key))

	def __ne__(self, other):
		return not (self == other)

	def event(self, event_id: int) -> Event:
		return self.events[event_id - _EVENT_ID_OFFSET]

	def events_by_id(self):
		return enumerate(self.events, start=_EVENT_ID_OFFSET)

	@property
	def initial_event(self) -> Event:
		return next(iter(self.events))

	@property
	def round_id(self):
		return self.initial_event.round_id

	@property
	def __key(self):
		return self.start_time, self.end_time, self.events


def create_game_rounds(events: Iterable[Event]) -> Iterator[GameRound]:
	round_events = defaultdict(list)
	for event in events:
		round_events[event.round_id].append(event)

	for event_list in round_events.values():
		event_list.sort(key=lambda e: e.event_id)

	ordered_event_lists = iter(sorted(round_events.items(), key=lambda item: item[0]))
	current_event_list = next(ordered_event_lists)
	current_events = current_event_list[1]
	current_round_start_time = current_events[0].event_time
	for next_event_list in ordered_event_lists:
		next_events = next_event_list[1]
		next_round_start_time = next_events[0].event_time
		yield GameRound(current_round_start_time, next_round_start_time, current_events)

		current_events = next_events
		current_round_start_time = next_round_start_time

	yield GameRound(current_round_start_time, None, current_events)


def read_events(session: session_data.SessionData) -> EventData:
	events_metadata = session.read_events_metadata()

	event_count = int(events_metadata[session_data.EventMetadataRow.EVENT_COUNT.value])
	entity_count = int(events_metadata[session_data.EventMetadataRow.ENTITY_COUNT.value])
	event_entity_descs = __read_event_entity_desc_matrix(session.events, event_count, entity_count)
	events = (Event(entity_descs) for entity_descs in event_entity_descs)

	participant_metadata = session.read_participant_metadata()
	participant_source_ids = participant_metadata[session_data.ParticipantMetadataRow.SOURCE_ID.value]
	interned_source_participant_ids = dict(
		(sys.intern(source_id), sys.intern(participant_id)) for (participant_id, source_id) in
		participant_source_ids.items())
	initial_instructor_id = sys.intern(events_metadata[session_data.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value])
	return EventData(events, interned_source_participant_ids, initial_instructor_id)


def __read_event_entity_desc_matrix(infile_path: str, event_count: int, entity_count: int) -> Tuple[
	Tuple[EntityData, ...], ...]:
	result = tuple(tuple(EntityData() for _ in range(entity_count)) for _ in range(event_count))

	with open(infile_path, 'r', encoding=session_data.ENCODING) as infile:
		rows = csv.reader(infile, dialect="excel-tab")
		col_idxs = dict((col_name, idx) for (idx, col_name) in enumerate(next(rows)))
		entity_id_col_idx = col_idxs[session_data.DataColumn.ENTITY_ID.value.name]
		event_id_col_idx = col_idxs[session_data.DataColumn.EVENT_ID.value.name]

		for row in rows:
			__transform_row_cell_values(row, col_idxs)
			event_id = row[event_id_col_idx]
			entity_descs = result[event_id - _EVENT_ID_OFFSET]
			entity_id = row[entity_id_col_idx]
			existing_entity_desc = entity_descs[entity_id - _ENTITY_ID_OFFSET]
			if existing_entity_desc:
				raise ValueError("Duplicate rows for event {}, entity {}.".format(event_id, entity_id))
			else:
				existing_entity_desc.col_idxs = col_idxs
				existing_entity_desc.row = row

	return result


def __transform_row_cell_value(row: MutableSequence[str], col_idxs: Mapping[str, int],
							   data_col: session_data.DataColumn):
	col_props = data_col.value
	idx = col_idxs[col_props.name]
	transformed_val = col_props.value_transformer(row[idx])
	row[idx] = transformed_val


def __transform_row_cell_values(row: MutableSequence[str], col_idxs: Mapping[str, int]):
	for data_col in session_data.DataColumn:
		__transform_row_cell_value(row, col_idxs, data_col)
