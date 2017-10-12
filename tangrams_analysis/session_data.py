import csv
import os
from decimal import Decimal
from enum import Enum, unique
from typing import Any, Callable, Dict, Iterator, Iterable, Tuple

import pandas as pd

DECIMAL_VALUE_TYPE = Decimal
ENCODING = 'utf-8'
NULL_VALUE_REPR = '?'

_DECIMAL_INFINITY = DECIMAL_VALUE_TYPE('Infinity')
_PARTICIPANT_METADATA_HEADER_ROW_NAME = "PARTICIPANT_ID"

__DECIMAL_VALUE_POOL = {}


class DataColumnProperties(object):
	def __init__(self, name: str, value_transformer: Callable[[str], Any]):
		self.name = name
		self.value_transformer = value_transformer

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)


def fetch_decimal_value(cell_value: str) -> DECIMAL_VALUE_TYPE:
	try:
		result = __DECIMAL_VALUE_POOL[cell_value]
	except KeyError:
		result = DECIMAL_VALUE_TYPE(cell_value)
		__DECIMAL_VALUE_POOL[cell_value] = result
	return result


def _is_truth_cell_value(val: str) -> bool:
	return val == "true"


@unique
class DataColumn(Enum):
	BLUE = DataColumnProperties("BLUE", int)
	EDGE_COUNT = DataColumnProperties("EDGE_COUNT", int)
	ENTITY_ID = DataColumnProperties("ENTITY", int)
	EVENT_ID = DataColumnProperties("EVENT", int)
	EVENT_NAME = DataColumnProperties("NAME", str)
	EVENT_TIME = DataColumnProperties("TIME", fetch_decimal_value)
	GREEN = DataColumnProperties("BLUE", int)
	HUE = DataColumnProperties("HUE", fetch_decimal_value)
	POSITION_X = DataColumnProperties("POSITION_X", fetch_decimal_value)
	POSITION_Y = DataColumnProperties("POSITION_Y", fetch_decimal_value)
	REFERENT_ENTITY = DataColumnProperties("REFERENT", _is_truth_cell_value)
	RED = DataColumnProperties("RED", int)
	ROUND_ID = DataColumnProperties("ROUND", int)
	SCORE = DataColumnProperties("SCORE", int)
	SELECTED_ENTITY = DataColumnProperties("SELECTED", _is_truth_cell_value)
	SIZE = DataColumnProperties("SIZE", fetch_decimal_value)
	SHAPE = DataColumnProperties("SHAPE", str)
	SUBMITTER = DataColumnProperties("SUBMITTER", str)


@unique
class EventMetadataRow(Enum):
	ENTITY_COUNT = "ENTITY_COUNT"
	EVENT_COUNT = "EVENT_COUNT"
	INITIAL_INSTRUCTOR_ID = "INITIAL_INSTRUCTOR_ID"
	ROUND_COUNT = "ROUND_COUNT"


@unique
class ParticipantMetadataRow(Enum):
	SOURCE_ID = "SOURCE_ID"


@unique
class SessionDatum(Enum):
	EVENTS = "events.tsv"
	EVENTS_METADATA = "events-metadata.tsv"
	PARTICIPANT_METADATA = "participant-metadata.tsv"
	UTTS = "utts.xml"

	@property
	def canonical_filename(self):
		return self.value


__SESSION_DATA_FILENAMES = frozenset(datum.canonical_filename for datum in SessionDatum)


class SessionData(object):
	def __init__(self, session_file_prefix: str):
		self.events = os.path.join(session_file_prefix, SessionDatum.EVENTS.canonical_filename)
		self.events_metadata = os.path.join(session_file_prefix, SessionDatum.EVENTS_METADATA.canonical_filename)
		self.participant_metadata = os.path.join(session_file_prefix,
												 SessionDatum.PARTICIPANT_METADATA.canonical_filename)
		self.utts = os.path.join(session_file_prefix, SessionDatum.UTTS.canonical_filename)

	def __eq__(self, other):
		return (self is other or (isinstance(other, type(self))
								  and self.__key == other.__key))

	def __hash__(self):
		return hash(self.__key)

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	def read_events(self) -> pd.DataFrame:
		return pd.read_csv(self.events, sep='\t', dialect=csv.excel_tab, float_precision="high",
						   encoding=ENCODING, memory_map=True)

	def read_events_metadata(self) -> Dict[str, str]:
		with open(self.events_metadata, 'r', encoding=ENCODING) as infile:
			rows = csv.reader(infile, dialect="excel-tab")
			return dict(rows)

	def read_participant_metadata(self) -> Dict[str, Dict[str, str]]:
		result = {}
		with open(self.participant_metadata, 'r', encoding=ENCODING) as infile:
			rows = csv.reader(infile, dialect="excel-tab")
			headed_rows = dict((row[0], row[1:]) for row in rows)
		participant_ids = headed_rows[_PARTICIPANT_METADATA_HEADER_ROW_NAME]
		participant_id_idxs = tuple((participant_id, idx) for (idx, participant_id) in enumerate(participant_ids))
		non_header_rows = ((metadatum_name, participant_values) for (metadatum_name, participant_values) in
						   headed_rows.items() if metadatum_name != _PARTICIPANT_METADATA_HEADER_ROW_NAME)
		for metadatum_name, participant_values in non_header_rows:
			participant_value_dict = dict(
				(participant_id, participant_values[idx]) for (participant_id, idx) in participant_id_idxs)
			result[metadatum_name] = participant_value_dict

		return result

	@property
	def __key(self):
		return self.events, self.events_metadata, self.utts


def is_session_dir(filenames: Iterable[str]) -> bool:
	result = False

	filenames_to_find = set(__SESSION_DATA_FILENAMES)
	for filename in filenames:
		filenames_to_find.discard(filename)
		if not filenames_to_find:
			result = True
			break

	return result


def walk_session_data(inpaths: Iterable[str]) -> Iterator[Tuple[str, SessionData]]:
	session_dirs = walk_session_dirs(inpaths)
	return ((session_dir, SessionData(session_dir)) for session_dir in session_dirs)


def walk_session_dirs(inpaths: Iterable[str]) -> Iterator[str]:
	for inpath in inpaths:
		for dirpath, _, filenames in os.walk(inpath, followlinks=True):
			if is_session_dir(filenames):
				yield dirpath
