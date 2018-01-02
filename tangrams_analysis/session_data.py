import csv
import os
from enum import Enum, unique
from typing import Dict, Iterator, Iterable, Tuple

import pandas as pd

ENCODING = 'utf-8'
PARTICIPANT_METADATA_CSV_DIALECT = csv.excel_tab
SESSION_METADATA_CSV_DIALECT = csv.excel_tab

_EVENT_FILE_DTYPES = {"NAME": "category", "SHAPE": "category", "SUBMITTER": "category"}


@unique
class EventMetadataRow(Enum):
	ENTITY_COUNT = "ENTITY_COUNT"
	EVENT_COUNT = "EVENT_COUNT"
	INITIAL_INSTRUCTOR_ID = "INITIAL_INSTRUCTOR_ID"
	ROUND_COUNT = "ROUND_COUNT"


@unique
class ParticipantMetadataRow(Enum):
	PARTICIPANT_ID = "PARTICIPANT_ID"
	SOURCE_ID = "SOURCE_ID"


@unique
class SessionDatum(Enum):
	EVENTS = "events.tsv"
	SESSION_METADATA = "session-metadata.tsv"
	PARTICIPANT_METADATA = "participant-metadata.tsv"
	UTTERANCES = "utts.xml"

	@property
	def canonical_filename(self):
		return self.value


__SESSION_DATA_FILENAMES = frozenset(datum.canonical_filename for datum in SessionDatum)


class SessionData(object):
	def __init__(self, session_file_prefix: str):
		self.events = os.path.join(session_file_prefix, SessionDatum.EVENTS.canonical_filename)
		self.session_metadata = os.path.join(session_file_prefix, SessionDatum.SESSION_METADATA.canonical_filename)
		self.participant_metadata = os.path.join(session_file_prefix,
												 SessionDatum.PARTICIPANT_METADATA.canonical_filename)
		self.utts = os.path.join(session_file_prefix, SessionDatum.UTTERANCES.canonical_filename)

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
		return pd.read_csv(self.events, sep=csv.excel_tab.delimiter, dialect=csv.excel_tab,
						   float_precision="round_trip",
						   encoding=ENCODING, memory_map=True, dtype=_EVENT_FILE_DTYPES)

	def read_session_metadata(self) -> Dict[str, str]:
		with open(self.session_metadata, 'r', encoding=ENCODING) as infile:
			rows = csv.reader(infile, dialect=SESSION_METADATA_CSV_DIALECT)
			return dict(rows)

	def read_participant_metadata(self) -> Dict[str, Dict[str, str]]:
		result = {}
		with open(self.participant_metadata, 'r', encoding=ENCODING) as infile:
			rows = csv.reader(infile, dialect=PARTICIPANT_METADATA_CSV_DIALECT)
			headed_rows = dict((row[0], row[1:]) for row in rows)
		participant_ids = headed_rows[ParticipantMetadataRow.PARTICIPANT_ID.value]
		participant_id_idxs = tuple((participant_id, idx) for (idx, participant_id) in enumerate(participant_ids))
		non_header_rows = ((metadatum_name, participant_values) for (metadatum_name, participant_values) in
						   headed_rows.items() if metadatum_name != ParticipantMetadataRow.PARTICIPANT_ID.value)
		for metadatum_name, participant_values in non_header_rows:
			participant_value_dict = dict(
				(participant_id, participant_values[idx]) for (participant_id, idx) in participant_id_idxs)
			result[metadatum_name] = participant_value_dict

		return result

	@property
	def __key(self):
		return self.events, self.session_metadata, self.utts


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
