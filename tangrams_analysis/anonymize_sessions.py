#!/usr/bin/env python3

import argparse
import csv
import os.path
import re
import sys

import java_properties_files
import session_data as sd

EVENT_LOG_FILENAME_PATTERN = re.compile("events-([^-].).*")
PLAYER_INITIAL_ROLE_LOG_MESSAGE_PATTERN = re.compile(r'.*?\"playerRoles\":\[\["MOVE_SUBMISSION\",\"([^\"]+)"\].*')


def anonymize_events(session_data: sd.SessionData, initial_player_id: str):
	events = session_data.read_events()
	unique_submitter_ids = events["SUBMITTER"].unique()
	if len(unique_submitter_ids) != 2:
		raise ValueError(
			"There were not exactly 2 submitter IDs found in events file at \"{}\": {}".format(session_data.events,
																							   unique_submitter_ids))
	else:
		events["SUBMITTER"] = events["SUBMITTER"].transform(
			lambda submitter_id: "A" if submitter_id == initial_player_id else "B")
	# events.to_csv(session_data.events, index=False, sep=csv.excel_tab.delimiter, encoding=sd.ENCODING)


def anonymize_events_metadata(session_data: sd.SessionData, initial_player_id: str):
	events_metadata = session_data.read_events_metadata()
	if sd.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value not in events_metadata:
		raise ValueError(
			"Could not find attribute \"{}\" in \"{}\".".format(sd.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value,
																session_data.events_metadata))
	else:
		events_metadata[sd.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value] = initial_player_id
		#with open(session_data.events_metadata, 'w', encoding=sd.ENCODING) as outf:
		#	writer = csv.writer(outf, dialect=sd.EVENTS_METADATA_CSV_DIALECT)
		# writer.writerows(sorted(events_metadata, key=lambda item: item[0]))


def parse_initial_player_id(event_log_file: str) -> str:
	result = None
	with open(event_log_file, 'r', encoding=sd.ENCODING) as inf:
		for line in inf:
			match = PLAYER_INITIAL_ROLE_LOG_MESSAGE_PATTERN.match(line)
			if match:
				result = match.group(1)
				break
	if result:
		return result
	else:
		raise ValueError("Could not find initial player ID in file \"{}\"".format(event_log_file))


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Anonymize tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	for session_dir, session_data in sd.walk_session_data(inpaths):
		session_desc_file = os.path.join(session_dir, "desc.properties")
		with open(session_desc_file, "r", encoding=sd.ENCODING) as inf:
			props = java_properties_files.parse_properties(inf)
			canonical_event_log_filename = props["canonicalEvents"]
			player_data = props["player"]
			player_event_log_filenames = {}
			for player_datum in player_data.values():
				player_id = player_datum["id"]
				player_event_log_filename = player_datum["eventLog"]
				player_event_log_filenames[player_id] = player_event_log_filename

		event_log_file = os.path.join(session_dir, canonical_event_log_filename)
		initial_player_id = parse_initial_player_id(event_log_file)

		anonymize_events(session_data, initial_player_id)
		anonymize_events_metadata(session_data, initial_player_id)
		print(player_event_log_filenames)

if __name__ == "__main__":
	__main(__create_argparser().parse_args())
