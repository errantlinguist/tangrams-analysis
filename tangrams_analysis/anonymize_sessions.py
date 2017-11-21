#!/usr/bin/env python3

import argparse
import csv
import os.path
import re
import sys
from collections import namedtuple
from enum import Enum, unique
from typing import Iterable, Mapping, Match

import iristk
import java_properties_files
import session_data as sd

PROPERTIES_FILENAME = "desc.properties"
PROPERTIES_FILE_ENCODING = "utf-8"

'''
NOTE: No extension is supplied so that files named e.g. "*-timesadjusted.txt" still get renamed
'''
EVENT_LOG_FILENAME_FORMAT_STRING = "events-{}"

EVENT_LOG_PLAYER_A_INITIAL_ROLE_FORMAT_STRING = '"playerRoles":[["MOVE_SUBMISSION","{}"],'
EVENT_LOG_PLAYER_B_INITIAL_ROLE_FORMAT_STRING = '],["WAITING_FOR_NEXT_MOVE","{}"]]'
EVENT_LOG_PLAYER_INITIAL_ROLE_PATTERN = re.compile(r'.*?\"playerRoles\":\[\["MOVE_SUBMISSION\",\"([^\"]+)"\].*')
EVENT_LOG_PLAYER_ID_FORMAT_STRING = "\"PLAYER_ID\":\"{}\""

SESSION_DIR_FILENAME_FORMAT_STRINGS = ("events-{}.txt", "img-info-{}.txt", "system-{}.log")

SYSTEM_LOGFILE_ENCODING = "UTF-8"
SYSTEM_LOGFILE_PATTERN = re.compile("system-([^.]+)\\.log")

RegexReplacementFormatter = namedtuple("RegexReplacementFormatter", "format_str pattern")


@unique
class ScreenshotFilenameFormatter(Enum):
	GAME_START = RegexReplacementFormatter("game-start-{}-{}.png",
										   re.compile("game-start-([^-]+)-([^.]+?)\\.png"))
	SELECTION = RegexReplacementFormatter("selection-piece-id{}-{}-{}.png",
										  re.compile("selection-piece-id([^-]+)-([^-]+)-([^.]+?)\\.png"))
	TURN = RegexReplacementFormatter("turn-{}-{}-{}.png",
									 re.compile("turn-([^-]+)-([^-]+)-([^.]+?)\\.png"))


class SessionAnonymizer(object):
	@staticmethod
	def rename(old_filepath: str, anonymized_filepath: str):
		if os.path.exists(anonymized_filepath):
			raise ValueError("Path \"{}\" already exists.".format(anonymized_filepath))
		else:
			print("Renaming \"{}\" to \"{}\".".format(old_filepath, anonymized_filepath))
			os.rename(old_filepath, anonymized_filepath)

	def __init__(self, initial_player_id: str):
		self.initial_player_id = initial_player_id
		self.screenshot_format_funcs = {
			ScreenshotFilenameFormatter.GAME_START: self.__anonymize_game_start_screenshot_filename,
			ScreenshotFilenameFormatter.SELECTION: self.__anonymize_selection_screenshot_filename,
			ScreenshotFilenameFormatter.TURN: self.__anonymize_turn_screenshot_filename}
		# noinspection PyTypeChecker
		assert len(self.screenshot_format_funcs) == len(ScreenshotFilenameFormatter)

	def __call__(self, session_dir: str, session_data: sd.SessionData, player_event_log_filenames: Mapping[str, str]):
		self.anonymize_events(session_data)
		self.anonymize_events_metadata(session_data)
		self.anonymize_player_event_log_files(player_event_log_filenames, session_dir)
		self.anonymize_props_file(session_dir)
		self.anonymize_system_logs(session_dir, player_event_log_filenames)

		self.rename_player_files(session_dir, player_event_log_filenames)
		self.rename_screenshot_files(os.path.join(session_dir, "screenshots"))
		self.rename_wav_files(session_dir, player_event_log_filenames)

	def __anonymize_game_start_screenshot_filename(self, match: Match, format_str: str) -> str:
		timestamp = match.group(1)
		player_id = match.group(2)
		anonymized_participant_id = self.__anonymize_player_id(player_id)
		return format_str.format(timestamp, anonymized_participant_id)

	def __anonymize_selection_screenshot_filename(self, match: Match, format_str: str) -> str:
		img_id = match.group(1)
		timestamp = match.group(2)
		player_id = match.group(3)
		anonymized_participant_id = self.__anonymize_player_id(player_id)
		return format_str.format(img_id, timestamp, anonymized_participant_id)

	def __anonymize_turn_screenshot_filename(self, match: Match, format_str: str) -> str:
		round_id = match.group(1)
		timestamp = match.group(2)
		player_id = match.group(3)
		anonymized_participant_id = self.__anonymize_player_id(player_id)
		return format_str.format(round_id, timestamp, anonymized_participant_id)

	def anonymize_events(self, session_data: sd.SessionData):
		print("Anonymizing event tabular data at \"{}\".".format(session_data.events))
		events = session_data.read_events()
		unique_submitter_ids = events["SUBMITTER"].unique()
		if len(unique_submitter_ids) != 2:
			raise ValueError(
				"There were not exactly 2 submitter IDs found in events file at \"{}\": {}".format(session_data.events,
																								   unique_submitter_ids))
		else:
			events["SUBMITTER"] = events["SUBMITTER"].transform(self.__anonymize_player_id)
			events.to_csv(session_data.events, index=False, sep=csv.excel_tab.delimiter, encoding=sd.ENCODING)

	def anonymize_events_metadata(self, session_data: sd.SessionData):
		print("Anonymizing event metadata at \"{}\".".format(session_data.events_metadata))
		events_metadata = session_data.read_events_metadata()
		if sd.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value not in events_metadata:
			raise ValueError(
				"Could not find attribute \"{}\" in \"{}\".".format(sd.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value,
																	session_data.events_metadata))
		else:
			events_metadata[sd.EventMetadataRow.INITIAL_INSTRUCTOR_ID.value] = self.initial_player_id
			with open(session_data.events_metadata, 'w', encoding=sd.ENCODING) as outf:
				writer = csv.writer(outf, dialect=sd.EVENTS_METADATA_CSV_DIALECT)
				writer.writerows(sorted(events_metadata, key=lambda item: item[0]))

	def anonymize_player_event_log_files(self, player_event_log_filenames: Mapping[str, str], session_dir: str):
		for player_event_log_filename in player_event_log_filenames.values():
			player_event_log_filepath = os.path.join(session_dir, player_event_log_filename)
			print("Anonymizing event log at \"{}\".".format(player_event_log_filepath))
			anonymized_lines = []
			with open(player_event_log_filepath, 'r', encoding=iristk.LOGFILE_ENCODING) as inf:
				for line in inf:
					anonymized_line = line
					# Try matching the longest player ID first
					for player_id in sorted(player_event_log_filenames, key=lambda elem: len(elem), reverse=True):
						anonymized_participant_id = self.__anonymize_player_id(player_id)
						initial_role_log_message_format_str = EVENT_LOG_PLAYER_A_INITIAL_ROLE_FORMAT_STRING if anonymized_participant_id == "A" else EVENT_LOG_PLAYER_B_INITIAL_ROLE_FORMAT_STRING
						initial_role_replacee_pattern = initial_role_log_message_format_str.format(player_id)
						initial_role_replacement_pattern = initial_role_log_message_format_str.format(
							anonymized_participant_id)
						anonymized_line = anonymized_line.replace(initial_role_replacee_pattern,
																  initial_role_replacement_pattern)

						player_id_replacee_pattern = EVENT_LOG_PLAYER_ID_FORMAT_STRING.format(player_id)
						player_id_replacement_pattern = EVENT_LOG_PLAYER_ID_FORMAT_STRING.format(
							anonymized_participant_id)
						anonymized_line = anonymized_line.replace(player_id_replacee_pattern,
																  player_id_replacement_pattern)
					anonymized_lines.append(anonymized_line)

			with open(player_event_log_filepath, 'w', encoding=iristk.LOGFILE_ENCODING) as outf:
				outf.writelines(anonymized_lines)

	def anonymize_props_file(self, session_dir: str):
		session_desc_file = os.path.join(session_dir, PROPERTIES_FILENAME)
		print("Anonymizing session properties at \"{}\".".format(session_desc_file))
		with open(session_desc_file, "r", encoding=PROPERTIES_FILE_ENCODING) as inf:
			props = java_properties_files.parse_properties(inf)
		player_data = props["player"]
		anonymized_player_data = {}
		for player_datum_id, player_datum in player_data.items():
			player_id = player_datum["id"]
			anonymized_participant_id = self.__anonymize_player_id(player_id)
			player_datum["id"] = anonymized_participant_id
			player_event_log_filename = player_datum["eventLog"]
			anonymized_player_event_log_filename = self.__anonymize_event_log_filename(player_event_log_filename,
																					   player_id)
			player_datum["eventLog"] = anonymized_player_event_log_filename

			# Anomyize the player data tuple IDs as well
			anonymized_player_data[anonymized_participant_id] = player_datum

		props["player"] = anonymized_player_data

		canonical_event_log_filename = props["canonicalEvents"]
		anonymized_canonical_event_log_filename = canonical_event_log_filename
		# Try matching the longest player ID first
		for player_id in sorted(player_data, key=lambda elem: len(elem), reverse=True):
			anonymized_canonical_event_log_filename = self.__anonymize_event_log_filename(
				anonymized_canonical_event_log_filename, player_id)
			if anonymized_canonical_event_log_filename != canonical_event_log_filename:
				break
		props["canonicalEvents"] = anonymized_canonical_event_log_filename
		prop_rows = []
		java_properties_files.append_prop_rows(props, prop_rows)

		with open(session_desc_file, 'w', encoding=PROPERTIES_FILE_ENCODING) as outf:
			for row in prop_rows:
				# This ensures that newlines are correctly printed
				print(row, file=outf)

	def anonymize_screenshot_filename(self, filename: str) -> str:
		result = filename
		for screenshot_formatter, format_func in self.screenshot_format_funcs.items():
			result, number_of_subs_made = screenshot_formatter.value.pattern.subn(
				lambda match: format_func(match, screenshot_formatter.value.format_str), result, count=1)
			if number_of_subs_made > 0:
				break
		return result

	def anonymize_system_logs(self, session_dir: str, player_ids = Iterable[str]):
		# Try matching the longest player ID first
		sorted_player_ids = tuple( sorted(player_ids, reverse=True))
		for file in os.listdir(session_dir):
			if SYSTEM_LOGFILE_PATTERN.match(file):
				filepath = os.path.join(session_dir, file)
				print("Anonymizing log file at \"{}\".".format(filepath))
				anonymized_lines = []
				with open(filepath, 'r', encoding=SYSTEM_LOGFILE_ENCODING) as inf:
					for line in inf:
						anonymized_line = line
						for player_id in sorted_player_ids:
							anonymized_player_id = self.__anonymize_player_id(player_id)
							anonymized_line = anonymized_line.replace(player_id, anonymized_player_id)
							anonymized_lines.append(anonymized_line)

				with open(filepath, 'w', encoding=SYSTEM_LOGFILE_ENCODING) as outf:
					outf.writelines(anonymized_lines)



	def rename_player_files(self, session_dir: str, player_ids: Iterable[str]):
		for player_id in player_ids:
			anonymized_participant_id = self.__anonymize_player_id(player_id)
			for format_str in SESSION_DIR_FILENAME_FORMAT_STRINGS:
				old_filepath = os.path.join(session_dir, format_str.format(player_id))
				if os.path.exists(old_filepath):
					anonymized_filepath = os.path.join(session_dir, format_str.format(anonymized_participant_id))
					self.rename(old_filepath, anonymized_filepath)

	def rename_screenshot_files(self, screenshot_dir: str):
		for file in os.listdir(screenshot_dir):
			old_filepath = os.path.join(screenshot_dir, file)
			anonymized_filename = self.anonymize_screenshot_filename(file)
			if anonymized_filename != file:
				anonymized_filepath = os.path.join(screenshot_dir, anonymized_filename)
				self.rename(old_filepath, anonymized_filepath)

	def rename_wav_files(self, session_dir: str, player_ids: Iterable[str]):
		for file in os.listdir(session_dir):
			# Try matching longest player name first
			anonymized_file = file
			for player_id in sorted(player_ids, reverse=True):
				anonymized_player_id = self.__anonymize_player_id(player_id)
				anonymized_file = anonymized_file.replace(player_id, anonymized_player_id)

			if anonymized_file != file:
				old_path = os.path.join(session_dir, file)
				anonymized_filepath = os.path.join(session_dir, anonymized_file)
				self.rename(old_path, anonymized_filepath)

	def __anonymize_event_log_filename(self, event_log_filename: str, player_id: str):
		anonymized_participant_id = self.__anonymize_player_id(player_id)
		filename_replacee = EVENT_LOG_FILENAME_FORMAT_STRING.format(player_id)
		filename_replacement = EVENT_LOG_FILENAME_FORMAT_STRING.format(anonymized_participant_id)
		return event_log_filename.replace(filename_replacee,
										  filename_replacement)

	def __anonymize_player_id(self, player_id: str) -> str:
		return "A" if player_id == self.initial_player_id else "B"


def parse_initial_player_id(event_log_file: str) -> str:
	result = None
	with open(event_log_file, 'r', encoding=iristk.LOGFILE_ENCODING) as inf:
		for line in inf:
			match = EVENT_LOG_PLAYER_INITIAL_ROLE_PATTERN.match(line)
			if match:
				result = match.group(1)
				break
	if result:
		return result
	else:
		raise ValueError("Could not find initial player ID in file \"{}\".".format(event_log_file))


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Anonymize tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths))
	for session_dir, session_data in sd.walk_session_data(inpaths):
		session_desc_file = os.path.join(session_dir, PROPERTIES_FILENAME)
		print("Reading session properties from \"{}\".".format(session_desc_file))
		with open(session_desc_file, "r", encoding=PROPERTIES_FILE_ENCODING) as inf:
			props = java_properties_files.parse_properties(inf)
		canonical_event_log_filename = props["canonicalEvents"]
		player_data = props["player"]
		player_event_log_filenames = {}
		for player_datum in player_data.values():
			player_id = sys.intern(player_datum["id"])
			player_event_log_filename = player_datum["eventLog"]
			player_event_log_filenames[player_id] = player_event_log_filename

		if len(player_event_log_filenames) != 2:
			raise ValueError("Not exactly two players described in file \"{}\".".format(session_desc_file))

		event_log_file = os.path.join(session_dir, canonical_event_log_filename)
		initial_player_id = sys.intern(parse_initial_player_id(event_log_file))
		anonymizer = SessionAnonymizer(initial_player_id)
		anonymizer(session_dir, session_data, player_event_log_filenames)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
