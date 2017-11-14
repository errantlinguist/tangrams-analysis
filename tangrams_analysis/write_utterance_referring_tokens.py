#!/usr/bin/env python3

import argparse
import os.path
import re
import sys

import java_properties_files
import session_data as sd

EVENT_LOG_FILENAME_PATTERN = re.compile("events-([^-].).*")
PLAYER_INITIAL_ROLE_LOG_MESSAGE_PATTERN = re.compile(r'.*?\"playerRoles\":\[\["MOVE_SUBMISSION\",\"([^\"]+)"\].*')

def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Anonymize tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	session_dirs = sd.walk_session_dirs(inpaths)
	for session_dir in session_dirs:
		session_desc_file = os.path.join(session_dir, "desc.properties")
		with open(session_desc_file, "r") as inf:
			props = java_properties_files.parse_properties(inf)
			#print(props)
			canonical_event_log_filename = props["canonicalEvents"]

		with open(os.path.join(session_dir, canonical_event_log_filename)) as inf:
			for line in inf:
				match = PLAYER_INITIAL_ROLE_LOG_MESSAGE_PATTERN.match(line)
				if match:
					initial_player_id = match.group(1)
					break



if __name__ == "__main__":
	__main(__create_argparser().parse_args())
