#!/usr/bin/env python3

import argparse
import datetime
import sys
from collections import defaultdict

import dateutil.parser

import cross_validation
import game_utterances
import session_data as sd
import utterances

DEFAULT_EXTRACTION_FILE_SUFFIX = ".extraction.tsv"
EVENT_ABSOLUTE_TIME_COL_NAME = "ABSOLUTE_TIME"


def __create_absolute_time(start_time: datetime.datetime, offset_secs: float) -> datetime.datetime:
	timedelta = datetime.timedelta(seconds=offset_secs)
	return start_time + timedelta


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Rank dialogue utterances by the mean rank of the dialogue classification.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process for game utterances.")
	result.add_argument("-r", "--results-file", metavar="INPATH", required=True,
						help="The cross-validation results file to use for calculating the mean rank of each utterance.")
	result.add_argument("-s", "--suffix", metavar="SUFFIX", default=DEFAULT_EXTRACTION_FILE_SUFFIX,
						help="The extraction file suffix.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = tuple(sorted(sd.walk_session_data(inpaths), key=lambda item: item[0]))
	game_round_utt_factory = game_utterances.SessionGameRoundUtteranceSequenceFactory(
		utterances.TokenSequenceFactory())
	print("Mapping utterances to events from {} session(s).".format(len(infile_session_data)), file=sys.stderr)

	token_seqs_by_event_time = defaultdict(list)
	for session_dir, session_data in infile_session_data:
		session_df = game_round_utt_factory(session_data)
		events_metadata = session_data.read_events_metadata()
		session_start_timestamp = events_metadata["START_TIME"]
		session_start = dateutil.parser.parse(session_start_timestamp)

		session_df[EVENT_ABSOLUTE_TIME_COL_NAME] = session_df["TIME"].transform(
			lambda offset_secs: __create_absolute_time(session_start, offset_secs))
		for col_values in session_df.itertuples(index=False):
			# noinspection PyProtectedMember
			value_dict = col_values._asdict()
			event_time = value_dict[EVENT_ABSOLUTE_TIME_COL_NAME]
			token_seqs_for_time = token_seqs_by_event_time[event_time]
			utts = value_dict[game_utterances.SessionGameRoundUtteranceSequenceFactory.UTTERANCE_SEQUENCE_COL_NAME]
			token_seqs_for_time.extend(utt.content for utt in utts)

	for event_time, token_seqs in token_seqs_by_event_time.items():
		print(event_time)
		print(token_seqs)

	# TODO: Finish

	results_file_inpath = args.results_file
	print("Processing \"{}\".".format(results_file_inpath), file=sys.stderr)
	cv_results = cross_validation.read_results_file(results_file_inpath)
	# print(cv_results.dtypes)
	event_times = cv_results["EVENT_TIME"]


# print(event_times)



if __name__ == "__main__":
	__main(__create_argparser().parse_args())
