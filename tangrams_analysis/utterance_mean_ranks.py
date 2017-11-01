#!/usr/bin/env python3

import argparse
import csv
import sys

import pandas as pd

import iristk
import game_utterances
import utterances
import session_data as sd

DEFAULT_EXTRACTION_FILE_SUFFIX = ".extraction.tsv"
ENCODING = 'utf-8'

INPUT_DTYPES = {"Cleaning.DISFLUENCIES": bool, "Cleaning.DUPLICATES": bool, "Cleaning.FILLERS": bool}


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
	for session_dir, session_data in infile_session_data:
		session_df = game_round_utt_factory(session_data)
		print(session_df)

	results_file_inpath = args.results_file
	print("Processing \"{}\".".format(results_file_inpath), file=sys.stderr)
	cv_results = pd.read_csv(results_file_inpath, sep=csv.excel_tab.delimiter, dialect=csv.excel_tab, float_precision="round_trip",
							 encoding=ENCODING, memory_map=True, parse_dates=["TIME", "EVENT_TIME"],
							 date_parser=iristk.parse_timestamp,
							 dtype=INPUT_DTYPES)
	# print(cv_results.dtypes)
	event_times = cv_results["EVENT_TIME"]
	print(event_times)




# print(event_times.transform(lambda val: type(val)))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
