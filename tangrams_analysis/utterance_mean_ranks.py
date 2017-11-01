#!/usr/bin/env python3

import argparse
import sys

import cross_validation
import game_utterances
import session_data as sd
import utterances

DEFAULT_EXTRACTION_FILE_SUFFIX = ".extraction.tsv"


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
	# print(session_df)

	results_file_inpath = args.results_file
	print("Processing \"{}\".".format(results_file_inpath), file=sys.stderr)
	cv_results = cross_validation.read_results_file(results_file_inpath)
	# print(cv_results.dtypes)
	event_times = cv_results["EVENT_TIME"]
	print(event_times)


# print(event_times.transform(lambda val: type(val)))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
