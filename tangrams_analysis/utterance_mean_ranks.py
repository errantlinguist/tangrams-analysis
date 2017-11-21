#!/usr/bin/env python3

import argparse
import datetime
import os.path
import sys

import pandas as pd

import cross_validation
import game_utterances
import session_data as sd
import utterances

DEFAULT_EXTRACTION_FILE_SUFFIX = ".extraction.tsv"
EVENT_ABSOLUTE_TIME_COL_NAME = "ABSOLUTE_TIME"


def create_session_df(infile: str, session: sd.SessionData,
					  game_round_utt_factory: game_utterances.SessionGameRoundUtteranceSequenceFactory) -> pd.DataFrame:
	result = game_round_utt_factory(session)
	result[game_utterances.EventColumn.DYAD_ID.value] = os.path.normpath(infile)
	# events_metadata = session.read_events_metadata()
	# session_start_timestamp = events_metadata["START_TIME"]
	# session_start = dateutil.parser.parse(session_start_timestamp)
	# result[EVENT_ABSOLUTE_TIME_COL_NAME] = result["TIME"].transform(
	#	lambda offset_secs: __create_absolute_time(session_start, offset_secs))
	return result


def normalize_result_dyad_paths(cv_results: pd.DataFrame):
	cv_results[game_utterances.EventColumn.DYAD_ID.value] = cv_results[
		game_utterances.EventColumn.DYAD_ID.value].transform(os.path.normpath)
	common_results_session_path = os.path.commonpath(
		(path for path in cv_results[game_utterances.EventColumn.DYAD_ID.value].unique()))
	print("Common path for session results is \"{}\"; Removing from dyad IDs.".format(common_results_session_path),
		  file=sys.stderr)
	cv_results[game_utterances.EventColumn.DYAD_ID.value] = cv_results[
		game_utterances.EventColumn.DYAD_ID.value].transform(
		lambda dyad_id: remove_prefix(dyad_id, common_results_session_path))
	cv_results[game_utterances.EventColumn.DYAD_ID.value] = cv_results[
		game_utterances.EventColumn.DYAD_ID.value].transform(os.path.dirname)


def normalize_session_paths(all_session_data: pd.DataFrame):
	common_session_path = os.path.commonpath(
		(path for path in all_session_data[game_utterances.EventColumn.DYAD_ID.value].unique()))
	print("Common session path is \"{}\"; Removing from dyad IDs.".format(common_session_path), file=sys.stderr)
	all_session_data[game_utterances.EventColumn.DYAD_ID.value] = all_session_data[
		game_utterances.EventColumn.DYAD_ID.value].transform(
		lambda dyad_id: remove_prefix(dyad_id, common_session_path))


def remove_prefix(text: str, prefix: str) -> str:
	# https://stackoverflow.com/a/16891418/1391325
	return text[len(prefix):] if text.startswith(prefix) else text


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

	all_session_data = pd.concat(
		(create_session_df(session_dir, session_data, game_round_utt_factory) for session_dir, session_data in
		 infile_session_data), copy=False)
	# noinspection PyUnresolvedReferences
	print("Dataframe shape: {}".format(all_session_data.shape), file=sys.stderr)
	# noinspection PyTypeChecker
	normalize_session_paths(all_session_data)
	# print(all_session_data[game_utterances.EventColumn.DYAD_ID.value])

	results_file_inpath = args.results_file
	print("Processing \"{}\".".format(results_file_inpath), file=sys.stderr)
	cv_results = cross_validation.read_results_file(results_file_inpath)
	normalize_result_dyad_paths(cv_results)
	print(cv_results[game_utterances.EventColumn.DYAD_ID.value])


# TODO: Finish
# print(cv_results.dtypes)
# event_results = pd.merge(cv_results, all_session_data, how="left", left_on=["EVENT_TIME"], right_on=[EVENT_ABSOLUTE_TIME_COL_NAME], copy=False)
# print(event_results.columns)
# print(event_results)



if __name__ == "__main__":
	__main(__create_argparser().parse_args())
