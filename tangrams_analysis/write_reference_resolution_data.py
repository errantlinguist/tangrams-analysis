#!/usr/bin/env python3

import argparse
import sys
from typing import Callable

import pandas as pd

import game_utterances
import session_data as sd
import utterances


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Cross-validation of reference resolution for tangram sessions.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def __create_session_df(name: str, session: sd.SessionData,
						session_data_frame_factory: Callable[[sd.SessionData], pd.DataFrame]):
	result = session_data_frame_factory(session)
	result["DYAD"] = name
	return result


def __main(args):
	inpaths = args.inpaths
	print("Looking for session data underneath {}.".format(inpaths), file=sys.stderr)
	infile_session_data = tuple(sorted(sd.walk_session_data(inpaths), key=lambda item: item[0]))

	session_data_frame_factory = game_utterances.SessionGameRoundUtteranceFactory(
		utterances.TokenSequenceFactory())
	session_df = pd.concat(
		__create_session_df(session_inpath, session_data, session_data_frame_factory) for (session_inpath, session_data)
		in
		infile_session_data)
	session_df.to_csv(sys.stdout, sep='\t', encoding="utf-8", index_label="Index")


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
