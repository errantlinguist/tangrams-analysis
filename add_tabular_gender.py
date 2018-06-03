#!/usr/bin/env python3

"""
Adds participant gender from session participant metadata files to a given tabular data file for those sessions.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import os
import sys
from typing import Mapping

import pandas as pd

import tangrams_analysis.session_data

TABULAR_FILE_CSV_DIALECT = csv.excel_tab
TABULAR_FILE_DTYPES = {"session": "category", "Adt": "bool", "Wgt": "bool", "RndAdt": "bool"}
TABULAR_FILE_ENCODING = "utf-8"


def instructor_gender(session_participant_metadata: Mapping[str, Mapping[str, Mapping[str, str]]],
					  row: pd.Series) -> str:
	session = row["session"]
	participant_metadata = session_participant_metadata[session]
	gender_metadata = participant_metadata["GENDER"]
	instructor = row["Instructor"]
	return gender_metadata[instructor]


def other_gender(session_participant_metadata: Mapping[str, Mapping[str, Mapping[str, str]]],
				 row: pd.Series) -> str:
	session = row["session"]
	participant_metadata = session_participant_metadata[session]
	gender_metadata = participant_metadata["GENDER"]
	instructor = row["Instructor"]
	other_genders = tuple(
		gender for (participant_id, gender) in gender_metadata.items() if participant_id != instructor)
	if len(other_genders) > 1:
		raise ValueError("Dyads with more than two participants are (currently) not supported.")
	else:
		return other_genders[0]


def parse_dir_session_name(dirpath: str) -> str:
	return os.path.basename(dirpath)


def read_tabular_data(infile: str) -> pd.DataFrame:
	return pd.read_csv(infile, dialect=TABULAR_FILE_CSV_DIALECT, sep=TABULAR_FILE_CSV_DIALECT.delimiter,
					   dtype=TABULAR_FILE_DTYPES,
					   float_precision="round_trip",
					   encoding=TABULAR_FILE_ENCODING, memory_map=True)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Adds participant gender from session participant metadata files to a given tabular data file for those sessions.")
	result.add_argument("infile", metavar="INFILE", help="The tabular file to add to.")
	result.add_argument("session_dir", metavar="PATH", help="The directory under which the dyad files are to be found.")
	return result


def __main(args):
	infile = args.infile
	print("Reading tabular data from \"{}\".".format(infile), file=sys.stderr)
	df = read_tabular_data(infile)
	session_names = frozenset(df["session"].unique())
	print("Read results for {} sessions.".format(len(session_names)), file=sys.stderr)
	session_dir = args.session_dir
	print("Will look for sessions underneath \"{}\".".format(session_dir), file=sys.stderr)
	session_data = tuple((parse_dir_session_name(indir), sd) for (indir, sd) in
						 tangrams_analysis.session_data.walk_session_data((session_dir,)))
	missing_session_names = session_names.difference(frozenset(session_name for session_name, _ in session_data))
	if missing_session_names:
		raise ValueError("Missing sessions: {}".format(missing_session_names))
	else:
		df["Instructor"] = df["round"].transform(lambda game_round: "B" if game_round % 2 == 0 else "A")
		session_participant_metadata = dict(
			(session_name, sd.read_participant_metadata()) for (session_name, sd) in session_data)
		df["InstructorGender"] = df.apply(lambda row: instructor_gender(session_participant_metadata, row), axis=1)
		df["ManipulatorGender"] = df.apply(lambda row: other_gender(session_participant_metadata, row), axis=1)
		df.to_csv(sys.stdout, sep=TABULAR_FILE_CSV_DIALECT.delimiter, encoding=TABULAR_FILE_ENCODING, index=False)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
