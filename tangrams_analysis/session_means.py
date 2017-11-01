#!/usr/bin/env python3

import argparse
import csv
import sys

import cross_validation


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Rank dialogue utterances by the mean rank of the dialogue classification.")
	result.add_argument("inpath", metavar="INPATH",
						help="The results file to process.")
	return result


def __main(args):
	inpath = args.inpath
	print("Processing results file \"{}\".".format(inpath), file=sys.stderr)
	cv_results = cross_validation.read_results_file(inpath)
	cv_results.drop(["Cleaning.DISFLUENCIES", "Cleaning.DUPLICATES", "Cleaning.FILLERS", "SESSION_ORDER", "TEST_ITER"],
					inplace=True, axis=1)
	dyads = cv_results.groupby("DYAD")
	dyad_means = dyads.agg("mean")
	dyad_means.to_csv(sys.stdout, sep=csv.excel_tab.delimiter, encoding="utf-8", index_label="DYAD")


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
