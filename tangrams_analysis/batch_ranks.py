#!/usr/bin/env python3
"""
Use with e.g. "find ~/Documents/Projects/Tangrams/Data/output/ -iname "*.tsv" -exec ./batch_ranks.py {} +"
"""

import argparse
import csv
import sys

import pandas as pd


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Computes means of multiple cross-validation results files.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	return result


def __main(args):
	inpaths = args.inpaths
	print("Will read {} path(s).".format(len(inpaths)), file=sys.stderr)
	writer = csv.writer(sys.stdout, dialect=csv.excel_tab)
	writer.writerow(("FILE", "MEAN_RANK"))
	for inpath in sorted(inpaths):
		results = pd.read_csv(inpath, dialect=csv.excel_tab, sep=csv.excel_tab.delimiter)
		mean_rank = results["RANK"].mean()
		writer.writerow((inpath, mean_rank))

	pass


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
