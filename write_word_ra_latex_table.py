#!/usr/bin/env python3


"""
Reads in a vocabulary file from Gabriel containing referring ability (RA) scores and counts for each word and then writes it as a LaTeX tabular environment.

The first column is the word, second is the count and third is the RA.

"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import sys

COUNT_COL_IDX = 1
RA_COL_IDX = 2


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Writes a vocabulary RA table in LaTeX format.")
	result.add_argument("infile", metavar="FILE",
						help="The vocabulary file to read.")
	return result


def __main(args):
	infile = args.infile
	print("Reading \"{}\".".format(infile), file=sys.stderr)
	with open(infile, 'r') as inf:
		reader = csv.reader(inf, dialect=csv.excel_tab)
		rows = tuple(sorted(reader, key=lambda row: float(row[RA_COL_IDX]), reverse=True))

	print("\\begin{tabular}{| l r r |}")
	for row in rows:
		word_repr = "\\lingform{%s}" % row[0]
		ra_repr = "${0:.2f}$".format(float(row[RA_COL_IDX]))
		count_repr = "${}$".format(row[COUNT_COL_IDX])
		latex_row = (word_repr, ra_repr, count_repr)
		latex_line = "\t" + "\t&\t".join(latex_row) + " \\\\"
		print(latex_line)
	print("\\end{tabular}")


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
