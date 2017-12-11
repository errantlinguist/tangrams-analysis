#!/usr/bin/env python3

"""
Creates a heatmap of vocabulary used in different sessions.

Use with e.g. "find ~/Documents/Projects/Tangrams/Data/Ready/ -iname "*.tsv" -exec ./vocabulary_heatmap.py {} +"
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import sys

def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Writes all unique Higgins Annotation Tool (HAT) XML annotation segments to the standard output stream.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The files to process.")

	return result


def __main(args):
	inpaths = args.inpaths
	print("Will read {} file(s).".format(len(inpaths)), file=sys.stderr)
	utts = create_utterance_set(*inpaths)
	print("Read {} unique utterance(s) from {} file(s).".format(len(utts), len(inpaths)), file=sys.stderr)
	rows = sorted(utts)
	for row in rows:
		print(row, file=sys.stdout)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
