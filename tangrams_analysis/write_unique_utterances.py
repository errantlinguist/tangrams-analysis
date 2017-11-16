#!/usr/bin/env python3
"""
Use with e.g. "find ~/Documents/Projects/Tangrams/Data/Derived/ -iname "*.xml" -exec ./write_utterance_referring_tokens.py {} +"
"""

import argparse
import sys
import xml.etree.ElementTree
from typing import Set

HAT_DATA_NAMESPACE = "http://www.speech.kth.se/higgins/2005/annotation/"
HAT_DATA_SCHEMA_LOCATION = "http://www.speech.kth.se/higgins/2005/annotation/annotation.xsd"
ANNOTATION_NAMESPACES = {"hat": HAT_DATA_NAMESPACE,
						 "xsi": "http://www.w3.org/2001/XMLSchema-instance"}


def create_utterance_set(*inpaths: str) -> Set[str]:
	result = set()
	for inpath in inpaths:
		print("Reading XML file \"{}\".".format(inpath), file=sys.stderr)
		tree = xml.etree.ElementTree.parse(inpath)
		segments = tree.iterfind(".//hat:segment", ANNOTATION_NAMESPACES)
		for segment in segments:
			tokens = segment.iterfind(".//hat:t", ANNOTATION_NAMESPACES)
			utt_str = " ".join(token.text for token in tokens)
			result.add(utt_str)
	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Finds all tabular utterance referring language files and creates a single map of unique utterances to the canonical referring tokens in that utterance.")
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
