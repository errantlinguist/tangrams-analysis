#!/usr/bin/env python3

"""
Creates a heatmap of vocabulary used in different sessions.

Use with e.g. "find ~/Documents/Projects/Tangrams/Data/Ready/ -iname "utts.tsv" -exec ./vocabulary_heatmap.py {} +"
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import sys

import utterances

import pandas as pd
import logging

from typing import Iterable


def __create_argparser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Writes all unique Higgins Annotation Tool (HAT) XML annotation segments to the standard output stream.")
    result.add_argument("inpaths", metavar="INPATH", nargs='+',
                        help="The files to process.")

    return result


def __read_utts_file(inpath: str, utt_reader: utterances.UtteranceTabularDataReader) -> pd.DataFrame:
    print("Reading utt file at \"{}\".".format(inpath), file=sys.stderr)
    return utt_reader(inpath)


def __main(args):
    inpaths = args.inpaths
    print("Will read {} file(s).".format(len(inpaths)), file=sys.stderr)
    utt_reader = utterances.UtteranceTabularDataReader()
    utts = pd.concat((__read_utts_file(inpath, utt_reader) for inpath in inpaths))
    print("Read {} unique utterance(s) from {} file(s).".format(utts.shape[0], len(inpaths)), file=sys.stderr)
    print(utts)


if __name__ == "__main__":
    __main(__create_argparser().parse_args())
