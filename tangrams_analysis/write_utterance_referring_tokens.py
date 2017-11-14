#!/usr/bin/env python3
"""
Use with e.g. find ~/Documents/Projects/Tangrams/Data/Derived/ -iname "*.tsv" -exec ./write_utterance_referring_tokens.py {} +
"""

import argparse
import csv
import sys
from typing import Dict

ENCODING = "utf-8"
INPUT_CSV_DIALECT = csv.excel_tab
OUTPUT_CSV_DIALECT = csv.excel_tab
REFERRING_LANGUAGE_COL_NAME = "REFERRING_TOKENS"
UTTERANCE_COL_NAME = "UTTERANCE"


def create_utterance_referring_token_map(*inpaths: str) -> Dict[str, str]:
	result = {}
	for inpath in inpaths:
		with open(inpath, "r", encoding=ENCODING) as inf:
			rows = csv.reader(inf, dialect=INPUT_CSV_DIALECT)
			header_col_idxs = dict((col, idx) for (idx, col) in enumerate(next(rows)))
			utt_col_idx = header_col_idxs[UTTERANCE_COL_NAME]
			ref_lang_col_idx = header_col_idxs[REFERRING_LANGUAGE_COL_NAME]

			for row in rows:
				utt = row[utt_col_idx]
				ref_lang = row[ref_lang_col_idx]
				old_ref_lang = result.get(utt)
				if old_ref_lang is None or old_ref_lang == ref_lang:
					result[utt] = ref_lang
				else:
					raise ValueError("Differing referring language for utterance \"{}\".".format(utt))
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
	utt_ref_lang = create_utterance_referring_token_map(*inpaths)
	writer = csv.writer(sys.stdout, dialect=OUTPUT_CSV_DIALECT)
	writer.writerows(sorted(utt_ref_lang.items(), key=lambda item: item[0]))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
