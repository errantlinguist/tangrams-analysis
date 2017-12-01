#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys
from numbers import Number
from typing import Iterable, Iterator, Sequence, Tuple

from rake_nltk import Rake

import utterances

INFILE_DIALECT = csv.excel_tab
OUTPUT_DIALECT = csv.excel_tab
TOKEN_DELIMITER_PATTERN = re.compile("\\s+")

__SENTENCE_TOKEN_DELIMITER = " "
__DOCUMENT_SENTENCE_DELIMITER = " "


def capitalize_first_letter(token: str) -> str:
	strlen = len(token)
	if strlen < 1:
		result = token
	elif strlen == 1:
		result = token.capitalize()
	else:
		first = token[0].capitalize()
		result = first + token[1:]

	return result


def create_sentence(tokens: Sequence[str], sentence_delim: str = ".") -> str:
	seqlen = len(tokens)
	if seqlen < 1:
		result = ""
	elif seqlen == 1:
		result = capitalize_first_letter(tokens[0]) + sentence_delim
	else:
		formatted_tokens = []
		first = capitalize_first_letter(tokens[0])
		formatted_tokens.append(first)
		last_idx = seqlen - 1
		formatted_tokens.extend(tokens[1:last_idx])
		last = tokens[last_idx] + sentence_delim
		formatted_tokens.append(last)
		result = __SENTENCE_TOKEN_DELIMITER.join(formatted_tokens)
	return result


def walk_dirs_containing_file(inpaths: Iterable[str], filename: str) -> Iterator[str]:
	for inpath in inpaths:
		for dirpath, _, filenames in os.walk(inpath, followlinks=True):
			if filename in filenames:
				resolved_path = os.path.join(inpath, dirpath)
				yield resolved_path


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Rank dialogue utterances by the mean rank of the dialogue classification.")
	result.add_argument("inpaths", metavar="INPATH", nargs='+',
						help="The directories to process.")
	result.add_argument("-f", "--file", metavar="NAME", required=True,
						help="The name of the file in each session directory to read.")
	return result


def read_tokens(infile: str, col_name: str) -> Iterator[Sequence[str]]:
	with open(infile, 'r') as inf:
		reader = csv.DictReader(inf, dialect=INFILE_DIALECT)
		for row in reader:
			token_str = row[col_name]
			tokens = TOKEN_DELIMITER_PATTERN.split(token_str)
			if tokens:
				yield tokens


def read_session_keywords(ref_lang_filepath: str) -> Sequence[Tuple[Number, str]]:
	token_seqs = read_tokens(ref_lang_filepath, "TOKENS")
	nonmetalanguage_token_seqs = (
		tuple(token for token in token_seq if utterances.is_semantically_relevant_token(token)) for token_seq in
		token_seqs)
	sentences = (create_sentence(token_seq) for token_seq in nonmetalanguage_token_seqs if token_seq)
	session_doc = __DOCUMENT_SENTENCE_DELIMITER.join(sentences)
	r = Rake()  # Uses stopwords for english from NLTK, and all punctuation characters.
	r.extract_keywords_from_text(session_doc)
	return r.get_ranked_phrases_with_scores()


def __main(args):
	inpaths = args.inpaths
	filename = args.file
	print("Looking for directories containing \"{}\" underneath {}.".format(filename, inpaths), file=sys.stderr)
	session_dirs = tuple(sorted(walk_dirs_containing_file(inpaths, filename)))

	writer = csv.writer(sys.stdout, dialect=OUTPUT_DIALECT)
	writer.writerow(("SESSION", "PHRASE", "SCORE"))
	for session_dir in session_dirs:
		ref_lang_filepath = os.path.join(session_dir, filename)
		scored_phrases = read_session_keywords(ref_lang_filepath)
		for score, phrase in scored_phrases:
			row = (session_dir, phrase, score)
			writer.writerow(row)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
