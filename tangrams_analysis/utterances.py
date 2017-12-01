import sys
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterable, Iterator, List, Sequence, Tuple
from xml.etree.ElementTree import Element, parse as parse_etree

import annotations

"""
NOTE: See "../src/main/resources/se/kth/speech/nlp/fillers.txt"
"""
FILLER_TOKENS = frozenset(("eh", "ehm", "em", "er", "err", "eugh," "uff", "uh", "uhm", "um", "umm",))
"""
NOTE: See "../src/main/resources/se/kth/speech/coin/tangrams/analysis/SegmentUtteranceFactory.properties"
"""
METALANGUAGE_TOKENS = frozenset(
	("ARTIFACT", "BREATH", "CLICK", "COUGH", "FOREIGN_WORD", "GASP", "GROAN", "GRUNT", "LAUGHTER", "META", "MOAN",
	 "NOISE", "PUFF", "SIGH", "SNIFF", "START_SIGNAL", "UNKNOWN",))
__TOKEN_TRUNCATION_MARKER = '-'


class SegmentUtteranceFactory(object):
	def __init__(self, token_seq_factory: Callable[[Iterable[str]], Sequence[str]],
				 source_speaker_id_factory: Callable[[str], str] = lambda source_id: source_id):
		"""

		:param token_seq_factory: A function for converting a sequence of bare token strings into normalized forms for use as utterance content.
		:param source_speaker_id_factory: A function for converting segment source IDs (e.g. "source234") to utterances speaker IDs (e.g. "superplayer" or "A" or "B")
		"""
		self.token_seq_factory = token_seq_factory
		"""A function for converting a sequence of bare token strings into normalized forms for use as utterance content."""
		self.source_speaker_id_factory = source_speaker_id_factory
		"""A function for converting segment source IDs (e.g. "source234") to utterances speaker IDs (e.g. "superplayer" or "A" or "B")."""

	def __call__(self, segments: Iterable[Element]) -> Iterator["Utterance"]:
		for segment in segments:
			utt = self.__create(segment)
			if utt:
				yield utt

	def __create(self, segment: Element) -> "Utterance":
		token_elems = segment.iterfind(".//hat:t", annotations.ANNOTATION_NAMESPACES)
		token_text = (elem.text for elem in token_elems)
		content = self.token_seq_factory(token_text)
		if content:
			speaker_id = self.source_speaker_id_factory(segment.get("source"))
			result = Utterance(segment.get("id"), speaker_id, float(segment.get("start")),
							   float(segment.get("end")), content)
		else:
			result = None

		return result


class Utterance(object):
	@staticmethod
	def between(utts: Iterable["Utterance"], start_time: float, end_time: float):
		return (utt for utt in utts if (utt.start_time >= start_time) and (utt.start_time < end_time))

	def __init__(self, segment_id: str, speaker_id: str, start_time: float, end_time: float, content: Sequence[str]):
		self.segment_id = segment_id
		self.speaker_id = speaker_id
		self.start_time = start_time
		self.end_time = end_time
		self.content = content

	@property
	def __key(self):
		return self.segment_id, self.speaker_id, self.start_time, self.end_time, self.content

	def __eq__(self, other):
		return (self is other or (isinstance(other, type(self))
								  and self.__key == other.__key))

	def __hash__(self):
		return hash(self.__key)

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)


def is_semantically_relevant_token(token: str) -> bool:
	return token not in METALANGUAGE_TOKENS and token not in FILLER_TOKENS and not is_disfluency(token)


class TokenSequenceFactory(object):
	def __init__(self, token_filter: Callable[[str], bool] = is_semantically_relevant_token):
		self.token_filter = token_filter
		self.token_seq_singletons = {}

	def __call__(self, tokens: Iterable[str]) -> Tuple[str]:
		content = tuple(stripped_token for stripped_token in (token.strip() for token in tokens) if
						stripped_token and self.token_filter(stripped_token))
		if content:
			try:
				result = self.token_seq_singletons[content]
			except KeyError:
				result = tuple(sys.intern(token) for token in content)
				self.token_seq_singletons[result] = result
		else:
			result = None

		return result


def create_speaker_dict(utts: Iterable[Utterance]) -> DefaultDict[str, List[Utterance]]:
	result = defaultdict(list)
	for utt in utts:
		result[utt.speaker_id].append(utt)
	return result


def dialogue_utt_str_repr(utts: Iterable[Utterance]) -> str:
	repr_list = []
	grouped_utts = group_utts_by_speaker_id(utts)
	for speaker_id, utt_group in grouped_utts:
		speaker_repr = __speaker_id_repr(speaker_id)
		repr_list.append(speaker_repr)
		sentence_repr = '"' + join_utt_sentence_reprs(utt_group) + '"'
		repr_list.append(sentence_repr)

	return ' '.join(repr_list)


def group_utts_by_speaker_id(utts: Iterable[Utterance]) -> List[Tuple[List[Utterance]]]:
	result = []

	current_speaker_id = None
	current_speaker_utts = []
	for utt in utts:
		utt_speaker_id = utt.speaker_id
		if utt_speaker_id == current_speaker_id:
			current_speaker_utts.append(utt)
		else:
			if current_speaker_utts:
				result.append((current_speaker_id, current_speaker_utts))
			current_speaker_id = utt_speaker_id
			current_speaker_utts = [utt]

	if current_speaker_utts:
		result.append((current_speaker_id, current_speaker_utts))

	return result


def is_disfluency(token: str) -> bool:
	return token.startswith(__TOKEN_TRUNCATION_MARKER) or token.endswith(__TOKEN_TRUNCATION_MARKER)


def join_utt_sentence_reprs(utts: Iterable[Utterance]) -> str:
	return ' '.join(token_seq_repr(utt.content) for utt in utts)


def read_segments(infile_path: str) -> Iterator[Element]:
	print("Reading XML file \"{}\".".format(infile_path), file=sys.stderr)
	doc_tree = parse_etree(infile_path)
	return doc_tree.iterfind(".//hat:segment", annotations.ANNOTATION_NAMESPACES)


def token_seq_repr(tokens: Iterable[str]) -> str:
	token_iter = iter(tokens)
	formatted_tokens = []

	next_token = __capitalize_first_char(next(token_iter))
	end_reached = False
	while not end_reached:
		current_token = next_token
		try:
			next_token = next(token_iter)
		except StopIteration:
			current_token = current_token + '.'
			end_reached = True
		formatted_tokens.append(current_token)

	return ' '.join(formatted_tokens)


def __capitalize_first_char(string: str) -> str:
	if len(string) < 2:
		return string.upper()
	else:
		first_char = string[0]
		return first_char.upper() + string[1:]


def __speaker_id_repr(speaker_id: Any) -> str:
	return "**{}:**".format(speaker_id)
