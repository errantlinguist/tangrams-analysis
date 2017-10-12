import re
from collections import defaultdict
from xml.sax.saxutils import escape

import lxml.builder
import lxml.etree as etree

import xml_files

HAT_DATA_NAMESPACE = "http://www.speech.kth.se/higgins/2005/annotation/"
HAT_DATA_NAMESPACE_NAME = "hat"
HAT_DATA_SCHEMA_LOCATION = "http://www.speech.kth.se/higgins/2005/annotation/annotation.xsd"
ANNOTATION_NAMESPACES = {HAT_DATA_NAMESPACE_NAME: HAT_DATA_NAMESPACE, "xsi": xml_files.XSI}

ELEMENT_MAKER = lxml.builder.ElementMaker(
	nsmap=ANNOTATION_NAMESPACES)

__DIGITS_PATTERN = re.compile('(\d+)')
__WHITESPACE_PATTERN = re.compile('\s+')


class AnnotationData(object):
	def __init__(self, qname_factory, element_maker: lxml.builder.ElementMaker, encoding: str):
		self.__qname_factory = qname_factory
		self.element_maker = element_maker
		self.encoding = encoding
		self.track_data = {}
		self.segment_data = SegmentData(qname_factory)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	def add(self, other):
		old_track_count = len(self.track_data)
		for track_id, other_track in other.track_data.items():
			if track_id in self.track_data:
				raise ValueError("Track ID \"%s\" already in dict." % track_id)
			else:
				self.track_data[track_id] = other_track
		# Remove channel IDs because HAT doesn't support multiple tracks for one Annotation which each have "channel" attrs
		if old_track_count < len(self.track_data):
			for track_datum in self.track_data.values():
				track_datum.remove_attr("channel")

		self.segment_data.add(other.segment_data)

	def create_xml_element(self):
		# http://stackoverflow.com/a/22902367/1391325
		result = self.element_maker("annotation")
		tracks_elem = etree.SubElement(result, self.__qname_factory("tracks"))
		for track_id, track_datum in sorted(self.track_data.items(), key=lambda k: natural_keys(k[0])):
			track_elem = etree.SubElement(tracks_elem, self.__qname_factory("track"), attrib={"id": track_id})
			sources_elem = etree.SubElement(track_elem, self.__qname_factory("sources"))
			for track_source in sorted(track_datum.sources_by_id.values(),
									   key=lambda elem: natural_keys(elem.get("id"))):
				sources_elem.append(track_source)

		segments_elem = self.segment_data.create_xml_element()
		result.append(segments_elem)

		return result


class AnnotationParser(object):
	def __init__(self, qname_factory, nsmap=None, id_prefix: str = ""):
		self.__qname_factory = qname_factory
		self.element_maker = ELEMENT_MAKER if nsmap is None else lxml.builder.ElementMaker(nsmap)
		self.id_prefix = id_prefix
		self.__tag_parsers = {self.__qname_factory("tracks"): self.__parse_tracks,
							  self.__qname_factory("segments"): self.__parse_segments}

	def __call__(self, doc_tree):
		result = AnnotationData(self.__qname_factory, self.element_maker, doc_tree.docinfo.encoding)
		tag_name = self.__qname_factory("annotation")
		for child in doc_tree.iter(tag_name):
			self.__parse_annotation(child, result)

		return result

	def __parse_annotation(self, annot, result):
		for child in annot:
			tag_name = child.tag
			parser = self.__tag_parsers[tag_name]
			parser(child, result)

	def __parse_segments(self, segments, result):
		segment_data = result.segment_data
		for segment in segments:
			attrs = segment.attrib
			segment_id = self.id_prefix + attrs["id"]
			attrs["id"] = segment_id
			segment_data.segments_by_id[segment_id] = segment
			track_id = self.id_prefix + attrs["track"]
			segment_data.track_segments[track_id].append(segment)
			attrs["track"] = track_id
			source_id = self.id_prefix + attrs["source"]
			segment_data.source_segments[source_id].append(segment)
			attrs["source"] = source_id

	def __parse_tracks(self, tracks, result):
		source_tag_name = self.__qname_factory("source")
		track_data = result.track_data
		for track in track_data:
			track_source_data = TrackDatum()
			track_id = self.id_prefix + track.get("id")
			track.set("id", track_id)
			track_data[track_id] = track_source_data

			for source in track.iter(source_tag_name):
				source_id = self.id_prefix + source.get("id")
				source.set("id", source_id)
				track_source_data.add(source)


class QNameStringFactory(object):
	def __init__(self, namespace):
		self.tag_qnames = {}
		self.__tag_prefix = "{" + namespace + "}"

	def __call__(self, tag_name: str):
		result = self.tag_qnames.get(tag_name)
		if not result:
			result = self.__tag_prefix + tag_name
			self.tag_qnames[tag_name] = result
		return result


class SegmentData(object):
	def __init__(self, qname_factory):
		self.__qname_factory = qname_factory
		self.segments_by_id = {}
		self.track_segments = defaultdict(list)
		self.source_segments = defaultdict(list)

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	def add(self, other):
		for segment_id, segment in other.segments_by_id.items():
			if segment_id in self.segments_by_id:
				raise ValueError("Segment ID \"%s\" already in dict." % segment_id)
			else:
				self.segments_by_id[segment_id] = segment
				self.track_segments.update(other.track_segments)
				self.source_segments.update(other.source_segments)

	def create_xml_element(self):
		result = etree.Element(self.__qname_factory("segments"))
		for segment in sorted(self.segments_by_id.values(), key=lambda elem: natural_keys(elem.get("id"))):
			result.append(segment)

		return result


class TrackDatum(object):
	def __init__(self):
		self.sources_by_id = {}
		self.sources_by_channel = {}
		self.sources_by_href = {}

	def __repr__(self):
		return self.__class__.__name__ + str(self.__dict__)

	def add(self, source):
		source_id = source.get("id")
		self.sources_by_id[source_id] = source
		channel = source.get("channel")
		if not is_blank_or_none(channel):
			self.sources_by_channel[channel] = source
		href = source.get("href")
		if not is_blank_or_none(href):
			self.sources_by_href[href] = source

	def remove(self, source):
		source_id = source.get("id")
		del self.sources_by_id[source_id]
		channel = source.get("channel")
		if not is_blank_or_none(channel):
			del self.sources_by_channel[channel]
		href = source.get("href")
		if not is_blank_or_none(href):
			del self.sources_by_href[href]

	def remove_attr(self, attr):
		if attr == "id":
			raise ValueError("Cannot remove ID attribute: This is absolutely necessary.")
		else:
			for source in self.sources_by_id.values():
				# source.set(attr, None)
				try:
					del source.attrib["channel"]
				except KeyError:
					# Attr not present; just continue
					pass


def create_skeleton_annnotation_elem(audio_file_path: str):
	"""
	FIXME: Buggy!!!
	:param audio_file_path: The audio file to create an empty annotation file for.
	:return: A new XML element.
	"""
	root = ELEMENT_MAKER.Element("hat:annotation", ANNOTATION_NAMESPACES)
	xml_files.add_xml_schema_location(root, HAT_DATA_SCHEMA_LOCATION)
	tracks = ELEMENT_MAKER.SubElement(root, "hat:tracks")
	__create_track_source_elem(tracks, "source1", "0", audio_file_path)
	__create_track_source_elem(tracks, "source2", "1", audio_file_path)
	ELEMENT_MAKER.SubElement(root, "hat:segments")
	return root


def is_blank_or_none(string: str):
	return string is None or len(string) < 1 or string.isspace()


def natural_keys(text):
	"""
	alist.sort(key=natural_keys) sorts in human order
	
	:see: http://nedbatchelder.com/blog/200712/human_sorting.html
	:see: http://stackoverflow.com/a/5967539/1391325
	"""
	return [__atoi(c) for c in __DIGITS_PATTERN.split(text)]


def sanitize_dom_id(string: str):
	result = __WHITESPACE_PATTERN.sub('-', string)
	return escape(result)


def __atoi(text: str):
	"""
	:see: http://stackoverflow.com/a/5967539/1391325
	"""
	return int(text) if text.isdigit() else text


def __create_track_source_elem(tracks_elem, source_id: str, channel: str, audio_file_path: str):
	track_sources = ELEMENT_MAKER.SubElement(tracks_elem, "hat:sources")
	result = ELEMENT_MAKER.SubElement(track_sources, "hat:source")
	result.set("id", source_id)
	result.set("channel", channel)
	result.set("href", audio_file_path)
	return result
