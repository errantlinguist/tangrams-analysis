import os
import re

from typing import Dict, Iterable, Iterator, Union

ATTRIBUTE_VALUE_PAIR_DELIM_PATTERN = re.compile("(?<!\\\)[=:]")
COMMENT_LINE_PREFIXES = ('#', '!')
PROPERTIES_FILEPATH_PATTERN = re.compile(".+?\.properties$")

PropertyGroup = Union[str, "PropertyDict"]
PropertyDict = Dict[str, PropertyGroup]


def parse_properties(lines: Iterable[str]) -> PropertyDict:
	"""
	NOTE: This method cannot handle logical lines spread over multiple natural lines using escaped newline characters.
	See <https://docs.oracle.com/javase/8/docs/api/java/util/Properties.html#load-java.io.Reader->
	:param lines: The logical lines to parse as properties.
	:return: A dictionary of key-value pairs representing the (possibly nested) properties.
	:rtype: dict
	"""
	result = {}
	stripped_lines = (line.strip() for line in lines)
	semantic_lines = (line for line in stripped_lines if line and not line.startswith(COMMENT_LINE_PREFIXES))
	props = (ATTRIBUTE_VALUE_PAIR_DELIM_PATTERN.split(line, maxsplit=1) for line in semantic_lines)
	for prop in props:
		assert len(prop) == 2
		prop_name = prop[0]
		prop_value = prop[1]
		__set_prop_value(result, prop_name, prop_value)
	return result


def walk_properties_files(*inpaths: str) -> Iterator[str]:
	for inpath in inpaths:
		for dirpath, _, filenames in os.walk(inpath, followlinks=True):
			for filename in filenames:
				if PROPERTIES_FILEPATH_PATTERN.match(filename):
					resolved_path = os.path.join(dirpath, filename)
					yield resolved_path


def __fetch_prop_group(props: PropertyDict, prop_name: str) -> PropertyGroup:
	try:
		result = props[prop_name]
	except KeyError:
		result = {}
		props[prop_name] = result
	return result


def __set_nested_prop_value(props: PropertyDict, prop_name_components: Iterable[str], prop_value: str):
	terminal_prop_group = props
	name_component_iter = iter(prop_name_components)
	next_name_component = next(name_component_iter)
	reached_terminal_group = False
	while not reached_terminal_group:
		current_name_component = next_name_component
		try:
			next_name_component = next(name_component_iter)
			# The current component is not terminal; Get a nested group for it
			terminal_prop_group = __fetch_prop_group(terminal_prop_group, current_name_component)
		except StopIteration:
			# The current component is terminal; Insert the given value at this level
			terminal_prop_group[current_name_component] = prop_value
			reached_terminal_group = True


def __set_prop_value(props: PropertyDict, prop_name: str, prop_value: str):
	prop_name_components = prop_name.split('.')
	if len(prop_name_components) < 2:
		props[prop_name] = prop_value
	else:
		__set_nested_prop_value(props, prop_name_components, prop_value)
