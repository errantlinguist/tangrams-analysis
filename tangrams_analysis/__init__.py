"""
ML and analysis tools for the game \"tangrams-restricted\" <https://github.com/errantlinguist/tangrams-restricted>.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import re
from typing import Tuple, Union

__DIGITS_PATTERN = re.compile('(\d+)')


def natural_keys(text: str) -> Tuple[Union[int, str], ...]:
	"""
	alist.sort(key=natural_keys) sorts in human order

	:see: http://nedbatchelder.com/blog/200712/human_sorting.html
	:see: http://stackoverflow.com/a/5967539/1391325
	"""
	return tuple(__atoi(c) for c in __DIGITS_PATTERN.split(text))


def __atoi(text: str) -> Union[int, str]:
	"""
	:see: http://stackoverflow.com/a/5967539/1391325
	"""
	return int(text) if text.isdigit() else text
