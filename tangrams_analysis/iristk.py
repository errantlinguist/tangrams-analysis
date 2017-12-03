"""
Utilities for processing files used and created by IrisTK <http://www.iristk.net/>.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright 2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

import datetime

LOGFILE_ENCODING = "utf-8"


def parse_timestamp(date_string: str) -> datetime.datetime:
	"""""
	Parses IrisTK-style timestamp strings, i.e. those matching "yyyy-[m]m-[d]d hh:mm:ss[.f...]" with an optional milliseconds part (the ".%f" at the end).
	
	See "iristk.system.LoggingModule.logSystemEvent(Event)" in the IrisTK source, which sets the "time" attribute of an event to be broadcast as an instance of "java.sql.Timestamp", the "Object.toString()" method of which produces the aforementioned string format.
	:param date_string: The timestamp string to parse.
	:return: A new timestamp object for the given string.
	:rtype: datetime.datetime
	"""""
	try:
		# yyyy-[m]m-[d]d hh:mm:ss[.f...]
		result = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
	except ValueError:
		result = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
	return result
