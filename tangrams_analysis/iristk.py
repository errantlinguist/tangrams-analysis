import datetime


def parse_timestamp(date_string: str) -> datetime.datetime:
	"""""
	Parses IrisTK-style timestamp strings, i.e. those matching "yyyy-[m]m-[d]d hh:mm:ss[.f...]" with an optional milliseconds part (the ".%f" at the end).
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
