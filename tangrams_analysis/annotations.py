"""
Functionalities for reading and writing Higgins Annotation Tool (HAT) XML annotation files <http://www.speech.kth.se/hat/>.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2016-2017 Todd Shore"
__license__ = "Apache License, Version 2.0"

HAT_DATA_NAMESPACE = "http://www.speech.kth.se/higgins/2005/annotation/"
HAT_DATA_NAMESPACE_NAME = "hat"
HAT_DATA_SCHEMA_LOCATION = "http://www.speech.kth.se/higgins/2005/annotation/annotation.xsd"
ANNOTATION_NAMESPACES = {HAT_DATA_NAMESPACE_NAME: HAT_DATA_NAMESPACE,
						 "xsi": "http://www.w3.org/2001/XMLSchema-instance"}
