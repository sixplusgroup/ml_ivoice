import re
from plane import CJK
import logging
import sys

version = sys.version_info
above_36 = version.major >= 3 and version.minor > 6

logger = logging.getLogger(__name__)


def chinese_split(input):
  """Split Chinese input by:
  - Adding space between every Chinese character. Note: English word will remain as original

  Args:
      input (string): text to apply the regex func to
  """

  regex = re.compile("(?P<%s>%s)" % (CJK.name, CJK.pattern), CJK.flag)
  result = ""
  start = 0
  try:
    for t in regex.finditer(input):
      result += input[start: t.start()].strip()
      result += (
          " "
          + " ".join(
        [char for char in list(input[t.start(): t.end()]) if char != " "]
      )
          + " "
      )
      start = t.end()
    result += input[start:].strip()
  except TypeError as err:
    # mal row
    logger.warning(f"parsing data: {input} with error: {str(err)}")
  return result


def is_ascii(text):
  if above_36:
    return text.isascii()
  else:
    try:
      text.encode("ascii")
    except UnicodeEncodeError:
      return False
    else:
      return True
