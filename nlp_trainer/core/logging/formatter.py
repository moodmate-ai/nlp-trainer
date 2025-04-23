import logging
import json


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        assert isinstance(record.msg, dict), (
            f"You need to log only dict typed object, not its {type(record.msg)}"
        )
        return json.dumps(record.msg)
