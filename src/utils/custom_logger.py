import sys
try:
    # Pycharm IDE specific implementation detail
    # This is only needed in order to detect if we launched into debug mode from the Pycharm IDE
    import pydevd

    DEBUGGING = True
except ImportError:
    # Not running from inside the Pycharm IDE, so let's check if the code was called
    # from a terminal with the `-d` system flag
    DEBUGGING = sys.flags.debug == 1  # note that sys.flags is a namedtuple object

from typing import Union, Any
import logging
from logging import Formatter

# the following "table" of _data was taken from the official python docs for
# python version 3.6.8 on May 5, 2019
# found here: https://docs.python.org/3.6/library/logging.html
# see that link for the full description of the available attributes
#
# Attribute name	    Format
# args	                You shouldn’t need to format this yourself.
# asctime	            %(asctime)s
# created	            %(created)f
# exc_info	            You shouldn’t need to format this yourself.
# filename	            %(filename)s
# funcName	            %(funcName)s
# levelname	            %(levelname)s
# levelno	            %(levelno)s
# lineno	            %(lineno)d
# message	            %(message)s
# module	            %(module)s
# msecs	                %(msecs)d
# msg	                You shouldn’t need to format this yourself.
# name	                %(name)s
# pathname	            %(pathname)s
# process	            %(process)d
# processName	        %(processName)s
# relativeCreated	    %(relativeCreated)d
# thread	            %(thread)d
# threadName	        %(threadName)s

# Level	    Numeric value
# CRITICAL  50
# ERROR     40
# WARNING   30
# INFO      20
# DEBUG     10
# NOTSET    0
# if DEBUGGING:
#     logging.basicConfig(level=logging.DEBUG)

LOGGER_COLLECTION = {}
#
# class LoggerExtraDefined(logging.Logger):
#     """
#     a subclass of logging.Logger which allows the definition of the extra dict at instantiation time
#     so that we may define contextual information for our log records that may be most easily known
#     at the time of the logger's instantiation.
#
#     This subclass will still defer to invocations of the extra keyword argument on calls to log, but
#     if a call to log leaves extra as None, we substitute in the reference given at instantiation.
#     """
#
#     def __init__(self, name, level=logging.NOTSET, extra: dict = None):
#         super().__init__(name, level)
#         self.extra = extra if extra is not None else {"extra": ""}
#
#     def assign_exta_dict(self, extra: dict,do_update:bool=True):
#         extra = extra if extra is not None else {}
#         if do_update:
#             self.extra.update(**extra)
#         else:
#             self.extra = extra
#
#     def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
#         extra = extra if extra is not None else self.extra
#         super(LoggerExtraDefined, self)._log(level, msg, args, exc_info, extra)


# logger formatting
# logging.setLoggerClass(LoggerExtraDefined)


# regarding *_PRI_LOGGER_HEADER_COLOR_PREFIX, defined bellow:
#   HIGH, and LOW priority header color prefix templates vary only in that HIGH sets 7 as a flag.
#   By including 7 as a formatting flag (the order of flags does not matter) the text will trade
#   foreground and background colors.
#   E.G. for a critical message header, supposing the background
#   is set to black by default, we will see black text on a bright-red background.
#
#   As added emphasis, the order of flags is irrelevant. "1;4;7;Xm"<=>"7;1;4;Xm" or any other
#   possible permutation of the ordering.
#
#   For further reference and details, please see the following wikipedia link:
#       https://en.wikipedia.org/wiki/ANSI_escape_code
ESC_SEQ = "\033["
HIGH_PRI_LOGGER_HEADER_COLOR_PREFIX = ESC_SEQ + "1;4;7;{}m"
LOW_PRI_LOGGER_HEADER_COLOR_PREFIX = ESC_SEQ + "1;4;{}m"
LOGGER_MSG_COLOR = ESC_SEQ + "1;{}m"
LOGGER_MSG_COLOR_INVERTED = ESC_SEQ + "1;7;{}m"
NOTSET_ID = 90  # charcoal
DBG_ID = 35  # purple
INFO_ID = 32  # green -- might be just my eyes, but green(32) and yellow(33) look almost identical
WARN_ID = 36  # dull-cyan
ERR_ID = 95  # bright-purple
CRIT_ID = 91  # bright-red
RESET = ESC_SEQ + "0m"
color_formatting_dict = {
    "reset":RESET,
    "format_keys": {
        "esc_seq": ESC_SEQ,
        "hi_pri_prefix": HIGH_PRI_LOGGER_HEADER_COLOR_PREFIX,
        "lo_pri_prefix": LOW_PRI_LOGGER_HEADER_COLOR_PREFIX,
        "logger_msg_color":LOGGER_MSG_COLOR,
        "logger_msg_color_inverted":LOGGER_MSG_COLOR_INVERTED,
    },
    "color_codes":{
        "not_set":NOTSET_ID,
        "debug":DBG_ID,
        "info":INFO_ID,
        "warning":WARN_ID,
        "error":ERR_ID,
        "critical":CRIT_ID,
        "white_dull":30,
        "red_dull":31,
        "green_dull":32,
        "yellow_dull":33,
        "purple_dull":35,
        "blue_dull":34,
        "cyan_dull":36,
        "gray_dull":37,
        "gray_bright":90,
        "red_bright":91,
        "green_bright":92,
        "yellow_bright":93,
        "blue_bright":94,
        "purple_bright":95,
        "cyan_bright":96,
        "white_bright":97,
    }
}
log_id_map = {logging.ERROR: ERR_ID, logging.INFO: INFO_ID, logging.WARNING: WARN_ID,
              logging.DEBUG: DBG_ID, logging.NOTSET: NOTSET_ID,logging.CRITICAL:CRIT_ID}
LEVELS_STR2INT = {
    "NOTSET": logging.NOTSET,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL}


def _build_default_formatter(level:int,child_name:str,colr_id:int)->logging.Formatter:
    if level < logging.WARNING:  # logging.WARNING == 30
        formatter = Formatter(
            "{header}%(asctime)s - {} - %(module)s.%(funcName)s(...) - %(lineno)d:{reset}{msg_style}"
            "\n\t%(message)s{reset}\n".format(
                child_name.capitalize(),
                header=LOW_PRI_LOGGER_HEADER_COLOR_PREFIX.format(colr_id),
                msg_style=LOGGER_MSG_COLOR.format(colr_id),
                reset=RESET),
            "%H:%M:%S")
    else:
        formatter = Formatter(
            "{header}%(asctime)s - {} - %(module)s.%(funcName)s(...) - %(lineno)d:{reset}{msg_style}"
            " ::> %(message)s{reset}".format(
                child_name.capitalize(),
                header=LOW_PRI_LOGGER_HEADER_COLOR_PREFIX.format(colr_id),
                msg_style=LOGGER_MSG_COLOR.format(colr_id),
                reset=RESET),
            "%H:%M:%S")
    return formatter


def _configure_handler(formatter:logging.Formatter, level:int,handler_delegate,handler_kwargs:dict)->logging.Handler:
    handler_kwargs.setdefault("stream", sys.stdout)
    handler = handler_delegate(**handler_kwargs)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _logger_setup(logger: logging.Logger,
                  header_label: str,
                  formatter: Formatter = None,
                  level: int = logging.INFO,
                  level_normalized: int = logging.INFO,
                  color:int=None,
                  handler_delegate=logging.StreamHandler,
                  **handler_kwargs) ->logging.Logger:
    """A convenience function for creating well named loggers with optional custom formatter.

    This function will implement an already defined formatter for you if the formatter param is
    None. For an example of a good general use logger see the example code at the bottom of this
    file.

    SPECIAL NOTE:
        Although this function's signature allows the caller to pass virtually any logging.Handler
        subclass into the handler_delegate parameter, I've only tested this functionality against
        the logging.StreamHandler class. If you wish to use it for others you will likely encounter
        bugs. But if you are up to the task it seems like it could potentially be a helpful tool for
        allowing the instantiation of a wide array of utility loggers under a single interface.

    :param header_label:
    :param logger: an initialized logger object that needs to have its formatter and optinoal handlers set.

    :param child_name:  The auxiliary logger you wish to create to handle some specific task. The
                        logger which maps to this child will set its level, handlers, and formatters
                        according to your inputs, allowing you to specify many different loggers to
                        manage _data output in a form that suites you.
    :type child_name:   A string

    :param formatter:   An optional parameter that specifies the manner in which you wish to format
                        the available logging-event-_data as well as how you wish to present the
                        message _data for your log events. The default formatter will take one of
                        two styles, based the given `level`.

                        For level==logging.INFO (20) and bellow:
                            log messages will be presented in two parts.
                                * First, a header line that's formatted to be bold, and underlined,
                                  that gives the time-stamp for when the log was submitted, the
                                  child_name, the model and function and line number from which the
                                  log-message was originated
                                * Followed by an indented new-line where the log-message will be
                                  be printed. The message
                        For level==logging.INFO+1 (21) and above:
                            log messages will be presented in two parts.
                                * First, a header line that's formatted to be bold, and underlined,
                                  that gives the time-stamp for when the log was submitted, the
                                  child_name, the model and function and line number from which the
                                  log-message was originated
                                * Followed, on the same line as the header, the log-message. This
                                  distinction, as opposed to the indented new-line in lower level
                                  messaged, is done because it is often the case the when higher
                                  level messages occur, there are very many of them. Forcing each
                                  one to then be a multi-line message actually makes it much harder
                                  to visually parse.
                                * Special note: In order to aid in automated parsing of these
                                  log-messages, the header details and log message will be seperated
                                  by the following character key:
                                        `::>`

    :type formatter:    an instance of logging.Formatter

    :param handler_delegate: An optional parameter that Specifies the type of handler you want to
                             associate to the logger instance that's mapped to the
                             root_name.child_name you've passed in. The handler will be set up
                             inside of this function, this parameter simply allows you to indicate
                             the way you wish to have your output handled.
                             (E.G. to std_out, std_err, or some file output stream)
    :type handler_delegate:  This should be a delegate function of your desired handler's
                             constructor, DEFAULT=logging.StreamHandler

    :param level:   Specifies the desired logging level of the resulting logger.
                    DEFAULT=logging.DEBUG
    :type level:    An int, must be in the range of [0,0xffffffff]

    :type handler_kwargs: Any additional keywords that should be passed into the construction of the
                          handler. These are necessary for the instantiation of handlers that will
                          output to anything other than sys.std_out, and sys.std_err.

    :return:    a reference to the logger instance that's mapped to the input naming scheme of
                <root_name>.<child_name >
    :rtype: logging.Logger
    """
    logger.propagate = False
    level_normalized = level if level_normalized is None else level_normalized
    try:
        colr_id = log_id_map[level_normalized]
    except KeyError:
        # level_normalized isn't in our custom log_id_map
        if level_normalized in logging._levelToName:
            # but it has been registered with the logging library
            LEVELS_STR2INT[logging._levelToName[level_normalized]] = level_normalized
        for lvl,color_code in sorted(log_id_map.items(),key=lambda tpl:tpl[0]):
            if lvl<=level_normalized:
                colr_id = color_code
            else:
                break
        else:
            colr_id = log_id_map["NOTSET"]
        log_id_map[level_normalized] = colr_id
    if formatter is None:
        formatter = _build_default_formatter(level,header_label,colr_id)
    logger.addHandler(_configure_handler(formatter,level,handler_delegate,handler_kwargs))
    return logger


def get_logger(root_name: str,
               child_name: str = None,
               header_label: str = None,
               formatter: Formatter = None,
               level: Union[int, str] = None,
               color:int=None,
               logger_registration_dict: dict = LOGGER_COLLECTION,
               handler_delegate=logging.StreamHandler, **handler_kwargs)->logging.Logger:
    level = level if level is not None else (logging.DEBUG if DEBUGGING else logging.NOTSET)
    if isinstance(level, str):
        level = LEVELS_STR2INT.get(level.upper(), logging.NOTSET)
    if not (logging.NOTSET<=level<=logging.CRITICAL):
        _logger_setup_internal_warning_logger(f"NOTICE: The level passed to the logger_setup function was not in the range of [{logging.NOTSET},{logging.CRITICAL}] and is going to be clipped to those bounds")
    level_normalized = max(level, logging.NOTSET)
    while level_normalized>logging.NOTSET:
        name = logging._levelToName.get(level_normalized,None)
        if name is not None:
            break
        level_normalized -= 1
    child_name = child_name if child_name else logging.getLevelName(level_normalized)
    header_label = header_label or child_name
    logger = logging.getLogger(root_name).getChild(child_name)
    _logger_setup_kwargs = dict(logger=logger, header_label=header_label, formatter=formatter, level=level,
                                level_normalized=level_normalized,color=color, handler_delegate=handler_delegate, **handler_kwargs)
    if logger.name not in logger_registration_dict:
        logger_registration_dict[logger.name] = _logger_setup(**_logger_setup_kwargs)
    return logger_registration_dict[logger.name]


class LogManager:
    def __init__(self) -> None:
        self._extra_dicts = {}
        self._registered_loggers = LOGGER_COLLECTION

    @property
    def registered_loggers(self):
        return self._registered_loggers

    @registered_loggers.setter
    def registered_loggers(self, value):
        if isinstance(value,dict):
            self._registered_loggers.update(**value)

    def forget_logger(self,logger_name:str):
        return self._registered_loggers.pop(logger_name,None)

    @property
    def extra_dicts(self):
        return self._extra_dicts

    def __getattribute__(self, item):
        if "logger" in item:
            registered_loggers = object.__getattribute__(self,"_registered_loggers")
            name = [key for key in registered_loggers if item in key]
            if name:
                logger:logging.Logger
                kwargs:dict
                logger,kwargs = registered_loggers.get(name[0], (None,None))
                if not logger.handlers:
                    logger = _logger_setup(logger, **kwargs)
                if logger.level>=logging.CRITICAL:
                    fmtr = logger.handlers[0].formatter
                    print(fmtr,fmtr._fmt,sep="\n")
                return logger
        ret = object.__getattribute__(self,item)
        return ret

    def add_logger_as_field(self, **name_and_getLogger_kwargs):
        """Creates new fields in this class instance that takes key:value pairs where key is the desired name for the
        field, and value is a dict of keyword args that will be passed into the `get_logger` function defined in this
        file.
        E.G:
            # here's a simple working example
            log_mgr = LogManager()
            log_mgr.add_logger_as_field(info_logger=dict(root_name=__name__, child_name="diagnostic info",level="INFO"))
            log_mgr.info_logger.info("This message is handled by the resulting logger,
                                      which you access like a property of the log_mgr instance")


        :param name_and_getLogger_kwargs: A dict of key:value pairs, where each key is the desired name for the field,
                                          and the value is a dictionary of kwargs that will be passed through to the
                                          get_logger function. The resulting logger instance is then mapped to the
                                          given field name in the LogManager instance.

        :return: None
        """
        bad_values = []
        for field,value in name_and_getLogger_kwargs.items():
            if isinstance(value,dict):
                given_child_name = value.pop("child_name",None)
                if given_child_name is None:
                    given_child_name = field
                value["child_name"] = field
                value.setdefault("header_label",given_child_name)
                logger = get_logger(logger_registration_dict=self._registered_loggers, **value)
                setattr(self,field,logger)
                value["level_normalized"] = None
                value["level"] = LEVELS_STR2INT.get(value["level"],logging.NOTSET)
                value.setdefault("header_label",".".join(logger.name.split(".")[1:]))
                del value["root_name"],value["child_name"]
                self._registered_loggers[logger.name] = self._registered_loggers[logger.name],value
            else:
                bad_values.append((field,value))
        for f,v in bad_values:
            _logger_setup_internal_warning_logger(f"LogManager.add_or_build_logger method was passed an invalid arg."
                                                  f"\n\t{f}:{v}")

    def add_extra_dict_as_field(self,**extra_name_and_dict):
        for field,value in extra_name_and_dict.items():
            self._extra_dicts[field] = value
            setattr(self,field,value)



_logger_setup_internal_warning_logger = get_logger(__name__, "logger_setup_warning", level="WARNING")
LOGGER_COLLECTION.pop(_logger_setup_internal_warning_logger.name,None)
_logger_setup_internal_warning_logger = _logger_setup_internal_warning_logger.warning