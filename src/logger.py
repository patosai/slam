LEVELS_MAP = {"DEBUG": 0,
              "INFO": 1,
              "WARN": 2,
              "ERROR": 3}
LOG_LEVEL = 0


def set_log_level(level):
    global LEVELS_MAP, LOG_LEVEL
    assert level in LEVELS_MAP
    LOG_LEVEL = LEVELS_MAP.get(level)


def debug(*args):
    global LEVELS_MAP, LOG_LEVEL
    if LOG_LEVEL <= LEVELS_MAP.get("DEBUG"):
        print(*args)


def info(*args):
    global LEVELS_MAP, LOG_LEVEL
    if LOG_LEVEL <= LEVELS_MAP.get("INFO"):
        print(*args)


def warn(*args):
    global LEVELS_MAP, LOG_LEVEL
    if LOG_LEVEL <= LEVELS_MAP.get("WARN"):
        print(*args)


def error(*args):
    global LEVELS_MAP, LOG_LEVEL
    if LOG_LEVEL <= LEVELS_MAP.get("ERROR"):
        print(*args)
