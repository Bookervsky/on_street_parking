import logging
import os


def setup_log(log_dir="", log_level=logging.INFO):
    """
    Set up the basics of logging system. We have two logging handlers:
    one to the console and the other to a log file
    :param log_dir:
    :param log_level:
    :return:
    """

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 创建一个命名的日志记录器对象
    logger = logging.getLogger()
    # 设置日志级别为DEBUG
    logger.setLevel(log_level)

    # 创建一个文件处理器，并指定文件名和编码格式
    file_handler = logging.FileHandler(
        log_dir + "running_results.log", mode="w", encoding="utf-8"
    )
    warning_handler = logging.FileHandler(
        log_dir + "warning.log", mode="w", encoding="utf-8"
    )
    warning_handler.setLevel(logging.WARNING)
    # 创建一个控制台处理器，并指定输出流
    console_handler = logging.StreamHandler()

    # 创建一个日志格式器，并指定格式字符串
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s " "%(module)s - %(funcName)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    # 为文件处理器和控制台处理器设置格式器
    file_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 为日志记录器对象添加文件处理器和控制台处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(warning_handler)

    logging.getLogger("gurobipy").setLevel(logging.WARNING)

    return logger