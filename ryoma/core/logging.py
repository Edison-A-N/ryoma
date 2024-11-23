import sys
from datetime import datetime
from typing import Any
import colorama
from colorama import Fore, Style

# 初始化colorama以支持Windows
colorama.init()


class PrettyLogger:
    LEVELS = {
        "DEBUG": (Fore.CYAN, "🔍"),
        "INFO": (Fore.GREEN, "ℹ️"),
        "WARNING": (Fore.YELLOW, "⚠️"),
        "ERROR": (Fore.RED, "❌"),
        "CRITICAL": (Fore.RED + Style.BRIGHT, "💀"),
    }

    def _log(self, level: str, message: Any, **kwargs) -> None:
        color, icon = self.LEVELS[level]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 格式化额外的参数
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        extras = f" | {extras}" if extras else ""

        # 构建日志消息
        log_message = (
            f"{Fore.BLUE}{timestamp}{Style.RESET_ALL} "
            f"{color}{icon} {level:8}{Style.RESET_ALL} | "
            f"{str(message)}{extras}"
        )

        print(log_message, file=sys.stderr)

    def debug(self, message: Any, **kwargs) -> None:
        self._log("DEBUG", message, **kwargs)

    def info(self, message: Any, **kwargs) -> None:
        self._log("INFO", message, **kwargs)

    def warning(self, message: Any, **kwargs) -> None:
        self._log("WARNING", message, **kwargs)

    def error(self, message: Any, **kwargs) -> None:
        self._log("ERROR", message, **kwargs)

    def critical(self, message: Any, **kwargs) -> None:
        self._log("CRITICAL", message, **kwargs)


# 创建全局logger实例
logger = PrettyLogger()
