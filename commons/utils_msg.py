#!/usr/bin/env python3

from datetime import datetime

# ==============================
# カラー装飾用の定数
# ==============================
class Colors:
    RESET = '\033[0m' # Reset color
    BOLD = '\033[1m' # Bold text
    RED = '\033[91m' # Red text
    GREEN = '\033[92m' # Green text
    YELLOW = '\033[93m' # Yellow text
    BLUE = '\033[94m' # Blue text
    MAGENTA = '\033[95m' # Magenta text
    CYAN = '\033[96m' # Cyan text
    GRAY = '\033[90m' # Gray text

def msg_info(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{Colors.BLUE}{Colors.BOLD}💡 [INFO]{Colors.RESET} {timestamp} {Colors.CYAN}{msg}{Colors.RESET}"

def msg_error(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{Colors.RED}{Colors.BOLD}❌ [ERROR]{Colors.RESET} {timestamp} {Colors.RED}{msg}{Colors.RESET}"

def msg_debug(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{Colors.YELLOW}🔍 [DEBUG]{Colors.RESET} {timestamp} {Colors.GRAY}{msg}{Colors.RESET}"

def msg_success(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{Colors.GREEN}{Colors.BOLD}✅ [SUCCESS]{Colors.RESET} {timestamp} {Colors.GREEN}{msg}{Colors.RESET}"

