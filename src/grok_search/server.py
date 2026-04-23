"""Server entrypoint for the Grok Search FastMCP application."""

import os
import signal
import sys
import threading

from .app import mcp


def main() -> None:
    """
    Run the FastMCP server and ensure child shutdown on parent exit.

    Returns:
        None.
    """
    if threading.current_thread() is threading.main_thread():

        def handle_shutdown(signum, frame) -> None:
            """
            Exit immediately when a shutdown signal is received.

            Args:
                signum: The received signal number.
                frame: The current stack frame.

            Returns:
                None.
            """
            os._exit(0)

        signal.signal(signal.SIGINT, handle_shutdown)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, handle_shutdown)

    if sys.platform == "win32":
        import ctypes
        import time

        parent_pid = os.getppid()

        def is_parent_alive(pid: int) -> bool:
            """
            Check whether the parent process is still alive on Windows.

            Args:
                pid: The parent process identifier.

            Returns:
                True when the parent process is still running.
            """
            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return True
            exit_code = ctypes.c_ulong()
            result = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(result and exit_code.value == still_active)

        def monitor_parent() -> None:
            """
            Exit the process when the parent process disappears.

            Returns:
                None.
            """
            while True:
                if not is_parent_alive(parent_pid):
                    os._exit(0)
                time.sleep(2)

        threading.Thread(target=monitor_parent, daemon=True).start()

    try:
        mcp.run(transport="stdio", show_banner=False)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
