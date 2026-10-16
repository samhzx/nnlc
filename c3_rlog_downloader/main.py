"""Standalone executable entry point."""

from __future__ import annotations

import argparse

from c3_rlog_downloader import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="独立 C3 rlog.zst 下载客户端")
    parser.add_argument("--version", action="store_true", help="显示版本后退出")
    args = parser.parse_args()
    if args.version:
        print(__version__)
        return
    from c3_rlog_downloader.gui import launch_gui

    launch_gui()


if __name__ == "__main__":
    main()
