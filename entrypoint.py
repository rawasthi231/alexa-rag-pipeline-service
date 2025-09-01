#!/usr/bin/env python3
import os
import sys


def _sanitize_port(value: str) -> str:
    # Allow only digits; default to 5500 if invalid
    if value and value.isdigit():
        # Basic range check for TCP ports
        iv = int(value)
        if 1 <= iv <= 65535:
            return value
    return "5500"


def main() -> None:
    port = _sanitize_port(os.getenv("PORT", "5500"))
    args = [
        "gunicorn",
        "--bind",
        f":{port}",
        "--workers",
        "4",
        "--worker-connections",
        "1000",
        "--timeout",
        "30",
        "--keep-alive",
        "5",
        "main:app",
    ]
    # Replace current process so signals are delivered correctly
    os.execvp(args[0], args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Entrypoint failed: {exc}", file=sys.stderr)
        sys.exit(1)
