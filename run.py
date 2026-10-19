"""DocScan entry point: launches the FastAPI server and opens the browser."""

import argparse
import logging
import socket
import sys
import threading
import time
import webbrowser
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("docscan.runner")


def is_port_in_use(host: str, port: int) -> bool:
    """Check if the specified host and port are already bound by another process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        result = sock.connect_ex((host, port))
        return result == 0


def _open_browser_delayed(url: str, delay_seconds: float = 1.0) -> None:
    """Wait briefly for Uvicorn to bind and start listening, then open web browser."""
    time.sleep(delay_seconds)
    logger.info(f"Opening browser at: {url}")
    webbrowser.open(url)


def main() -> None:
    """Parse CLI options, check port availability, and start DocScan."""
    parser = argparse.ArgumentParser(
        description="DocScan: CamScanner-style classical document scanner web app"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server TCP port (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser",
    )

    args = parser.parse_args()

    # Verify port availability
    if is_port_in_use(args.host, args.port):
        logger.error(
            f"Port {args.port} is already in use on {args.host}.\n"
            f"Please terminate the conflicting process or launch with a different port, e.g.:\n"
            f"    python run.py --port {args.port + 1}"
        )
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    logger.info(f"Starting DocScan server at {url} ...")

    if not args.no_browser:
        threading.Thread(
            target=_open_browser_delayed, args=(url, 0.8), daemon=True
        ).start()

    try:
        uvicorn.run(
            "docscan.server:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except Exception as exc:
        logger.error(f"Server encountered fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
