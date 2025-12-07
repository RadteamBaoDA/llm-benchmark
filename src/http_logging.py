import json
import logging
from typing import Any, Dict, Optional

from .config import LoggingConfig


class HttpLogger:
    """Isolated HTTP request/response logger used by the engine."""

    def __init__(self, logging_config: LoggingConfig):
        self.config = logging_config
        self.logger = logging.getLogger("src.http")
        self._configure()

    def _configure(self) -> None:
        log_level_str = self.config.level.upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        self.logger.setLevel(log_level)

        # Reset handlers so config changes take effect cleanly
        self.logger.handlers = []

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if log_level == logging.DEBUG or not self.config.log_file:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

        if self.config.log_requests or self.config.log_responses:
            self.logger.info(
                "HTTP logging enabled - Level: %s, Requests: %s, Responses: %s",
                log_level_str,
                self.config.log_requests,
                self.config.log_responses,
            )

    def _truncate(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length] + "... (truncated)"

    def log_request(self, request_id: int, url: str, headers: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if not self.config.log_requests:
            return

        safe_headers = headers.copy()
        if "Authorization" in safe_headers:
            safe_headers["Authorization"] = "Bearer ***"

        payload_str = json.dumps(payload, ensure_ascii=False)
        truncated_payload = self._truncate(payload_str, self.config.max_payload_length)

        self.logger.debug(
            "REQ #%s -> %s\n  Headers: %s\n  Payload: %s",
            request_id,
            url,
            safe_headers,
            truncated_payload,
        )

    def log_response(
        self,
        request_id: int,
        status_code: int,
        latency: float,
        response_body: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        if not self.config.log_responses:
            return

        if error:
            self.logger.error("REQ #%s <- ERROR (%.1fms): %s", request_id, latency * 1000, error)
            return

        body_preview = ""
        if response_body:
            truncated_body = self._truncate(response_body, self.config.max_response_length)
            body_preview = f"\n  Body: {truncated_body}"

        log_level = logging.DEBUG if status_code < 400 else logging.WARNING
        self.logger.log(
            log_level,
            "REQ #%s <- %s (%.1fms)%s",
            request_id,
            status_code,
            latency * 1000,
            body_preview,
        )
