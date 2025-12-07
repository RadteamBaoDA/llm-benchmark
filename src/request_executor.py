"""
HTTP request execution helpers extracted from BenchmarkEngine for clarity.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from .config import BenchmarkConfig
from .event_bus import EventBus
from .events import RequestCompleted, RequestFailed, RequestStarted
from .http_logging import HttpLogger
from .mock_data import MockRequest
from .modes import QueueMetrics


class RequestExecutor:
    def __init__(
        self,
        config: BenchmarkConfig,
        event_bus: EventBus,
        http_logger: HttpLogger,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.http_logger = http_logger

    # ---------------------------------------------------------------------
    # Request lifecycle helpers
    # ---------------------------------------------------------------------
    def _build_request_target(self, mock_request: MockRequest) -> Tuple[str, Dict[str, str]]:
        prefix = self.config.api.endpoint_prefix.rstrip("/")
        url = f"{self.config.api.base_url.rstrip('/')}{prefix}{mock_request.endpoint}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api.api_key:
            headers["Authorization"] = f"Bearer {self.config.api.api_key}"
        return url, headers

    async def _emit_request_started(
        self,
        req_id: int,
        mock_request: MockRequest,
        url: str,
        payload: dict,
        headers: Dict[str, str],
    ) -> None:
        self.http_logger.log_request(req_id, url, headers, payload)
        await self.event_bus.publish(
            RequestStarted(
                request_id=req_id,
                endpoint=mock_request.endpoint,
                payload=payload,
                url=url,
            )
        )

    async def _emit_request_failed(
        self,
        req_id: int,
        mock_request: MockRequest,
        payload: dict,
        latency: float,
        status_code: Optional[int],
        error_msg: str,
    ) -> None:
        await self.event_bus.publish(
            RequestFailed(
                request_id=req_id,
                endpoint=mock_request.endpoint,
                payload=payload,
                latency=latency,
                status_code=status_code,
                error=error_msg,
            )
        )

    def _extract_tokens(self, data: Dict[str, Any]) -> Tuple[int, int, int]:
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        return total_tokens, prompt_tokens, completion_tokens

    # ---------------------------------------------------------------------
    # Request execution
    # ---------------------------------------------------------------------
    async def make_request(
        self,
        client: httpx.AsyncClient,
        mock_request: MockRequest,
        capture_response: bool,
        timeout: Optional[int],
        request_id: Optional[int],
        queue_metrics: Optional[QueueMetrics],
        request_submit_times: Dict[int, float],
        request_start_times: Dict[int, float],
    ) -> Tuple[
        Optional[float],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[Dict[str, Any]],
        Optional[str],
        Optional[int],
    ]:
        url, headers = self._build_request_target(mock_request)
        request_timeout = timeout if timeout is not None else self.config.api.timeout
        req_id = request_id if request_id is not None else 0

        await self._emit_request_started(req_id, mock_request, url, mock_request.payload, headers)

        if request_id is not None and request_id in request_submit_times:
            request_start_times[request_id] = time.perf_counter()

        t0 = time.perf_counter()
        http_status = None

        try:
            response = await client.post(
                url,
                headers=headers,
                json=mock_request.payload,
                timeout=request_timeout,
            )
            latency = time.perf_counter() - t0
            http_status = response.status_code

            response_text = response.text
            self.http_logger.log_response(req_id, http_status, latency, response_text)
            response.raise_for_status()

            data = json.loads(response_text)
            tokens, prompt_tokens, completion_tokens = self._extract_tokens(data)

            await self.event_bus.publish(
                RequestCompleted(
                    request_id=req_id,
                    endpoint=mock_request.endpoint,
                    payload=mock_request.payload,
                    latency=latency,
                    status_code=http_status,
                    tokens=tokens or 0,
                    prompt_tokens=prompt_tokens or 0,
                    completion_tokens=completion_tokens or 0,
                    response=data if capture_response else None,
                )
            )

            return (
                latency,
                tokens,
                prompt_tokens,
                completion_tokens,
                data if capture_response else None,
                None,
                http_status,
            )

        except httpx.HTTPStatusError as e:
            latency = time.perf_counter() - t0
            http_status = e.response.status_code

            try:
                error_body = e.response.text
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", "") or error_json.get(
                    "detail", ""
                )
            except Exception:
                error_body = str(e)
                error_detail = ""

            error_msg = f"HTTP {http_status}"
            if error_detail:
                error_msg = f"HTTP {http_status}: {error_detail}"
            elif http_status == 429:
                error_msg = "HTTP 429: Rate Limited / Queue Full"
            elif http_status == 503:
                error_msg = "HTTP 503: Service Unavailable / Server Overloaded"

            self.http_logger.log_response(req_id, http_status, latency, error_body, error_msg)

            if http_status in (429, 503) and queue_metrics:
                queue_metrics.record_rejection()

            await self._emit_request_failed(
                req_id,
                mock_request,
                mock_request.payload,
                latency,
                http_status,
                error_msg,
            )

            return latency, None, None, None, None, error_msg, http_status

        except httpx.TimeoutException:
            latency = time.perf_counter() - t0
            error_msg = f"Timeout after {request_timeout}s"
            self.http_logger.log_response(req_id, 0, latency, error=error_msg)
            if queue_metrics:
                queue_metrics.record_timeout()
            await self._emit_request_failed(
                req_id,
                mock_request,
                mock_request.payload,
                latency,
                None,
                error_msg,
            )
            return latency, None, None, None, None, error_msg, None

        except httpx.ConnectError as e:
            latency = time.perf_counter() - t0
            error_msg = f"Connection error: {str(e)}"
            self.http_logger.log_response(req_id, 0, latency, error=error_msg)
            await self._emit_request_failed(
                req_id,
                mock_request,
                mock_request.payload,
                latency,
                None,
                error_msg,
            )
            return latency, None, None, None, None, error_msg, None

        except Exception as e:
            latency = time.perf_counter() - t0
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.http_logger.log_response(req_id, 0, latency, error=error_msg)
            await self._emit_request_failed(
                req_id,
                mock_request,
                mock_request.payload,
                latency,
                None,
                error_msg,
            )
            return latency, None, None, None, None, error_msg, None

    async def make_streaming_request(
        self,
        client: httpx.AsyncClient,
        mock_request: MockRequest,
        capture_response: bool,
        timeout: Optional[int],
        request_id: Optional[int],
        queue_metrics: Optional[QueueMetrics],
        request_submit_times: Dict[int, float],
        request_start_times: Dict[int, float],
    ) -> Tuple[
        Optional[float],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[Dict[str, Any]],
        Optional[str],
        Optional[int],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        url, headers = self._build_request_target(mock_request)
        request_timeout = timeout if timeout is not None else self.config.api.timeout
        req_id = request_id if request_id is not None else 0

        payload = mock_request.payload.copy()
        payload["stream"] = True

        await self._emit_request_started(req_id, mock_request, url, payload, headers)

        t0 = time.perf_counter()
        ttft = None
        token_times = []
        http_status = None
        chunks = []
        completion_tokens = 0

        try:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            ) as response:
                http_status = response.status_code
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        chunks.append(chunk)

                        current_time = time.perf_counter()
                        if ttft is None:
                            ttft = (current_time - t0) * 1000
                        token_times.append(current_time)

                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if delta.get("content"):
                                completion_tokens += 1
                    except json.JSONDecodeError:
                        continue

                latency = time.perf_counter() - t0

        except httpx.HTTPStatusError as e:
            latency = time.perf_counter() - t0
            http_status = e.response.status_code
            error_msg = f"HTTP {http_status}"
            self.http_logger.log_response(req_id, http_status, latency, error=error_msg)
            await self._emit_request_failed(
                req_id,
                mock_request,
                payload,
                latency,
                http_status,
                error_msg,
            )
            return latency, None, None, None, None, error_msg, http_status, None, None, None

        except httpx.TimeoutException:
            latency = time.perf_counter() - t0
            error_msg = f"Timeout after {request_timeout}s"
            self.http_logger.log_response(req_id, 0, latency, error=error_msg)
            await self._emit_request_failed(
                req_id,
                mock_request,
                payload,
                latency,
                None,
                error_msg,
            )
            return latency, None, None, None, None, error_msg, None, None, None, None

        except Exception as e:
            latency = time.perf_counter() - t0
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.http_logger.log_response(req_id, 0, latency, error=error_msg)
            await self._emit_request_failed(
                req_id,
                mock_request,
                payload,
                latency,
                None,
                error_msg,
            )
            return latency, None, None, None, None, error_msg, None, None, None, None

        latency_ms = latency * 1000
        tpot = None
        if ttft is not None and completion_tokens > 1:
            tpot = (latency_ms - ttft) / (completion_tokens - 1)

        itl_avg = None
        if len(token_times) > 1:
            itls = [(token_times[i] - token_times[i - 1]) * 1000 for i in range(1, len(token_times))]
            itl_avg = sum(itls) / len(itls)

        prompt_tokens = 0
        total_tokens = completion_tokens
        if chunks:
            last_chunk = chunks[-1]
            if "usage" in last_chunk:
                prompt_tokens = last_chunk["usage"].get("prompt_tokens", 0)
                total_tokens = last_chunk["usage"].get("total_tokens", completion_tokens)

        self.http_logger.log_response(req_id, http_status or 0, latency, f"Streaming: {completion_tokens} tokens")

        response_data = {"chunks": chunks} if capture_response else None

        await self.event_bus.publish(
            RequestCompleted(
                request_id=req_id,
                endpoint=mock_request.endpoint,
                payload=payload,
                latency=latency,
                status_code=http_status,
                tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response=response_data,
            )
        )

        return (
            latency,
            total_tokens,
            prompt_tokens,
            completion_tokens,
            response_data,
            None,
            http_status,
            ttft,
            tpot,
            itl_avg,
        )
