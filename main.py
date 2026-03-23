"""
AstrBot Langfuse Integration Plugin

This plugin integrates Langfuse LLM observability with AstrBot,
enabling tracing of LLM calls and message events.
"""

import asyncio
import logging
import os
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from astrbot.api import logger as astrbot_logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

# Context variable for plugin metadata (shared with other plugins like VideoVision)
# Plugins can set this before LLM calls to customize Langfuse observation names
_langfuse_observation_ctx: ContextVar[Optional[dict]] = ContextVar('langfuse_observation')

# Try to get the context variable from VideoVision plugin if available
try:
    from astrbot_plugin_video_vision.main import langfuse_observation_ctx
    # VideoVision plugin is loaded, use its context variable
except ImportError:
    # VideoVision not loaded, use our own
    langfuse_observation_ctx = _langfuse_observation_ctx

# Create a dedicated logger that writes to file for debugging
DEBUG_LOG_FILE = "/tmp/astrbot_langfuse_debug.log"

def setup_debug_logger():
    """Setup a debug logger that writes to a file"""
    debug_logger = logging.getLogger("langfuse_plugin")
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.handlers = []
    try:
        file_handler = logging.FileHandler(DEBUG_LOG_FILE, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        debug_logger.addHandler(file_handler)
    except Exception:
        pass
    return debug_logger

debug_log = setup_debug_logger()

def log_both(level, msg):
    """Log to both AstrBot logger and debug file"""
    try:
        if level == "DEBUG":
            astrbot_logger.debug(f"[Langfuse] {msg}")
        elif level == "INFO":
            astrbot_logger.info(f"[Langfuse] {msg}")
        elif level == "WARNING":
            astrbot_logger.warning(f"[Langfuse] {msg}")
        elif level == "ERROR":
            astrbot_logger.error(f"[Langfuse] {msg}")
    except Exception:
        pass
    debug_log.debug(msg)

# Log module load
log_both("INFO", "=" * 60)
log_both("INFO", "MODULE LOADED - astrbot_plugin_langfuse")
log_both("INFO", f"Python version: {sys.version}")
log_both("INFO", f"Debug log file: {DEBUG_LOG_FILE}")
log_both("INFO", "=" * 60)

# Lazy import
Langfuse = None
propagate_attributes = None
LANGFUSE_AVAILABLE = False

def _ensure_langfuse_imported():
    """Lazily import langfuse package."""
    global Langfuse, propagate_attributes, LANGFUSE_AVAILABLE

    if LANGFUSE_AVAILABLE:
        return True

    try:
        from langfuse import Langfuse as _Langfuse
        from langfuse._client.propagation import propagate_attributes as _propagate_attributes
        Langfuse = _Langfuse
        propagate_attributes = _propagate_attributes
        LANGFUSE_AVAILABLE = True
        log_both("INFO", "langfuse package imported successfully")
        return True
    except ImportError as e:
        log_both("ERROR", f"langfuse package not available: {e}")
        return False


@dataclass
class SessionInfo:
    """Session information for tracking conversations"""
    session_id: str
    trace_id: str
    last_activity: float
    current_observation: Optional[object] = None
    metadata: dict = field(default_factory=dict)


class LangfusePlugin(Star):
    """Langfuse integration plugin for AstrBot"""

    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)

        self.plugin_config = config or {}
        self.langfuse_client = None
        self.enabled = False
        self.sessions: dict[str, SessionInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        log_both("INFO", f"Config keys: {list(self.plugin_config.keys())}")

    async def initialize(self) -> None:
        """Called when plugin is activated."""
        if not _ensure_langfuse_imported():
            return

        if not self.plugin_config.get("enabled", True):
            log_both("INFO", "Plugin is DISABLED in config")
            return

        secret_key = self.plugin_config.get("secret_key", "")
        public_key = self.plugin_config.get("public_key", "")
        base_url = self.plugin_config.get("base_url", "https://cloud.langfuse.com")

        if not secret_key or not public_key:
            log_both("ERROR", "SECRET_KEY or PUBLIC_KEY is empty")
            return

        try:
            self.langfuse_client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=base_url,
            )
            self.enabled = True

            try:
                self.langfuse_client.auth_check()
                log_both("INFO", "Auth check PASSED - Langfuse connected!")
            except Exception as e:
                log_both("WARNING", f"Auth check failed: {e}")

            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
            log_both("INFO", "Plugin initialized successfully")

        except Exception as e:
            log_both("ERROR", f"Failed to create Langfuse client: {e}")

    async def terminate(self) -> None:
        """Called when plugin is disabled or reloaded."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
            except Exception:
                pass

    def _get_or_create_session(self, user_id: str, platform: str) -> SessionInfo:
        """Get or create a session for a user"""
        session_key = f"{platform}:{user_id}"
        current_time = time.time()
        session_timeout = self.plugin_config.get("session_timeout", 3600)

        if session_key in self.sessions:
            session = self.sessions[session_key]
            if current_time - session.last_activity > session_timeout:
                session = SessionInfo(
                    session_id=str(uuid.uuid4()),
                    trace_id=str(uuid.uuid4()),
                    last_activity=current_time,
                    metadata={"platform": platform, "user_id": user_id},
                )
                self.sessions[session_key] = session
            else:
                session.last_activity = current_time
        else:
            session = SessionInfo(
                session_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                last_activity=current_time,
                metadata={"platform": platform, "user_id": user_id},
            )
            self.sessions[session_key] = session

        return session

    async def _cleanup_sessions(self):
        """Periodically cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)
                current_time = time.time()
                session_timeout = self.plugin_config.get("session_timeout", 3600)

                expired_keys = [
                    key for key, session in self.sessions.items()
                    if current_time - session.last_activity > session_timeout
                ]

                for key in expired_keys:
                    del self.sessions[key]

                if self.langfuse_client:
                    self.langfuse_client.flush()

            except asyncio.CancelledError:
                break
            except Exception:
                pass

    @filter.command("langfuse_status")
    async def langfuse_status(self, event: AstrMessageEvent):
        """Check Langfuse connection status"""
        if not self.enabled:
            yield event.plain_result(
                f"Langfuse not enabled.\n"
                f"Check logs: {DEBUG_LOG_FILE}"
            )
            return

        base_url = self.plugin_config.get("base_url", "https://cloud.langfuse.com")
        status_msg = (
            f"Langfuse Status:\n"
            f"- Status: Enabled\n"
            f"- Base URL: {base_url}\n"
            f"- Active Sessions: {len(self.sessions)}"
        )
        yield event.plain_result(status_msg)

    @filter.command("langfuse_flush")
    async def langfuse_flush(self, event: AstrMessageEvent):
        """Manually flush Langfuse traces"""
        if not self.enabled or not self.langfuse_client:
            yield event.plain_result("Langfuse not enabled.")
            return

        try:
            self.langfuse_client.flush()
            yield event.plain_result("Traces flushed.")
        except Exception as e:
            yield event.plain_result(f"Flush failed: {e}")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """Handle all message events for tracing"""
        if not self.enabled or not self.plugin_config.get("enabled_message_tracing", True):
            return

        if not self.langfuse_client:
            return

        try:
            user_id = event.unified_msg_origin
            platform = event.get_platform_name() if hasattr(event, "get_platform_name") else "unknown"
            session = self._get_or_create_session(user_id, platform)
            message_content = event.message_str or ""
            environment = self.plugin_config.get("environment", "production")

            # Use propagate_attributes to properly set session_id and user_id
            # This ensures all observations inherit these trace-level attributes
            with propagate_attributes(
                session_id=session.session_id,
                user_id=user_id,
                metadata={
                    "environment": environment,
                    "platform": platform,
                },
            ):
                observation = self.langfuse_client.start_observation(
                    name="user_message",
                    as_type="span",
                    input={"content": message_content},
                )
                observation.end()

        except Exception as e:
            log_both("ERROR", f"Message trace error: {e}")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Hook into LLM request for tracing"""
        if not self.enabled or not self.plugin_config.get("enabled_llm_tracing", True):
            return

        if not self.langfuse_client:
            return

        try:
            user_id = event.unified_msg_origin
            platform = event.get_platform_name() if hasattr(event, "get_platform_name") else "unknown"
            session = self._get_or_create_session(user_id, platform)
            environment = self.plugin_config.get("environment", "production")

            # Build input data
            input_data = {}

            if req.prompt:
                input_data["prompt"] = req.prompt

            if req.system_prompt:
                input_data["system_prompt"] = req.system_prompt

            if req.contexts:
                # Truncate contexts to avoid huge payloads
                input_data["contexts_count"] = len(req.contexts)
                input_data["contexts"] = req.contexts[:10]  # First 10 messages

            if req.image_urls:
                input_data["image_count"] = len(req.image_urls)

            if req.extra_user_content_parts:
                # Include extra content parts (this is where VideoVision injects analysis)
                extra_parts = []
                for part in req.extra_user_content_parts:
                    # Convert ContentPart to dict for logging
                    if hasattr(part, 'model_dump'):
                        extra_parts.append(part.model_dump())
                    elif hasattr(part, 'dict'):
                        extra_parts.append(part.dict())
                    elif isinstance(part, dict):
                        extra_parts.append(part)
                    else:
                        # Fallback: try to get text attribute
                        extra_parts.append({"type": "unknown", "repr": str(part)})

                input_data["extra_user_content_parts"] = extra_parts
                input_data["extra_user_content_count"] = len(extra_parts)

                # Log if video vision analysis is detected
                for part in extra_parts:
                    part_text = part.get("text", "") if isinstance(part, dict) else ""
                    if "[Video Content Analysis]" in part_text:
                        log_both("INFO", f"VideoVision analysis detected in LLM input ({len(part_text)} chars)")
                        input_data["has_video_analysis"] = True

            # Get model name
            model = req.model or "unknown"

            # Store model in session for use in response
            session.metadata["model"] = model
            session.metadata["prompt"] = req.prompt

            # Determine observation name based on context variable or content
            observation_name = "llm_generation"
            observation_metadata = {}

            # Check if a plugin has set observation metadata via context variable
            ctx_observation = langfuse_observation_ctx.get()
            if ctx_observation:
                observation_name = ctx_observation.get("name", "llm_generation")
                observation_metadata = ctx_observation.get("metadata", {})
                log_both("INFO", f"Using custom observation name from context: {observation_name}")
            else:
                # Fallback: check if video analysis was injected into this request (main conversation)
                if req.extra_user_content_parts:
                    for part in req.extra_user_content_parts:
                        part_text = ""
                        if hasattr(part, 'text'):
                            part_text = part.text
                        elif hasattr(part, 'model_dump'):
                            part_dict = part.model_dump()
                            part_text = part_dict.get("text", "")
                        elif isinstance(part, dict):
                            part_text = part.get("text", "")

                        if "[Video Content Analysis]" in part_text:
                            observation_name = "Main Conversation (with VideoVision)"
                            observation_metadata["has_video_analysis"] = True
                            break

            # Store observation name in session for use in response
            session.metadata["observation_name"] = observation_name

            # Use propagate_attributes to properly set session_id and user_id
            with propagate_attributes(
                session_id=session.session_id,
                user_id=user_id,
                metadata={
                    "environment": environment,
                    "platform": platform,
                    **observation_metadata,
                },
            ):
                # Create a generation observation (will be updated in response)
                observation = self.langfuse_client.start_observation(
                    name=observation_name,
                    as_type="generation",
                    model=model,
                    input=input_data,
                )

                # Store observation in session for later update
                session.current_observation = observation
                session.metadata["observation_id"] = observation.observation_id if hasattr(observation, 'observation_id') else None

            log_both("INFO", f"LLM request traced - model: {model}")

        except Exception as e:
            log_both("ERROR", f"LLM request trace error: {e}")
            log_both("ERROR", traceback.format_exc())

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """Hook into LLM response for tracing"""
        if not self.enabled or not self.plugin_config.get("enabled_llm_tracing", True):
            return

        if not self.langfuse_client:
            return

        try:
            user_id = event.unified_msg_origin
            platform = event.get_platform_name() if hasattr(event, "get_platform_name") else "unknown"
            session = self._get_or_create_session(user_id, platform)
            environment = self.plugin_config.get("environment", "production")

            # Get completion text
            completion_text = ""
            if resp.completion_text:
                completion_text = resp.completion_text
            elif resp.result_chain:
                completion_text = resp.result_chain.get_plain_text() if hasattr(resp.result_chain, 'get_plain_text') else str(resp.result_chain)

            # Get model name - try multiple sources
            model = session.metadata.get("model", "unknown")

            # Try to get model from raw_completion
            if resp.raw_completion and hasattr(resp.raw_completion, 'model'):
                model = resp.raw_completion.model

            # Get usage info
            usage_dict = None
            if resp.usage:
                usage_dict = {
                    "input": resp.usage.input if hasattr(resp.usage, 'input') else 0,
                    "output": resp.usage.output if hasattr(resp.usage, 'output') else 0,
                    "total": resp.usage.total if hasattr(resp.usage, 'total') else 0,
                }

            # Check if we have a pending observation from request
            if session.current_observation:
                # Update the existing observation (already has session_id and user_id from request)
                session.current_observation.update(
                    output=completion_text,
                    model=model,
                )

                # Log custom observation name if used
                obs_name = session.metadata.get("observation_name", "llm_generation")
                if obs_name != "llm_generation":
                    log_both("INFO", f"Custom observation '{obs_name}' completed")

                if usage_dict:
                    session.current_observation.update(usage=usage_dict)

                session.current_observation.end()
                session.current_observation = None
            else:
                # Create a new generation observation with propagate_attributes
                with propagate_attributes(
                    session_id=session.session_id,
                    user_id=user_id,
                    metadata={
                        "environment": environment,
                        "platform": platform,
                    },
                ):
                    generation = self.langfuse_client.start_observation(
                        name="llm_generation",
                        as_type="generation",
                        model=model,
                        input=session.metadata.get("prompt", ""),
                        output=completion_text,
                    )

                    if usage_dict:
                        generation.update(usage=usage_dict)

                    generation.end()

            # Flush to ensure data is sent
            self.langfuse_client.flush()

            log_both("INFO", f"LLM response traced - model: {model}, tokens: {usage_dict}")

        except Exception as e:
            log_both("ERROR", f"LLM response trace error: {e}")
            log_both("ERROR", traceback.format_exc())
