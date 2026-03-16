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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from astrbot.api import logger as astrbot_logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

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
log_both("INFO", f"Working directory: {os.getcwd()}")
log_both("INFO", f"Debug log file: {DEBUG_LOG_FILE}")
log_both("INFO", "=" * 60)

# Lazy import - langfuse will be imported when needed
Langfuse = None
LANGFUSE_AVAILABLE = False

def _ensure_langfuse_imported():
    """Lazily import langfuse package. Returns True if successful."""
    global Langfuse, LANGFUSE_AVAILABLE

    if LANGFUSE_AVAILABLE:
        return True

    try:
        from langfuse import Langfuse as _Langfuse
        Langfuse = _Langfuse
        LANGFUSE_AVAILABLE = True
        log_both("INFO", "langfuse package imported successfully")
        return True
    except ImportError as e:
        log_both("ERROR", f"langfuse package not available: {e}")
        log_both("ERROR", "Make sure langfuse is installed. Check requirements.txt")
        return False


@dataclass
class SessionInfo:
    """Session information for tracking conversations"""
    session_id: str
    trace_id: str
    last_activity: float
    metadata: dict = field(default_factory=dict)


class LangfusePlugin(Star):
    """Langfuse integration plugin for AstrBot"""

    def __init__(self, context: Context, config: dict = None):
        """
        Initialize the plugin.

        Args:
            context: AstrBot context object
            config: Plugin configuration (passed by AstrBot from _conf_schema.json)
        """
        log_both("INFO", "-" * 50)
        log_both("INFO", "__init__ CALLED")
        log_both("INFO", f"Timestamp: {datetime.now().isoformat()}")

        super().__init__(context)

        self.plugin_config = config or {}
        self.langfuse_client = None
        self.enabled = False
        self.sessions: dict[str, SessionInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        log_both("INFO", f"Config type: {type(config)}")
        log_both("INFO", f"Config keys: {list(self.plugin_config.keys()) if self.plugin_config else 'empty'}")

        # Log config values safely
        for k, v in self.plugin_config.items():
            if "key" in k.lower() or "secret" in k.lower():
                safe_val = f"(len={len(str(v))})" if v else "(empty)"
            else:
                safe_val = str(v)[:50]
            log_both("INFO", f"  config[{k}]: {safe_val}")

        log_both("INFO", "__init__ completed")
        log_both("INFO", "-" * 50)

    async def initialize(self) -> None:
        """Called when plugin is activated."""
        log_both("INFO", "=" * 60)
        log_both("INFO", "initialize() CALLED - Plugin activation started")
        log_both("INFO", f"Timestamp: {datetime.now().isoformat()}")

        if not _ensure_langfuse_imported():
            log_both("ERROR", "Cannot initialize - langfuse package not available")
            return

        if not self.plugin_config.get("enabled", True):
            log_both("INFO", "Plugin is DISABLED in config")
            return

        secret_key = self.plugin_config.get("secret_key", "")
        public_key = self.plugin_config.get("public_key", "")
        base_url = self.plugin_config.get("base_url", "https://cloud.langfuse.com")

        log_both("INFO", "Config values:")
        log_both("INFO", f"  secret_key: {'(len=' + str(len(secret_key)) + ')' if secret_key else '(EMPTY)'}")
        log_both("INFO", f"  public_key: {'(len=' + str(len(public_key)) + ')' if public_key else '(EMPTY)'}")
        log_both("INFO", f"  base_url: {base_url}")

        if not secret_key or not public_key:
            log_both("ERROR", "SECRET_KEY or PUBLIC_KEY is empty - check plugin config in AstrBot")
            return

        try:
            log_both("INFO", "Creating Langfuse client...")
            self.langfuse_client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=base_url,
            )
            self.enabled = True
            log_both("INFO", f"Langfuse client created: {self.langfuse_client is not None}")

            # Test auth
            try:
                self.langfuse_client.auth_check()
                log_both("INFO", "Auth check PASSED - Langfuse connection successful!")
            except Exception as e:
                log_both("WARNING", f"Auth check failed: {e}")

            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
            log_both("INFO", "Session cleanup task started")
            log_both("INFO", "Plugin initialized SUCCESSFULLY")

        except Exception as e:
            log_both("ERROR", f"Failed to create Langfuse client: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")

        log_both("INFO", f"Final state - enabled: {self.enabled}")
        log_both("INFO", "=" * 60)

    async def terminate(self) -> None:
        """Called when plugin is disabled or reloaded."""
        log_both("INFO", "terminate() called")

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
                log_both("INFO", "Flushed remaining traces")
            except Exception as e:
                log_both("ERROR", f"Error flushing: {e}")

        log_both("INFO", "Plugin terminated")

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

                if expired_keys:
                    log_both("INFO", f"Cleaned up {len(expired_keys)} expired sessions")

                if self.langfuse_client:
                    self.langfuse_client.flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log_both("ERROR", f"Cleanup error: {e}")

    def _create_trace(self, session: SessionInfo, name: str, input_data: dict,
                      output_data: Optional[dict] = None, metadata: Optional[dict] = None):
        """Create a trace using the new langfuse SDK API"""
        if not self.enabled or not self.langfuse_client:
            return None

        try:
            # Use the new SDK API: start_observation creates a span
            # At the top level, this automatically creates a trace
            environment = self.plugin_config.get("environment", "production")

            observation = self.langfuse_client.start_observation(
                name=name,
                as_type="span",
                input=input_data,
                metadata=metadata or {},
            )

            # Set trace-level attributes
            observation.update(
                session_id=session.session_id,
                metadata={"environment": environment, **(metadata or {})},
            )

            if output_data:
                observation.update(output=output_data)

            observation.end()

            # Flush to ensure data is sent
            self.langfuse_client.flush()
            log_both("INFO", f"Trace created: {name}")
            return observation

        except Exception as e:
            log_both("ERROR", f"Trace error: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")
            return None

    def _create_generation(self, session: SessionInfo, model: str, prompt: str,
                           completion: str, prompt_tokens: Optional[int] = None,
                           completion_tokens: Optional[int] = None,
                           total_tokens: Optional[int] = None,
                           metadata: Optional[dict] = None):
        """Create a generation trace using the new langfuse SDK API"""
        if not self.enabled or not self.langfuse_client:
            return None

        try:
            environment = self.plugin_config.get("environment", "production")

            # Create a generation observation
            generation = self.langfuse_client.start_observation(
                name=f"llm_{model}",
                as_type="generation",
                model=model,
                input=prompt if prompt else {"query": "N/A"},
                output=completion,
                metadata={"environment": environment, **(metadata or {})},
            )

            # Update with usage if available
            if total_tokens:
                generation.update(
                    usage={
                        "input": prompt_tokens or 0,
                        "output": completion_tokens or 0,
                        "total": total_tokens,
                    }
                )

            generation.end()

            self.langfuse_client.flush()
            log_both("INFO", f"Generation trace created: {model}")
            return generation

        except Exception as e:
            log_both("ERROR", f"Generation trace error: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")
            return None

    @filter.command("langfuse_status")
    async def langfuse_status(self, event: AstrMessageEvent):
        """Check Langfuse connection status"""
        if not self.enabled:
            yield event.plain_result(
                f"Langfuse is not enabled.\n"
                f"enabled={self.enabled}, client={self.langfuse_client is not None}\n"
                f"Check logs at: {DEBUG_LOG_FILE}"
            )
            return

        base_url = self.plugin_config.get("base_url", "https://cloud.langfuse.com")
        active_sessions = len(self.sessions)

        status_msg = (
            f"Langfuse Status:\n"
            f"- Status: Enabled\n"
            f"- Base URL: {base_url}\n"
            f"- Active Sessions: {active_sessions}\n"
            f"- LLM Tracing: {'Enabled' if self.plugin_config.get('enabled_llm_tracing', True) else 'Disabled'}\n"
            f"- Message Tracing: {'Enabled' if self.plugin_config.get('enabled_message_tracing', True) else 'Disabled'}"
        )
        yield event.plain_result(status_msg)

    @filter.command("langfuse_flush")
    async def langfuse_flush(self, event: AstrMessageEvent):
        """Manually flush Langfuse traces"""
        if not self.enabled or not self.langfuse_client:
            yield event.plain_result(f"Langfuse is not enabled. (enabled={self.enabled})")
            return

        try:
            self.langfuse_client.flush()
            yield event.plain_result("Langfuse traces flushed successfully.")
        except Exception as e:
            yield event.plain_result(f"Failed to flush: {e}")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """Handle all message events for tracing"""
        if not self.enabled or not self.plugin_config.get("enabled_message_tracing", True):
            return

        try:
            user_id = event.unified_msg_origin
            platform = event.get_platform_name() if hasattr(event, "get_platform_name") else "unknown"
            session = self._get_or_create_session(user_id, platform)
            message_content = event.message_str or ""
            environment = self.plugin_config.get("environment", "production")

            self._create_trace(
                session=session,
                name="user_message",
                input_data={"content": message_content, "user_id": user_id, "platform": platform},
                metadata={"environment": environment},
            )

        except Exception as e:
            log_both("ERROR", f"Message trace error: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Hook into LLM request for tracing"""
        if not self.enabled or not self.plugin_config.get("enabled_llm_tracing", True):
            return

        try:
            user_id = event.unified_msg_origin
            platform = event.get_platform_name() if hasattr(event, "get_platform_name") else "unknown"
            session = self._get_or_create_session(user_id, platform)
            environment = self.plugin_config.get("environment", "production")

            # Build input data - check what attributes ProviderRequest has
            input_data = {
                "prompt": req.prompt if hasattr(req, 'prompt') else str(req),
            }

            if hasattr(req, 'system_prompt') and req.system_prompt:
                input_data["system_prompt"] = req.system_prompt

            # Check for conversation history with different possible attribute names
            if hasattr(req, 'contexts') and req.contexts:
                input_data["contexts"] = [str(c) for c in req.contexts]
            elif hasattr(req, 'history') and req.history:
                input_data["history"] = [str(h) for h in req.history]

            log_both("INFO", f"LLM request - ProviderRequest attrs: {[a for a in dir(req) if not a.startswith('_')]}")

            self._create_trace(
                session=session,
                name="llm_request",
                input_data=input_data,
                metadata={"environment": environment},
            )

        except Exception as e:
            log_both("ERROR", f"LLM request trace error: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """Hook into LLM response for tracing"""
        if not self.enabled or not self.plugin_config.get("enabled_llm_tracing", True):
            return

        try:
            user_id = event.unified_msg_origin
            platform = event.get_platform_name() if hasattr(event, "get_platform_name") else "unknown"
            session = self._get_or_create_session(user_id, platform)
            environment = self.plugin_config.get("environment", "production")

            completion_text = ""
            if hasattr(resp, 'completion_text') and resp.completion_text:
                completion_text = resp.completion_text
            elif hasattr(resp, 'result_chain') and resp.result_chain:
                completion_text = str(resp.result_chain)

            model = resp.model if hasattr(resp, "model") else "unknown"

            log_both("INFO", f"LLM response - LLMResponse attrs: {[a for a in dir(resp) if not a.startswith('_')]}")

            self._create_generation(
                session=session,
                model=model,
                prompt="",
                completion=completion_text,
                prompt_tokens=resp.usage.prompt_tokens if hasattr(resp, 'usage') and resp.usage and hasattr(resp.usage, 'prompt_tokens') else None,
                completion_tokens=resp.usage.completion_tokens if hasattr(resp, 'usage') and resp.usage and hasattr(resp.usage, 'completion_tokens') else None,
                total_tokens=resp.usage.total_tokens if hasattr(resp, 'usage') and resp.usage and hasattr(resp.usage, 'total_tokens') else None,
                metadata={"environment": environment},
            )

        except Exception as e:
            log_both("ERROR", f"LLM response trace error: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")
