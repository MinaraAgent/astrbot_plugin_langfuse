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
from typing import Optional, TYPE_CHECKING

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

# Lazy import - langfuse will be imported when needed, AFTER AstrBot installs dependencies
Langfuse = None
LangfuseError = Exception
LANGFUSE_AVAILABLE = False

def _ensure_langfuse_imported():
    """Lazily import langfuse package. Returns True if successful."""
    global Langfuse, LangfuseError, LANGFUSE_AVAILABLE

    if LANGFUSE_AVAILABLE:
        return True

    try:
        from langfuse import Langfuse as _Langfuse
        # LangfuseError - just use Exception, the specific error class path varies by version
        Langfuse = _Langfuse
        LangfuseError = Exception
        LANGFUSE_AVAILABLE = True
        log_both("INFO", "langfuse package imported successfully (lazy import)")
        log_both("INFO", f"langfuse version: {getattr(_Langfuse, '__version__', 'unknown')}")
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

        # Call parent __init__
        super().__init__(context)

        # Store config - this is passed by AstrBot from _conf_schema.json
        self.plugin_config = config or {}
        self.langfuse_client = None
        self.enabled = False
        self.sessions: dict[str, SessionInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        log_both("INFO", f"Config type: {type(config)}")
        log_both("INFO", f"Config value: {config}")
        log_both("INFO", f"self.plugin_config keys: {list(self.plugin_config.keys()) if self.plugin_config else 'empty'}")

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
        """
        Called when plugin is activated.
        This is the correct method name for AstrBot (not 'init').
        """
        log_both("INFO", "=" * 60)
        log_both("INFO", "initialize() CALLED - Plugin activation started")
        log_both("INFO", f"Timestamp: {datetime.now().isoformat()}")

        # Try to import langfuse now (after AstrBot has installed dependencies)
        if not _ensure_langfuse_imported():
            log_both("ERROR", "Cannot initialize - langfuse package not available")
            log_both("ERROR", "Try: pip install langfuse in AstrBot's Python environment")
            return

        # Check if disabled in config
        if not self.plugin_config.get("enabled", True):
            log_both("INFO", "Plugin is DISABLED in config")
            return

        # Get config values
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
                enabled=True,
            )
            self.enabled = True
            log_both("INFO", f"Langfuse client created: {self.langfuse_client is not None}")

            # Test auth
            try:
                self.langfuse_client.auth_check()
                log_both("INFO", "Auth check PASSED - Langfuse connection successful!")
            except Exception as e:
                log_both("WARNING", f"Auth check failed: {e}")

            # Start cleanup task
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

    def _trace_message(self, session: SessionInfo, name: str, input_data: dict,
                       output_data: Optional[dict] = None, metadata: Optional[dict] = None):
        """Create a trace for a message event"""
        if not self.enabled or not self.langfuse_client:
            return None

        try:
            trace = self.langfuse_client.trace(
                id=session.trace_id,
                name=name,
                session_id=session.session_id,
                input=input_data,
                metadata=metadata or {},
            )

            if output_data:
                trace.update(output=output_data)

            self.langfuse_client.flush()
            log_both("INFO", f"Trace created: {name}")
            return trace

        except Exception as e:
            log_both("ERROR", f"Trace error: {e}")
            return None

    def _trace_llm_call(self, session: SessionInfo, model: str, prompt: str,
                        completion: str, prompt_tokens: Optional[int] = None,
                        completion_tokens: Optional[int] = None,
                        total_tokens: Optional[int] = None,
                        metadata: Optional[dict] = None):
        """Create a trace for an LLM call"""
        if not self.enabled or not self.langfuse_client:
            return None

        try:
            trace = self.langfuse_client.trace(
                id=session.trace_id,
                name="llm_call",
                session_id=session.session_id,
                metadata=metadata or {},
            )

            usage_dict = None
            if total_tokens:
                usage_dict = {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                }

            generation = trace.generation(
                name=f"llm_{model}",
                model=model,
                input=prompt,
                output=completion,
                usage=usage_dict,
            )

            self.langfuse_client.flush()
            log_both("INFO", f"LLM trace created: {model}")
            return generation

        except Exception as e:
            log_both("ERROR", f"LLM trace error: {e}")
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

            self._trace_message(
                session=session,
                name="user_message",
                input_data={"content": message_content, "user_id": user_id, "platform": platform},
                metadata={"environment": environment},
            )

        except Exception as e:
            log_both("ERROR", f"Message trace error: {e}")

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

            if not hasattr(event, "_langfuse_trace_id"):
                event._langfuse_trace_id = session.trace_id

            self._trace_message(
                session=session,
                name="llm_request",
                input_data={
                    "prompt": req.prompt,
                    "system_prompt": req.system_prompt,
                    "conversation_history": [
                        {"role": msg.role, "content": str(msg.content)}
                        for msg in (req.conversation_history or [])
                    ],
                },
                metadata={"environment": environment, "session_trace_id": session.trace_id},
            )

        except Exception as e:
            log_both("ERROR", f"LLM request trace error: {e}")

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

            completion_text = resp.completion_text or (str(resp.result_chain) if resp.result_chain else "")
            model = resp.model if hasattr(resp, "model") else "unknown"

            self._trace_llm_call(
                session=session,
                model=model,
                prompt="",
                completion=completion_text,
                prompt_tokens=resp.usage.prompt_tokens if resp.usage and hasattr(resp.usage, "prompt_tokens") else None,
                completion_tokens=resp.usage.completion_tokens if resp.usage and hasattr(resp.usage, "completion_tokens") else None,
                total_tokens=resp.usage.total_tokens if resp.usage and hasattr(resp.usage, "total_tokens") else None,
                metadata={"environment": environment},
            )

        except Exception as e:
            log_both("ERROR", f"LLM response trace error: {e}")
