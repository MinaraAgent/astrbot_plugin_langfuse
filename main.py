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

# Use both AstrBot logger AND Python's standard logging for maximum visibility
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

    # Clear existing handlers
    debug_logger.handlers = []

    # File handler
    try:
        file_handler = logging.FileHandler(DEBUG_LOG_FILE, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        debug_logger.addHandler(file_handler)
    except Exception as e:
        pass  # Silently fail if can't create file

    return debug_logger

# Setup debug logger
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

try:
    from langfuse import Langfuse
    from langfuse.api.resources.commons.errors.errors import BaseError as LangfuseError

    LANGFUSE_AVAILABLE = True
    log_both("INFO", "langfuse package is available")
    log_both("INFO", f"langfuse module location: {Langfuse.__module__ if hasattr(Langfuse, '__module__') else 'unknown'}")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    LangfuseError = Exception
    log_both("WARNING", f"langfuse package not installed: {e}")


@dataclass
class SessionInfo:
    """Session information for tracking conversations"""

    session_id: str
    trace_id: str
    last_activity: float
    metadata: dict = field(default_factory=dict)


class LangfusePlugin(Star):
    """Langfuse integration plugin for AstrBot"""

    def __init__(self, context: Context):
        log_both("INFO", "-" * 50)
        log_both("INFO", "__init__ CALLED - Plugin class being instantiated")
        log_both("INFO", f"Timestamp: {datetime.now().isoformat()}")

        super().__init__(context)
        self.langfuse_client: Optional[Langfuse] = None
        self.enabled = False
        self.config = {}
        self.sessions: dict[str, SessionInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        log_both("INFO", f"Context type: {type(context)}")
        log_both("INFO", f"Context dir: {[x for x in dir(context) if not x.startswith('_')]}")

        # Check if context has config-related attributes
        if hasattr(context, 'config'):
            log_both("INFO", f"context.config: {context.config}")
        if hasattr(context, 'get_config'):
            log_both("INFO", "context has get_config method")

        log_both("INFO", "__init__ completed")
        log_both("INFO", "-" * 50)

    async def _get_config(self) -> dict:
        """Get plugin configuration from AstrBot config"""
        log_both("INFO", ">>> _get_config() called")
        log_both("INFO", f"  self.config type: {type(self.config)}")
        log_both("INFO", f"  self.config content: {self.config}")

        config = self.config or {}

        log_both("INFO", f"  Checking context.get_config...")
        if hasattr(self.context, "get_config"):
            try:
                context_config = self.context.get_config()
                log_both("INFO", f"  context.get_config() returned type: {type(context_config)}")
                log_both("INFO", f"  context.get_config() returned: {context_config}")
                if context_config:
                    config = context_config
            except Exception as e:
                log_both("WARNING", f"  context.get_config() raised exception: {e}")
        else:
            log_both("INFO", "  context does NOT have get_config method")

        # Also check for 'config' attribute
        if hasattr(self.context, "config"):
            log_both("INFO", f"  context.config: {self.context.config}")

        # Log each config key (without exposing full secret values)
        log_both("INFO", "  Final config values (safe):")
        for k, v in config.items():
            if "key" in k.lower() or "secret" in k.lower():
                safe_value = f"{str(v)[:8]}..." if v else "(empty)"
            else:
                safe_value = repr(v)[:50]
            log_both("INFO", f"    {k}: {safe_value}")

        return config

    async def _init_langfuse(self) -> bool:
        """Initialize Langfuse client"""
        log_both("INFO", ">>> _init_langfuse() called")

        if not LANGFUSE_AVAILABLE:
            log_both("ERROR", "langfuse package not installed. Run: pip install langfuse")
            return False

        log_both("INFO", "Calling _get_config()...")
        config = await self._get_config()
        log_both("INFO", f"_get_config() returned config with keys: {list(config.keys()) if config else 'None'}")

        secret_key = config.get("secret_key", "")
        public_key = config.get("public_key", "")
        base_url = config.get("base_url", "https://cloud.langfuse.com")
        enabled = config.get("enabled", True)

        log_both("INFO", "Extracted config values:")
        log_both("INFO", f"  - secret_key: {'(set, len=' + str(len(secret_key)) + ')' if secret_key else '(EMPTY)'}")
        log_both("INFO", f"  - public_key: {'(set, len=' + str(len(public_key)) + ')' if public_key else '(EMPTY)'}")
        log_both("INFO", f"  - base_url: {base_url}")
        log_both("INFO", f"  - enabled: {enabled}")

        if not secret_key or not public_key:
            log_both("ERROR", "SECRET KEY or PUBLIC KEY is EMPTY - cannot initialize Langfuse client")
            log_both("ERROR", "Please check your plugin configuration in AstrBot")
            return False

        try:
            log_both("INFO", "Creating Langfuse client instance...")
            log_both("INFO", f"  Parameters: host={base_url}, enabled={enabled}")

            self.langfuse_client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=base_url,
                enabled=enabled,
            )

            log_both("INFO", f"Langfuse client created: {self.langfuse_client is not None}")
            log_both("INFO", f"Langfuse client type: {type(self.langfuse_client)}")

            # Log internal state of the client
            if hasattr(self.langfuse_client, 'enabled'):
                log_both("INFO", f"Client internal enabled: {self.langfuse_client.enabled}")
            if hasattr(self.langfuse_client, 'base_url'):
                log_both("INFO", f"Client internal base_url: {self.langfuse_client.base_url}")

            self.enabled = True
            log_both("INFO", f"Plugin self.enabled set to: {self.enabled}")

            # Test the connection by trying to auth
            log_both("INFO", "Testing connection with auth_check()...")
            try:
                auth_result = self.langfuse_client.auth_check()
                log_both("INFO", f"Auth check SUCCESS: {auth_result}")
            except Exception as auth_err:
                log_both("WARNING", f"Auth check failed (but client created): {auth_err}")
                log_both("WARNING", f"Auth check traceback: {traceback.format_exc()}")

            return True
        except Exception as e:
            log_both("ERROR", f"Failed to initialize Langfuse client: {e}")
            log_both("ERROR", f"Exception type: {type(e)}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")
            return False

    def _get_or_create_session(self, user_id: str, platform: str) -> SessionInfo:
        """Get or create a session for a user"""
        session_key = f"{platform}:{user_id}"
        current_time = time.time()
        session_timeout = self.config.get("session_timeout", 3600)

        log_both("INFO", f"_get_or_create_session - key: {session_key}")

        if session_key in self.sessions:
            session = self.sessions[session_key]
            # Check if session has expired
            if current_time - session.last_activity > session_timeout:
                # Create new session
                log_both("INFO", f"  Session expired, creating new session")
                session = SessionInfo(
                    session_id=str(uuid.uuid4()),
                    trace_id=str(uuid.uuid4()),
                    last_activity=current_time,
                    metadata={"platform": platform, "user_id": user_id},
                )
                self.sessions[session_key] = session
            else:
                session.last_activity = current_time
                log_both("INFO", f"  Using existing session")
        else:
            log_both("INFO", f"  Creating new session")
            session = SessionInfo(
                session_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                last_activity=current_time,
                metadata={"platform": platform, "user_id": user_id},
            )
            self.sessions[session_key] = session

        log_both("INFO", f"  session_id: {session.session_id[:8]}...")
        log_both("INFO", f"  trace_id: {session.trace_id[:8]}...")
        return session

    async def _cleanup_sessions(self):
        """Periodically cleanup expired sessions"""
        log_both("INFO", "Session cleanup task started")
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                current_time = time.time()
                session_timeout = self.config.get("session_timeout", 3600)

                expired_keys = [
                    key
                    for key, session in self.sessions.items()
                    if current_time - session.last_activity > session_timeout
                ]

                for key in expired_keys:
                    del self.sessions[key]

                if expired_keys:
                    log_both("INFO", f"Cleaned up {len(expired_keys)} expired sessions")

                # Flush Langfuse client periodically
                if self.langfuse_client:
                    self.langfuse_client.flush()
                    log_both("INFO", "Periodic flush completed")

            except asyncio.CancelledError:
                log_both("INFO", "Session cleanup task cancelled")
                break
            except Exception as e:
                log_both("ERROR", f"Error in session cleanup: {e}")

    def _trace_message(
        self,
        session: SessionInfo,
        name: str,
        input_data: dict,
        output_data: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        """Create a trace for a message event"""
        log_both("INFO", f">>> _trace_message() called - name: {name}")
        log_both("INFO", f"  self.enabled: {self.enabled}")
        log_both("INFO", f"  self.langfuse_client: {self.langfuse_client is not None}")

        if not self.enabled:
            log_both("WARNING", "  SKIPPING - plugin not enabled")
            return None

        if not self.langfuse_client:
            log_both("WARNING", "  SKIPPING - no langfuse_client")
            return None

        try:
            log_both("INFO", f"  Creating trace object...")
            log_both("INFO", f"    trace_id: {session.trace_id}")
            log_both("INFO", f"    session_id: {session.session_id}")
            log_both("INFO", f"    name: {name}")

            trace = self.langfuse_client.trace(
                id=session.trace_id,
                name=name,
                session_id=session.session_id,
                input=input_data,
                metadata=metadata or {},
            )

            log_both("INFO", f"  Trace object created: {trace is not None}")
            log_both("INFO", f"  Trace type: {type(trace)}")

            if output_data:
                log_both("INFO", f"  Updating trace with output_data...")
                trace.update(output=output_data)

            # Flush immediately for debugging
            log_both("INFO", "  Calling flush()...")
            self.langfuse_client.flush()
            log_both("INFO", "  flush() completed")

            log_both("INFO", f"  _trace_message() SUCCESS for '{name}'")

            return trace
        except LangfuseError as e:
            log_both("ERROR", f"  LangfuseError creating trace: {e}")
            log_both("ERROR", f"  Traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            log_both("ERROR", f"  Unexpected error creating trace: {e}")
            log_both("ERROR", f"  Exception type: {type(e)}")
            log_both("ERROR", f"  Traceback: {traceback.format_exc()}")
            return None

    def _trace_llm_call(
        self,
        session: SessionInfo,
        model: str,
        prompt: str,
        completion: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        metadata: Optional[dict] = None,
    ):
        """Create a trace for an LLM call"""
        log_both("INFO", f">>> _trace_llm_call() called - model: {model}")
        log_both("INFO", f"  self.enabled: {self.enabled}")
        log_both("INFO", f"  self.langfuse_client: {self.langfuse_client is not None}")

        if not self.enabled:
            log_both("WARNING", "  SKIPPING - plugin not enabled")
            return None

        if not self.langfuse_client:
            log_both("WARNING", "  SKIPPING - no langfuse_client")
            return None

        try:
            log_both("INFO", f"  Creating LLM trace...")
            log_both("INFO", f"    model: {model}")
            log_both("INFO", f"    prompt_len: {len(prompt)}")
            log_both("INFO", f"    completion_len: {len(completion)}")
            log_both("INFO", f"    tokens - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

            trace = self.langfuse_client.trace(
                id=session.trace_id,
                name="llm_call",
                session_id=session.session_id,
                metadata=metadata or {},
            )

            log_both("INFO", f"  Trace object created, adding generation...")

            usage_dict = None
            if total_tokens:
                usage_dict = {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                }
                log_both("INFO", f"  Usage dict: {usage_dict}")

            generation = trace.generation(
                name=f"llm_{model}",
                model=model,
                input=prompt,
                output=completion,
                usage=usage_dict,
            )

            log_both("INFO", f"  Generation created: {generation is not None}")

            # Flush immediately for debugging
            log_both("INFO", "  Calling flush()...")
            self.langfuse_client.flush()
            log_both("INFO", "  flush() completed")

            log_both("INFO", f"  _trace_llm_call() SUCCESS for model: {model}")

            return generation
        except LangfuseError as e:
            log_both("ERROR", f"  LangfuseError tracing LLM call: {e}")
            log_both("ERROR", f"  Traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            log_both("ERROR", f"  Unexpected error tracing LLM call: {e}")
            log_both("ERROR", f"  Exception type: {type(e)}")
            log_both("ERROR", f"  Traceback: {traceback.format_exc()}")
            return None

    async def init(self, config: dict):
        """Initialize the plugin"""
        log_both("INFO", "=" * 60)
        log_both("INFO", "init() CALLED - Plugin initialization started")
        log_both("INFO", f"Timestamp: {datetime.now().isoformat()}")
        log_both("INFO", f"Config type: {type(config)}")
        log_both("INFO", f"Config keys: {list(config.keys()) if config else 'None or empty'}")

        # Log each config value (safely)
        if config:
            for key, value in config.items():
                if "key" in key.lower() or "secret" in key.lower():
                    safe_value = f"(set, len={len(str(value))})" if value else "(empty)"
                else:
                    safe_value = repr(value)[:100]
                log_both("INFO", f"  Config[{key}]: {safe_value}")

        self.config = config or {}

        if not config.get("enabled", True):
            log_both("WARNING", "Plugin is DISABLED in config - skipping initialization")
            log_both("INFO", "=" * 60)
            return

        log_both("INFO", "Calling _init_langfuse()...")
        success = await self._init_langfuse()

        log_both("INFO", f"_init_langfuse() returned: {success}")

        if success:
            # Start session cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
            log_both("INFO", "Session cleanup task started")
            log_both("INFO", "Plugin initialized SUCCESSFULLY")
        else:
            log_both("ERROR", "Plugin initialization FAILED")

        log_both("INFO", f"Final state - enabled: {self.enabled}, langfuse_client: {self.langfuse_client is not None}")
        log_both("INFO", "=" * 60)

    async def terminate(self):
        """Cleanup when plugin is stopped"""
        log_both("INFO", ">>> terminate() called - Plugin stopping")

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
                log_both("ERROR", f"Error flushing traces: {e}")

        log_both("INFO", "Plugin terminated")

    @filter.command("langfuse_status")
    async def langfuse_status(self, event: AstrMessageEvent):
        """Check Langfuse connection status"""
        log_both("INFO", ">>> langfuse_status command called")

        if not LANGFUSE_AVAILABLE:
            log_both("WARNING", "  Langfuse package not installed")
            yield event.plain_result(
                "Langfuse package is not installed. Run: pip install langfuse"
            )
            return

        if not self.enabled:
            log_both("WARNING", f"  Plugin not enabled (self.enabled={self.enabled})")
            yield event.plain_result(
                f"Langfuse is not enabled or not configured properly.\n"
                f"enabled={self.enabled}, client={self.langfuse_client is not None}"
            )
            return

        config = await self._get_config()
        base_url = config.get("base_url", "https://cloud.langfuse.com")
        active_sessions = len(self.sessions)

        status_msg = (
            f"Langfuse Status:\n"
            f"- Status: Enabled\n"
            f"- Base URL: {base_url}\n"
            f"- Active Sessions: {active_sessions}\n"
            f"- LLM Tracing: {'Enabled' if config.get('enabled_llm_tracing', True) else 'Disabled'}\n"
            f"- Message Tracing: {'Enabled' if config.get('enabled_message_tracing', True) else 'Disabled'}\n"
            f"- Debug Log: {DEBUG_LOG_FILE}"
        )
        log_both("INFO", f"  Returning status: {status_msg}")
        yield event.plain_result(status_msg)

    @filter.command("langfuse_flush")
    async def langfuse_flush(self, event: AstrMessageEvent):
        """Manually flush Langfuse traces"""
        log_both("INFO", ">>> langfuse_flush command called")

        if not self.enabled or not self.langfuse_client:
            log_both("WARNING", f"  Cannot flush - enabled={self.enabled}, client={self.langfuse_client is not None}")
            yield event.plain_result(f"Langfuse is not enabled. (enabled={self.enabled})")
            return

        try:
            log_both("INFO", "  Calling flush()...")
            self.langfuse_client.flush()
            log_both("INFO", "  flush() completed successfully")
            yield event.plain_result("Langfuse traces flushed successfully.")
        except Exception as e:
            log_both("ERROR", f"  flush() failed: {e}")
            yield event.plain_result(f"Failed to flush traces: {e}")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """Handle all message events for tracing"""
        log_both("INFO", ">>> on_all_message() EVENT TRIGGERED")
        log_both("INFO", f"  self.enabled: {self.enabled}")
        log_both("INFO", f"  self.langfuse_client: {self.langfuse_client is not None}")
        log_both("INFO", f"  message_tracing enabled: {self.config.get('enabled_message_tracing', True)}")

        if not self.enabled or not self.config.get("enabled_message_tracing", True):
            log_both("INFO", "  SKIPPING - message tracing disabled")
            return

        try:
            # Get user and platform info
            user_id = event.unified_msg_origin
            platform = (
                event.get_platform_name()
                if hasattr(event, "get_platform_name")
                else "unknown"
            )

            log_both("INFO", f"  user_id: {user_id}")
            log_both("INFO", f"  platform: {platform}")

            session = self._get_or_create_session(user_id, platform)

            # Extract message content
            message_content = event.message_str or ""
            log_both("INFO", f"  message_content (first 100 chars): {message_content[:100]}")

            # Create trace for incoming message
            environment = self.config.get("environment", "production")

            log_both("INFO", "  Calling _trace_message()...")
            result = self._trace_message(
                session=session,
                name="user_message",
                input_data={
                    "content": message_content,
                    "user_id": user_id,
                    "platform": platform,
                },
                metadata={
                    "environment": environment,
                },
            )
            log_both("INFO", f"  _trace_message() returned: {result is not None}")

        except Exception as e:
            log_both("ERROR", f"Error tracing message: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Hook into LLM request for tracing"""
        log_both("INFO", ">>> on_llm_request() EVENT TRIGGERED")
        log_both("INFO", f"  self.enabled: {self.enabled}")
        log_both("INFO", f"  llm_tracing enabled: {self.config.get('enabled_llm_tracing', True)}")

        if not self.enabled or not self.config.get("enabled_llm_tracing", True):
            log_both("INFO", "  SKIPPING - LLM tracing disabled")
            return

        try:
            user_id = event.unified_msg_origin
            platform = (
                event.get_platform_name()
                if hasattr(event, "get_platform_name")
                else "unknown"
            )

            log_both("INFO", f"  user_id: {user_id}")
            log_both("INFO", f"  platform: {platform}")
            log_both("INFO", f"  prompt (first 100 chars): {(req.prompt or '')[:100]}")

            session = self._get_or_create_session(user_id, platform)

            environment = self.config.get("environment", "production")

            # Store the session trace_id for use in response hook
            if not hasattr(event, "_langfuse_trace_id"):
                event._langfuse_trace_id = session.trace_id

            log_both("INFO", "  Calling _trace_message() for llm_request...")
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
                metadata={
                    "environment": environment,
                    "session_trace_id": session.trace_id,
                },
            )

        except Exception as e:
            log_both("ERROR", f"Error tracing LLM request: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """Hook into LLM response for tracing"""
        log_both("INFO", ">>> on_llm_response() EVENT TRIGGERED")
        log_both("INFO", f"  self.enabled: {self.enabled}")
        log_both("INFO", f"  llm_tracing enabled: {self.config.get('enabled_llm_tracing', True)}")

        if not self.enabled or not self.config.get("enabled_llm_tracing", True):
            log_both("INFO", "  SKIPPING - LLM tracing disabled")
            return

        try:
            user_id = event.unified_msg_origin
            platform = (
                event.get_platform_name()
                if hasattr(event, "get_platform_name")
                else "unknown"
            )

            log_both("INFO", f"  user_id: {user_id}")
            log_both("INFO", f"  platform: {platform}")

            session = self._get_or_create_session(user_id, platform)

            environment = self.config.get("environment", "production")

            # Trace the LLM completion
            completion_text = ""
            if resp.completion_text:
                completion_text = resp.completion_text
            elif resp.result_chain:
                completion_text = str(resp.result_chain)

            model = resp.model if hasattr(resp, "model") else "unknown"

            log_both("INFO", f"  model: {model}")
            log_both("INFO", f"  completion_text (first 100 chars): {completion_text[:100]}")
            log_both("INFO", f"  resp.usage: {resp.usage if hasattr(resp, 'usage') else 'N/A'}")

            log_both("INFO", "  Calling _trace_llm_call()...")
            self._trace_llm_call(
                session=session,
                model=model,
                prompt="",
                completion=completion_text,
                prompt_tokens=resp.usage.prompt_tokens
                if resp.usage and hasattr(resp.usage, "prompt_tokens")
                else None,
                completion_tokens=resp.usage.completion_tokens
                if resp.usage and hasattr(resp.usage, "completion_tokens")
                else None,
                total_tokens=resp.usage.total_tokens
                if resp.usage and hasattr(resp.usage, "total_tokens")
                else None,
                metadata={
                    "environment": environment,
                },
            )

        except Exception as e:
            log_both("ERROR", f"Error tracing LLM response: {e}")
            log_both("ERROR", f"Traceback: {traceback.format_exc()}")
