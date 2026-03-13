"""
AstrBot Langfuse Integration Plugin

This plugin integrates Langfuse LLM observability with AstrBot,
enabling tracing of LLM calls and message events.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star

try:
    from langfuse import Langfuse
    from langfuse.api.resources.commons.errors.errors import BaseError as LangfuseError

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    LangfuseError = Exception


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
        super().__init__(context)
        self.langfuse_client: Optional[Langfuse] = None
        self.enabled = False
        self.config = {}
        self.sessions: dict[str, SessionInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def _get_config(self) -> dict:
        """Get plugin configuration from AstrBot config"""
        config = self.config
        if hasattr(self.context, "get_config"):
            config = self.context.get_config() or {}
        return config

    async def _init_langfuse(self) -> bool:
        """Initialize Langfuse client"""
        if not LANGFUSE_AVAILABLE:
            logger.error(
                "[Langfuse] langfuse package not installed. Run: pip install langfuse"
            )
            return False

        config = await self._get_config()

        secret_key = config.get("secret_key", "")
        public_key = config.get("public_key", "")
        base_url = config.get("base_url", "https://cloud.langfuse.com")

        if not secret_key or not public_key:
            logger.warning(
                "[Langfuse] Secret key or public key not configured. "
                "Please set them in plugin config."
            )
            return False

        try:
            self.langfuse_client = Langfuse(
                secret_key=secret_key,
                public_key=public_key,
                host=base_url,
                enabled=config.get("enabled", True),
            )
            self.enabled = True
            logger.info(
                f"[Langfuse] Client initialized successfully. Base URL: {base_url}"
            )
            return True
        except Exception as e:
            logger.error(f"[Langfuse] Failed to initialize client: {e}")
            return False

    def _get_or_create_session(self, user_id: str, platform: str) -> SessionInfo:
        """Get or create a session for a user"""
        session_key = f"{platform}:{user_id}"
        current_time = time.time()
        session_timeout = self.config.get("session_timeout", 3600)

        if session_key in self.sessions:
            session = self.sessions[session_key]
            # Check if session has expired
            if current_time - session.last_activity > session_timeout:
                # Create new session
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
                    logger.debug(
                        f"[Langfuse] Cleaned up {len(expired_keys)} expired sessions"
                    )

                # Flush Langfuse client periodically
                if self.langfuse_client:
                    self.langfuse_client.flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Langfuse] Error in session cleanup: {e}")

    def _trace_message(
        self,
        session: SessionInfo,
        name: str,
        input_data: dict,
        output_data: Optional[dict] = None,
        metadata: Optional[dict] = None,
        level: str = "DEFAULT",
    ):
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

            return trace
        except LangfuseError as e:
            logger.error(f"[Langfuse] Error creating trace: {e}")
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
        if not self.enabled or not self.langfuse_client:
            return None

        try:
            trace = self.langfuse_client.trace(
                id=session.trace_id,
                name="llm_call",
                session_id=session.session_id,
                metadata=metadata or {},
            )

            generation = trace.generation(
                name=f"llm_{model}",
                model=model,
                input=prompt,
                output=completion,
                usage={
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens,
                }
                if total_tokens
                else None,
            )

            return generation
        except LangfuseError as e:
            logger.error(f"[Langfuse] Error tracing LLM call: {e}")
            return None

    async def init(self, config: dict):
        """Initialize the plugin"""
        self.config = config

        if not config.get("enabled", True):
            logger.info("[Langfuse] Plugin is disabled in config")
            return

        success = await self._init_langfuse()
        if success:
            # Start session cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
            logger.info("[Langfuse] Plugin initialized successfully")

    async def terminate(self):
        """Cleanup when plugin is stopped"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
                logger.info("[Langfuse] Flushed remaining traces")
            except Exception as e:
                logger.error(f"[Langfuse] Error flushing traces: {e}")

        logger.info("[Langfuse] Plugin terminated")

    @filter.command("langfuse_status")
    async def langfuse_status(self, event: AstrMessageEvent):
        """Check Langfuse connection status"""
        if not LANGFUSE_AVAILABLE:
            yield event.plain_result(
                "Langfuse package is not installed. Run: pip install langfuse"
            )
            return

        if not self.enabled:
            yield event.plain_result(
                "Langfuse is not enabled or not configured properly."
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
            f"- Message Tracing: {'Enabled' if config.get('enabled_message_tracing', True) else 'Disabled'}"
        )
        yield event.plain_result(status_msg)

    @filter.command("langfuse_flush")
    async def langfuse_flush(self, event: AstrMessageEvent):
        """Manually flush Langfuse traces"""
        if not self.enabled or not self.langfuse_client:
            yield event.plain_result("Langfuse is not enabled.")
            return

        try:
            self.langfuse_client.flush()
            yield event.plain_result("Langfuse traces flushed successfully.")
        except Exception as e:
            yield event.plain_result(f"Failed to flush traces: {e}")

    @filter.on_message_event()
    async def on_message(self, event: AstrMessageEvent):
        """Handle all message events for tracing"""
        if not self.enabled or not self.config.get("enabled_message_tracing", True):
            return

        try:
            # Get user and platform info
            user_id = event.unified_msg_origin
            platform = (
                event.get_platform_name()
                if hasattr(event, "get_platform_name")
                else "unknown"
            )
            session = self._get_or_create_session(user_id, platform)

            # Extract message content
            message_content = ""
            if hasattr(event, "message_obj") and event.message_obj:
                message_content = (
                    str(event.message_obj.message)
                    if hasattr(event.message_obj, "message")
                    else ""
                )

            # Create trace for incoming message
            environment = self.config.get("environment", "production")
            self._trace_message(
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

        except Exception as e:
            logger.error(f"[Langfuse] Error tracing message: {e}")
