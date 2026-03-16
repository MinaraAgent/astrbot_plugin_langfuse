# Changelog

All notable changes to the AstrBot Langfuse Plugin will be documented in this file.

## [1.0.1] - 2025-03-16

### Fixed
- Fixed error: `start_observation()` doesn't accept `session_id` and `user_id` as direct parameters
- Session and user info now passed via metadata (compatible with Langfuse SDK v3)

## [1.0.0] - 2025-03-16

### Added
- Initial release of Langfuse integration for AstrBot
- LLM request/response tracing with model name and token usage
- Message event tracing for all incoming messages
- Session management with configurable timeout
- Support for `session_id` and `user_id` in traces for user tracking
- Environment tagging (production, development, etc.)
- Debug logging to `/tmp/astrbot_langfuse_debug.log`
- Commands:
  - `/langfuse_status` - Check connection status and active sessions
  - `/langfuse_flush` - Manually flush traces to Langfuse

### Configuration
- `secret_key` - Langfuse Secret Key
- `public_key` - Langfuse Public Key
- `base_url` - Langfuse API URL (default: https://cloud.langfuse.com)
- `enabled` - Enable/disable plugin (default: true)
- `enabled_llm_tracing` - Enable LLM call tracing (default: true)
- `enabled_message_tracing` - Enable message event tracing (default: true)
- `session_timeout` - Session timeout in seconds (default: 3600)
- `environment` - Environment name for traces (default: production)

### Features
- Links LLM request and response observations together
- Captures token usage (input, output, total)
- Captures model name from requests and responses
- Platform metadata for multi-platform deployments
- Automatic session cleanup for expired sessions
- Lazy import of langfuse package for better compatibility
