# AstrBot Langfuse Plugin

A Langfuse integration plugin for AstrBot that enables LLM observability and tracing.

## Features

- Trace LLM calls with detailed token usage
- Track message events across sessions
- Configurable session timeout
- Support for self-hosted Langfuse instances
- Easy configuration through AstrBot's plugin settings

## Installation

### From GitHub

1. In AstrBot, go to Plugin Manager
2. Click "Install from URL"
3. Enter: `https://github.com/minaraagent/astrbot_plugin_langfuse`
4. Click Install

### Manual Installation

1. Clone this repository to your AstrBot plugins directory:
   ```bash
   cd /path/to/astrbot/addons/
   git clone https://github.com/minaraagent/astrbot_plugin_langfuse.git
   ```

2. Install dependencies:
   ```bash
   pip install -r astrbot_plugin_langfuse/requirements.txt
   ```

3. Restart AstrBot

## Configuration

Configure the plugin through AstrBot's plugin settings or directly in the config:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable Langfuse tracing |
| `secret_key` | string | `""` | Your Langfuse Secret Key |
| `public_key` | string | `""` | Your Langfuse Public Key |
| `base_url` | string | `"https://cloud.langfuse.com"` | Langfuse API base URL |
| `enabled_llm_tracing` | bool | `true` | Enable LLM call tracing |
| `enabled_message_tracing` | bool | `true` | Enable message event tracing |
| `session_timeout` | int | `3600` | Session timeout in seconds |
| `environment` | string | `"production"` | Environment name for traces |

### Getting Langfuse API Keys

1. Sign up at [Langfuse Cloud](https://cloud.langfuse.com) or self-host
2. Go to Project Settings > API Keys
3. Create a new API key pair
4. Copy the Secret Key and Public Key to the plugin configuration

## Commands

| Command | Description |
|---------|-------------|
| `/langfuse_status` | Check Langfuse connection status |
| `/langfuse_flush` | Manually flush pending traces |

## Usage

Once configured, the plugin automatically:

1. Creates traces for incoming user messages
2. Tracks LLM calls with token usage
3. Groups messages into sessions based on user activity
4. Periodically flushes traces to Langfuse

## Self-Hosted Langfuse

To use a self-hosted Langfuse instance, set `base_url` to your instance URL:

```yaml
base_url: "http://localhost:3000"
```

## Requirements

- AstrBot >= 3.4.0
- Python >= 3.8
- langfuse >= 2.0.0

## License

MIT License
