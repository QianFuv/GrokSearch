# Grok Search MCP

Minimal MCP server for Grok-based web search and Tavily-backed fetch/map tools.

## What It Does

- `web_search`: Ask Grok to search the web and return the answer body.
- `get_sources`: Read the cached source list from a previous `web_search` call.
- `web_fetch`: Fetch a page through Tavily and return Markdown content.
- `web_map`: Map a site through Tavily and return discovered URLs.
- `get_config_info`: Show effective configuration and test the Grok connection.
- `switch_model`: Validate and persist the default Grok model.
- `describe_url`: Ask Grok to read one page and return a title plus extracts.
- `rank_sources`: Ask Grok to reorder numbered sources by relevance.

## Requirements

- Python 3.12+
- `uv`
- An OpenAI-compatible Grok endpoint

Tavily is optional. Without Tavily, `web_fetch` and `web_map` will not work.

## Install

### Claude Code

Configure Grok directly. Add Tavily if you want `web_fetch` and `web_map`:

```bash
claude mcp add-json grok-search --scope user '{
  "type": "stdio",
  "command": "uvx",
  "args": [
    "--from",
    "git+https://github.com/GuDaStudio/GrokSearch",
    "grok-search"
  ],
  "env": {
    "GROK_API_URL": "https://your-api-endpoint.com/v1",
    "GROK_API_KEY": "your-grok-api-key",
    "TAVILY_API_URL": "https://api.tavily.com",
    "TAVILY_API_KEY": "your-tavily-api-key"
  }
}'
```

## Local Development

```bash
uv sync --dev
uv run grok-search
```

The server runs over stdio.

## Configuration

### Core Variables

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `GROK_API_URL` | Yes | - | Must be OpenAI-compatible. |
| `GROK_API_KEY` | Yes | - | Used for Grok requests. |
| `GROK_MODEL` | No | `grok-4.20-beta` | Default model. |
| `TAVILY_API_URL` | No | `https://api.tavily.com` | Used by `web_fetch` and `web_map`. Tavily Hikari `/mcp` URLs are normalized to the same host's `/api/tavily` HTTP facade. |
| `TAVILY_API_KEY` | No | - | Required only for Tavily-backed tools. |

### Optional Variables

| Variable | Default |
| --- | --- |
| `TAVILY_ENABLED` | `true` |
| `GROK_DEBUG` | `false` |
| `GROK_LOG_LEVEL` | `INFO` |
| `GROK_LOG_DIR` | `logs` |
| `GROK_RETRY_MAX_ATTEMPTS` | `3` |
| `GROK_RETRY_MULTIPLIER` | `1` |
| `GROK_RETRY_MAX_WAIT` | `10` |

## Tool Summary

### `web_search`

Input:

- `query`
- `platform` (optional)
- `model` (optional)
- `extra_sources` (optional)

Output:

- `session_id`
- `content`
- `sources_count`

### `get_sources`

Input:

- `session_id`

Output:

- `session_id`
- `sources_count`
- `sources`

### `web_fetch`

Input:

- `url`

Output:

- Markdown page content

### `web_map`

Input:

- `url`
- `instructions` (optional)
- `max_depth` (optional)
- `max_breadth` (optional)
- `limit` (optional)
- `timeout` (optional)

Output:

- JSON string with discovered URLs

### `get_config_info`

Output:

- JSON string with effective config and connection test results

### `switch_model`

Input:

- `model`

Output:

- JSON string with validation and persistence result

### `describe_url`

Input:

- `url`
- `model` (optional)

Output:

- `url`
- `title`
- `extracts`

### `rank_sources`

Input:

- `query`
- `sources_text`
- `total`
- `model` (optional)

Output:

- `query`
- `order`
- `total`

## Notes

- `switch_model` stores the selected model in `~/.config/grok-search/config.json`.
- `web_search` caches sources in memory and returns them later through `get_sources`.
- Time-sensitive queries may include local time context automatically.

## License

[MIT](LICENSE)
