# Gemini MCP Client

A Streamlit-based UI client for interacting with MCP (Model Context Protocol) servers using Google Gemini models.

## Features

- **Multiple Gemini Models**: Support for various Gemini models including:
  - gemini-2.5-pro
  - gemini-2.5-flash
  - gemini-2.0-flash
  - gemini-1.5-pro/flash
  
- **MCP Server Integration**: Connect to any MCP server (.py or .js)

- **Context Window**: Maintains conversation history for context-aware responses
  - Configurable message limit (2-50 messages)
  - Toggle on/off as needed

- **System Instructions**: Set custom system prompts for the model

- **Tool Caching**: Caches MCP tool definitions for better performance


## Usage
First of all, complete the necessary installation from this site to run the MCP client: https://modelcontextprotocol.io/docs/develop/build-client

### Streamlit UI (Recommended)

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### CLI Mode

```bash
uv run python client.py path/to/your/mcp-server.py
```

## Configuration

### In the Streamlit UI:

1. **Google API Key**: Enter your API key.
2. **Model Selection**: Choose a Gemini model.
3. **MCP Server Path**: Path to your MCP server script.
4. **Context Settings**: Enable/disable context window and set message limit.
5. **System Instruction**: Optional system prompt for the model.

## Dependencies

- `streamlit` - Web UI framework
- `google-genai` - Google Generative AI SDK
- `mcp` - Model Context Protocol client
- `python-dotenv` - Environment variable management
