import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables from .env file in the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# Google model constant
GOOGLE_MODEL = "gemini-2.5-flash"


def convert_schema_for_gemini(schema: dict) -> dict:
    """Convert MCP tool schema to Gemini-compatible format"""
    if schema is None:
        return {}
    
    result = {}
    
    for key, value in schema.items():
        if key == "type":
            # Handle type arrays like ['string', 'null'] -> 'STRING'
            if isinstance(value, list):
                # Take the first non-null type
                for t in value:
                    if t != "null":
                        result["type"] = t.upper()
                        break
                # Mark as nullable if 'null' was in the list
                if "null" in value:
                    result["nullable"] = True
            else:
                result["type"] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            result["properties"] = {
                k: convert_schema_for_gemini(v) for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result["items"] = convert_schema_for_gemini(value)
        elif key in ["required", "description", "enum"]:
            result[key] = value
    
    return result


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self._client = None

    @property
    def client(self):
        """Lazy-initialize Google Generative AI client when needed"""
        if self._client is None:
            self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        return self._client

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
                               For WSL paths, use format: /home/user/path/to/server.py
                               For WSL commands, use format: wsl:/path/to/command:arg1:arg2
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        is_wsl_command = server_script_path.startswith("wsl:")

        if not (is_python or is_js or is_wsl_command):
            raise ValueError("Server script must be a .py, .js file, or wsl:command format")

        # Check if this is a WSL/Linux path
        is_wsl_path = server_script_path.startswith("/")

        if is_wsl_command:
            # Parse wsl:command:arg1:arg2 format
            parts = server_script_path.split(":")
            command_path = parts[1]
            args = parts[2:] if len(parts) > 2 else []
            server_params = StdioServerParameters(
                command="wsl.exe",
                args=[command_path] + args,
                env=None,
            )
        elif is_python:
            if is_wsl_path:
                # Run server in WSL using the project's venv
                parent_dir = "/".join(server_script_path.rsplit("/", 1)[:-1])
                venv_python = f"{parent_dir}/.venv/bin/python"
                server_params = StdioServerParameters(
                    command="wsl",
                    args=["--", venv_python, server_script_path],
                    env=None,
                )
            else:
                path = Path(server_script_path).resolve()
                server_params = StdioServerParameters(
                    command="uv",
                    args=["--directory", str(path.parent), "run", path.name],
                    env=None,
                )
        else:
            if is_wsl_path:
                server_params = StdioServerParameters(command="wsl", args=["--", "node", server_script_path], env=None)
            else:
                server_params = StdioServerParameters(command="node", args=[server_script_path], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Google Gemini and available tools"""
        response = await self.session.list_tools()
        
        # Convert MCP tools to Gemini function declarations
        function_declarations = []
        for tool in response.tools:
            # Convert schema to Gemini-compatible format
            gemini_schema = convert_schema_for_gemini(tool.inputSchema)
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=gemini_schema,
                )
            )
        
        gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Initial Google Gemini API call
        messages = [types.Content(role="user", parts=[types.Part(text=query)])]
        
        response = self.client.models.generate_content(
            model=GOOGLE_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(tools=gemini_tools),
        )

        # Process response and handle tool calls
        final_text = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                final_text.append(part.text)
            elif hasattr(part, "function_call"):
                tool_name = part.function_call.name
                tool_args = dict(part.function_call.args)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]") #return tool result

                # Get text content from result
                result_text = ""
                for content in result.content:
                    if hasattr(content, "text"):
                        result_text = content.text
                        break

                # Continue conversation with tool results
                messages.append(response.candidates[0].content)
                messages.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": result_text}
                        ))]
                    )
                )

                # Get next response from Gemini
                response = self.client.models.generate_content(
                    model=GOOGLE_MODEL,
                    contents=messages,
                    config=types.GenerateContentConfig(tools=gemini_tools),
                )
                
                if response.candidates[0].content.parts:
                    for p in response.candidates[0].content.parts:
                        if hasattr(p, "text") and p.text:
                            final_text.append(p.text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])

        # Check if we have a valid API key to continue
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("\nNo GOOGLE_API_KEY found. To query these tools with Gemini, set your API key:")
            print("  export GOOGLE_API_KEY=your-api-key-here")
            return

        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
