import asyncio
import streamlit as st
from contextlib import AsyncExitStack
from pathlib import Path

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

st.set_page_config(
    page_title="Gemini MCP Client",
    layout="wide"
)


def convert_schema_for_gemini(schema: dict) -> dict:
    if schema is None:
        return {}
    
    result = {}
    
    for key, value in schema.items():
        if key == "type":
            if isinstance(value, list):
                for t in value:
                    if t != "null":
                        result["type"] = t.upper()
                        break
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


def get_server_params(server_script_path: str) -> StdioServerParameters:
    is_python = server_script_path.endswith(".py")
    is_js = server_script_path.endswith(".js")
    is_wsl_command = server_script_path.startswith("wsl:")

    if not (is_python or is_js or is_wsl_command):
        raise ValueError("Server script must be a .py, .js file, or wsl:command format")

    is_wsl_path = server_script_path.startswith("/")

    if is_wsl_command:
        parts = server_script_path.split(":")
        command_path = parts[1]
        args = parts[2:] if len(parts) > 2 else []
        return StdioServerParameters(
            command="wsl.exe",
            args=[command_path] + args,
            env=None,
        )
    elif is_python:
        if is_wsl_path:
            parent_dir = "/".join(server_script_path.rsplit("/", 1)[:-1])
            venv_python = f"{parent_dir}/.venv/bin/python"
            return StdioServerParameters(
                command="wsl",
                args=["--", venv_python, server_script_path],
                env=None,
            )
        else:
            path = Path(server_script_path).resolve()
            return StdioServerParameters(
                command="uv",
                args=["--directory", str(path.parent), "run", path.name],
                env=None,
            )
    else:
        if is_wsl_path:
            return StdioServerParameters(command="wsl", args=["--", "node", server_script_path], env=None)
        else:
            return StdioServerParameters(command="node", args=[server_script_path], env=None)


async def get_tools_from_server(server_path: str) -> list[dict]:
    server_params = get_server_params(server_path)
    
    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        response = await session.list_tools()
        tools_info = []
        for tool in response.tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            })
        return tools_info


def build_conversation_history(chat_messages: list) -> list:
    history = []
    for msg in chat_messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))
    return history


async def process_query_with_server(
    server_path: str, 
    api_key: str, 
    model: str, 
    query: str,
    chat_history: list = None,
    system_instruction: str = None,
    tools_cache: list = None
) -> str:
    server_params = get_server_params(server_path)
    client = genai.Client(api_key=api_key)
    
    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        if tools_cache:
            function_declarations = []
            for tool in tools_cache:
                gemini_schema = convert_schema_for_gemini(tool["inputSchema"])
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool["name"],
                        description=tool["description"],
                        parameters=gemini_schema,
                    )
                )
        else:
            response = await session.list_tools()
            function_declarations = []
            for tool in response.tools:
                gemini_schema = convert_schema_for_gemini(tool.inputSchema)
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=gemini_schema,
                    )
                )
        
        gemini_tools = [types.Tool(function_declarations=function_declarations)]
        
        messages = []
        if chat_history:
            messages.extend(build_conversation_history(chat_history))
        messages.append(types.Content(role="user", parts=[types.Part(text=query)]))
        
        config = types.GenerateContentConfig(tools=gemini_tools)
        if system_instruction:
            config = types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=system_instruction
            )
        
        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )

        final_text = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                final_text.append(part.text)
            elif hasattr(part, "function_call"):
                tool_name = part.function_call.name
                tool_args = dict(part.function_call.args)

                result = await session.call_tool(tool_name, tool_args)
                final_text.append(f"**Tool Called:** `{tool_name}`\n**Args:** `{tool_args}`")

                result_text = ""
                for content in result.content:
                    if hasattr(content, "text"):
                        result_text = content.text
                        break

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

                response = client.models.generate_content(
                    model=model,
                    contents=messages,
                    config=config,
                )
                
                if response.candidates[0].content.parts:
                    for p in response.candidates[0].content.parts:
                        if hasattr(p, "text") and p.text:
                            final_text.append(p.text)

        return "\n\n".join(final_text)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "tools" not in st.session_state:
        st.session_state.tools = []
    if "tools_cache" not in st.session_state:
        st.session_state.tools_cache = []
    if "server_path" not in st.session_state:
        st.session_state.server_path = ""
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model" not in st.session_state:
        st.session_state.model = "gemini-2.5-flash"
    if "system_instruction" not in st.session_state:
        st.session_state.system_instruction = ""
    if "context_window_enabled" not in st.session_state:
        st.session_state.context_window_enabled = True
    if "max_context_messages" not in st.session_state:
        st.session_state.max_context_messages = 20


def main():
    init_session_state()
    
    st.title("Gemini MCP Client")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Configuration")
        
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="Enter your Google API key...",
            help="Get your API key from Google AI Studio"
        )
        
        model_index = GEMINI_MODELS.index(st.session_state.model) if st.session_state.model in GEMINI_MODELS else 1
        selected_model = st.selectbox(
            "Select Model",
            options=GEMINI_MODELS,
            index=model_index,
            help="Choose a Gemini model for your queries"
        )
        
        st.markdown("---")
        
        st.header("MCP Server")
        
        server_path = st.text_input(
            "Server Script Path",
            value=st.session_state.server_path,
            placeholder="path/to/server.py",
            help="Path to your MCP server script (.py or .js)"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            connect_btn = st.button("Connect", use_container_width=True)
        
        with col2:
            disconnect_btn = st.button("Disconnect", use_container_width=True)
        
        if st.session_state.connected:
            st.success("Connected to server")
            if st.session_state.tools:
                st.markdown("**Available Tools:**")
                for tool in st.session_state.tools:
                    tool_name = tool["name"] if isinstance(tool, dict) else tool
                    st.markdown(f"- `{tool_name}`")
        else:
            st.warning("Not connected")
        
        if connect_btn:
            if not api_key:
                st.error("Please enter your API key!")
            elif not server_path:
                st.error("Please enter the server script path!")
            else:
                with st.spinner("Connecting to server..."):
                    try:
                        tools = asyncio.run(get_tools_from_server(server_path))
                        st.session_state.connected = True
                        st.session_state.tools = tools
                        st.session_state.api_key = api_key
                        st.session_state.model = selected_model
                        st.session_state.server_path = server_path
                        st.session_state.tools_cache = tools
                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")
        
        if disconnect_btn and st.session_state.connected:
            st.session_state.connected = False
            st.session_state.tools = []
            st.session_state.tools_cache = []
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        st.header("Context Settings")
        
        context_enabled = st.checkbox(
            "Enable Context Window",
            value=st.session_state.context_window_enabled,
            help="Keep conversation history for context-aware responses"
        )
        st.session_state.context_window_enabled = context_enabled
        
        if context_enabled:
            max_messages = st.slider(
                "Max Context Messages",
                min_value=2,
                max_value=50,
                value=st.session_state.max_context_messages,
                help="Number of previous messages to include"
            )
            st.session_state.max_context_messages = max_messages
        
        st.markdown("---")
        
        st.header("System Instruction")
        system_instruction = st.text_area(
            "System Prompt (cached)",
            value=st.session_state.system_instruction,
            placeholder="Enter system instructions for the model...",
            help="This instruction is cached and sent with every request",
            height=100
        )
        st.session_state.system_instruction = system_instruction
        
        st.markdown("---")
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info(f"**Current Model:** {selected_model}")
        
        if st.session_state.context_window_enabled and st.session_state.chat_history:
            st.markdown(f"**Context:** {len(st.session_state.chat_history)} messages")
    
    if not st.session_state.connected:
        st.info("Please configure your API key, select a model, and connect to an MCP server from the sidebar.")
        
        with st.expander("How to use"):
            st.markdown("""
            1. **Enter your Google API Key** - Get one from [Google AI Studio](https://aistudio.google.com/apikey)
            2. **Select a Gemini Model** - Choose based on your needs:
               - `gemini-2.5-pro` - Most capable, best for complex tasks
               - `gemini-2.5-flash` - Fast and efficient (recommended)
               - `gemini-2.0-flash` - Previous generation, still powerful
               - `gemini-1.5-pro/flash` - Stable, production-ready
            3. **Enter MCP Server Path** - Path to your MCP server script
            4. **Click Connect** - Establish connection to the server
            5. **Start Chatting!** - Ask questions and use available tools
            """)
    else:
        if api_key and api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        if selected_model != st.session_state.model:
            st.session_state.model = selected_model
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_history = None
                        if st.session_state.context_window_enabled and st.session_state.chat_history:
                            max_msgs = st.session_state.max_context_messages
                            chat_history = st.session_state.chat_history[-max_msgs:]
                        
                        system_instruction = st.session_state.system_instruction if st.session_state.system_instruction else None
                        
                        response = asyncio.run(process_query_with_server(
                            st.session_state.server_path,
                            st.session_state.api_key,
                            st.session_state.model,
                            prompt,
                            chat_history=chat_history,
                            system_instruction=system_instruction,
                            tools_cache=st.session_state.tools_cache
                        ))
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.session_state.chat_history.append({"role": "user", "content": prompt})
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
