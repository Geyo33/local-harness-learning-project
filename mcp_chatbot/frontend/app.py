import gradio as gr
from mcp_chatbot.core.session import ChatSession
from mcp_chatbot.core.server import Server
from mcp_chatbot.core.llm_client import LLMClient
from mcp_chatbot.core.config import Configuration
from schemas import UserContext
from pathlib import Path
import logging
from mcp_chatbot.frontend.multimodal import build_content_array, IMAGE_EXTENSIONS
from mcp_chatbot.settings import load_settings, save_settings
from mcp_chatbot.tools.file_manager import FileManager


class GradioChat:
    def __init__(self) -> None:
        self.session: ChatSession = None

chat_session = GradioChat()

async def setup_and_initialize():
    """Handles all configuration and startup logic."""
    logging.info("--- Starting MCP Client Setup ---")
    try:
        config = Configuration()
        config_path = "mcp_chatbot/servers_config.json"

        if not Path(config_path).exists():
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            Path(config_path).write_text('{}')

        server_config = config.load_config(config_path)
        logging.info(f"server config: {server_config}")
        
        servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()] if server_config else []
        llm_client = LLMClient(config.llm_api_key, config.model)
        
        chat_session_initial = ChatSession(servers, llm_client)


        await chat_session_initial.list_servers()
        await chat_session_initial.initialize_servers()
        await chat_session_initial.build_system_message()

        chat_session.session = chat_session_initial

        logging.info("--- MCP Client Initialized Successfully ---")

    except Exception as e:
        logging.info(f"FATAL SETUP ERROR: {e}")
    
async def servers_state():
    if chat_session.session:
        servers_name_list = [server.name for server in chat_session.session.servers]
    else:
        servers_name_list = []

    return gr.CheckboxGroup(servers_name_list, label="activate all", show_select_all=True, container=True, interactive=True, scale=5)

async def update_servers(selected_servers: list):
    logging.info("\n--- Initiating servers update ---")
    # selected_servers is list like ['server1', 'server3']
    if not selected_servers:
        logging.info("\n--- No server selected ---")
        await chat_session.session.cleanup_servers()
        await chat_session.session.list_servers()
        await chat_session.session.initialize_servers()
        await chat_session.session.build_system_message()
        return

    await chat_session.session.cleanup_servers()
    for server in selected_servers:
        chat_session.session.active_servers[server] = True

    logging.info(f"Updating with: {selected_servers}")

    await chat_session.session.initialize_servers()
    await chat_session.session.build_system_message()
    logging.info("\n--- Servers updated ---")


async def shutdown_client():
    """Called when Gradio is closing to ensure clean exit."""
    if chat_session.session:
        logging.info("\n--- Initiating Graceful Shutdown ---")
        await chat_session.session.cleanup_servers()
        chat_session.session = None

async def save_planner_mode(value: str):
    current = load_settings()
    current["planner_mode"] = value
    save_settings(current)
    if chat_session.session:
        chat_session.session.planner_mode = value
    return f"Planner mode set to: {value}"


async def save_file_root(path: str):
    path = path.strip()
    if not path:
        return "Error: path is empty."
    p = Path(path)
    if not p.exists():
        return f"Error: path does not exist: {path}"
    if not p.is_dir():
        return f"Error: path is not a directory: {path}"
    if chat_session.session is None:
        return "Error: session not ready. Wait for the app to finish loading."
    current = load_settings()
    current["file_root"] = path
    save_settings(current)
    try:
        chat_session.session.file_manager = FileManager(p)
        await chat_session.session.build_system_message()
        return f"Saved. File tools active at: `{path}`"
    except Exception as e:
        return f"Saved to disk, but error activating FileManager: {e}"

# --- Logic Functions ---

def _render_task_html() -> str:
    if not chat_session.session or not chat_session.session.task_manager:
        return ""
    block = chat_session.session.task_manager.render_block()
    if not block:
        return ""
    lines = block.strip().splitlines()
    rows = []
    for line in lines:
        if line.startswith("<tasks>") or line.startswith("</tasks>"):
            continue
        if line.startswith("[ ]"):
            icon, text = "⬜", line[3:].strip()
        elif line.startswith("[~]"):
            icon, text = "🔄", line[3:].strip()
        elif line.startswith("[x]"):
            icon, text = "✅", line[3:].strip()
        elif line.startswith("[-]"):
            icon, text = "❌", line[3:].strip()
        else:
            continue
        rows.append(f'<div style="padding: 2px 0; font-size: 13px;">{icon} {text}</div>')
    if not rows:
        return ""
    inner = "\n".join(rows)
    return f'<div style="padding: 6px 4px;">{inner}</div>'


async def update_task_panel():
    return gr.update(value=_render_task_html())

def update_usage():
    breakdown = chat_session.session.token_usage.get("breakdown", {})
    max_tokens = chat_session.session.token_usage["context_size"]

    def fmt(n):
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    if not breakdown or not max_tokens:
        html = '<div style="padding:6px 4px;color:#4a5568;font-size:12px;text-align:center;">Context: —</div>'
        return gr.update(value=html)

    system = breakdown.get("system", 0)
    tools = breakdown.get("tools", 0)
    history = breakdown.get("history", 0)
    total = breakdown.get("total", 0)
    pct = breakdown.get("pct", 0.0)

    sys_pct = system / max_tokens * 100
    tools_pct = tools / max_tokens * 100
    hist_pct = history / max_tokens * 100

    bar_color = "#e53e3e" if pct >= 0.85 else ("#d69e2e" if pct >= 0.70 else "#13b0a8")

    html = f"""
    <div style="padding: 6px 4px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="color: #e2e8f0; font-size: 12px; white-space: nowrap; font-weight: 500;">Context</span>
            <div style="flex: 1; background-color: #1a1f2e; border-radius: 3px; overflow: hidden; border: 1px solid #2d3748; height: 16px; display: flex;">
                <div style="width: {sys_pct:.1f}%; background: #4a6fa5; height: 100%;" title="System: {fmt(system)}"></div>
                <div style="width: {tools_pct:.1f}%; background: #805ad5; height: 100%;" title="Tools: {fmt(tools)}"></div>
                <div style="width: {hist_pct:.1f}%; background: {bar_color}; height: 100%;" title="History: {fmt(history)}"></div>
            </div>
            <span style="color: #a0aec0; font-size: 11px; white-space: nowrap;">{round(pct * 100)}% · {fmt(total)}/{fmt(max_tokens)}</span>
        </div>
        <div style="display: flex; gap: 10px; font-size: 10px; color: #718096; padding-left: 2px;">
            <span><span style="color:#4a6fa5;">■</span> sys {fmt(system)}</span>
            <span><span style="color:#805ad5;">■</span> tools {fmt(tools)}</span>
            <span><span style="color:{bar_color};">■</span> hist {fmt(history)}</span>
            <span style="color:#4a5568;">free {fmt(max_tokens - total)}</span>
        </div>
    </div>
    """
    return gr.update(value=html)

async def handle_chat(input: dict[str, str | list], history: list, tool_validation_box, tool_allow_btn, tool_deny_btn):
    message, files = input.values()
    files = files or []

    if not message.strip() and not files:
        yield "", history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

    # Build OpenAI content array (or plain string when no files)
    content = build_content_array(message, files)

    # Build display history:
    #   - Text message with text-file name indicators in one bubble
    #   - Each image as a separate bubble (Gradio renders the file path as thumbnail)
    new_history = list(history)

    display_text = message
    for f in files:
        if Path(f).suffix.lower() not in IMAGE_EXTENSIONS:
            display_text += f"\n📄 {Path(f).name}"

    if display_text.strip():
        new_history.append({"role": "user", "content": display_text})

    for f in files:
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
            new_history.append({"role": "user", "content": {"path": f}})#(f,)})

    yield gr.update(value="", interactive=False), new_history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    try:
        new_history = new_history + [{"role": "assistant", "content": ""}]
        full_response = ""

        async for chunk in chat_session.session.handle_user_input(content):

            # ── Tool approval request ──────────────────────────────────────────
            # The backend paused and is waiting for user input; the chunk is the
            # pending tool-call JSON being streamed before the o|o sentinel.
            if chat_session.session.tool_call_detected:
                name, arguments = chunk
                new_history.append({
                    "role": "assistant",
                    "content": arguments,
                    "metadata": {"title": f"🛠️ Attempting to call: {name} with arguments: ", "status": "pending"},
                })
                
                yield gr.update(interactive=True), new_history, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.skip(), gr.skip()
                continue

            # ── Structured event chunk (tool result or denial) ─────────────────
            if "o|o" in chunk:
                parts = chunk.split("o|o", 3)
                name, args, result, is_safe = parts[0], parts[1], parts[2], parts[3]

                if name == "Tool call denied":
                    # Denial: show a cancelled tool block, no further streaming.
                    new_history[-1] = {
                        "role": "assistant",
                        "content": f"**Result:**\n{result}",
                        "metadata": {"title": "🚫 Tool call denied by user", "status": "done"},
                    }
                    full_response = ""
                    new_history.append({"role": "assistant", "content": full_response})
                else:
                    # Successful tool execution: render a collapsible tool block.
                    if is_safe == "True":
                        new_history.append({"role": "assistant", "content": ""})
                    new_history[-1] = {
                        "role": "assistant",
                        "content": f"**Arguments:**\n```json\n{args}\n```\n\n**Result:**\n{result}",
                        "metadata": {"title": f"🛠️ Tool used: {name}", "status": "done"},
                    }
                    # Prepare a fresh assistant bubble for the final answer.
                    full_response = ""
                    new_history.append({"role": "assistant", "content": full_response})
                if is_safe == "True":
                    yield gr.update(interactive=False), new_history, gr.skip(), gr.skip(), gr.skip(), gr.update(value=_render_task_html()), update_usage()
                else:
                    yield gr.update(interactive=False), new_history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=_render_task_html()), update_usage()
                continue

            # ── Plain text chunk (final streamed answer) ───────────────────────
            full_response += chunk
            new_history[-1] = {"role": "assistant", "content": full_response}
            yield gr.skip(), new_history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    except Exception as e:
        error_msg = f"Error during chat processing: {str(e)}"
        logging.info(error_msg)
        new_history = history + [{"role": "assistant", "content": error_msg}]
        yield "", new_history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.skip(), gr.skip()
    finally:
        yield gr.update(interactive=True), new_history, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

css = """

#col { height: calc(100vh - 112px - 16px); }
#col2 { height: calc(100vh - 112px - 67px); }
h1 {
    text-align: center;
}
#centertext {
    text-align: center;
}
"""

# --- Main App ---

def build_app():
    with gr.Blocks() as demo:
        
        # Not used yet
        user_context_state = gr.State(UserContext(**{
            "user_id": "",
            "history": [],
            "mcp_servers": []
        }))
        
        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row(equal_height=True):

                    with gr.Column(scale=5, elem_id="col"):
                        chatbot = gr.Chatbot(label="", show_label=False)
                        
                        tool_validation_box = gr.Markdown(max_height=50, value="# --- Allow tool execution ? ---", container=True, padding=True, visible=False)
                        with gr.Row(max_height=36):
                            tool_allow_btn = gr.Button("Allow", variant="primary", size="lg", scale=1, visible=False)
                            tool_deny_btn = gr.Button("Deny", variant="stop", size="lg", scale=1, visible=False)
                        
                        with gr.Row(max_height=80):
                            msg_input = gr.MultimodalTextbox(
                                sources=["upload","microphone"],
                                max_lines=2,
                                autoscroll=False,
                                stop_btn=True,
                                show_label=False,
                                placeholder="...",
                                container=True,
                            )
                        
                    
                    with gr.Column(scale=1, variant="panel", min_width=100, elem_id="col2"):
                        with gr.Row(scale=1):
                            with gr.Column(variant="compact"):
                                with gr.Accordion(label="Tasks", open=True):
                                    task_list_display = gr.HTML(value="", label="Tasks", container=True, min_height=200, show_label=False)
                                auto_tool = gr.Checkbox(label="Auto-tool", container=True) 
                                
                                with gr.Accordion(label="MCP servers", open=True):   
                                    refresh_servers = gr.Button("🗘", variant="secondary", size="lg")     
                                    mcp_list = gr.CheckboxGroup(["          ","          ","          "], label="activate all", show_select_all=True, container=True)
                                    with gr.Row():
                                        save_servers = gr.Button("Save", variant="primary", size="sm", min_width=20)
                                        btn_stop = gr.Button("Stop", variant="stop", size="sm", min_width=20)

                with gr.Row(max_height=40):
                    usage_bar = gr.HTML()
                with gr.Row():
                    gr.Markdown("---")

                            

            with gr.Tab("Settings"):
                gr.Markdown("### File Access")
                file_root_input = gr.Textbox(
                    label="Base directory",
                    placeholder="Paste or type an absolute folder path",
                    value=load_settings().get("file_root", ""),
                    interactive=True,
                )
                save_file_root_btn = gr.Button("Save", variant="primary")
                file_root_status = gr.Markdown("")
                gr.Markdown("### Planner Mode")
                planner_mode_radio = gr.Radio(
                    choices=["off", "auto", "always"],
                    value=load_settings().get("planner_mode", "auto"),
                    label="Planner mode",
                    info="auto: plan on complex queries. always: plan every turn. off: disabled.",
                )
                planner_mode_status = gr.Markdown("")

        demo.load(setup_and_initialize).success(servers_state, outputs=mcp_list).success(update_usage, outputs=usage_bar).success(update_task_panel, outputs=task_list_display)

        demo.unload(shutdown_client)

        msg_input.submit(
            handle_chat,
            inputs=[msg_input, chatbot, tool_validation_box, tool_allow_btn, tool_deny_btn],
            outputs=[msg_input, chatbot, tool_validation_box, tool_allow_btn, tool_deny_btn, task_list_display, usage_bar],
        ).success(update_usage, outputs=usage_bar).success(update_task_panel, outputs=task_list_display)

        refresh_servers.click(servers_state, outputs=mcp_list)
        save_servers.click(update_servers, inputs=mcp_list)

        async def allow_tool():
            await chat_session.session.set_allow_tool_action("allow")
        tool_allow_btn.click(allow_tool)
        async def deny_tool(msg_input: gr.MultimodalTextbox):
            await chat_session.session.set_allow_tool_action("deny", msg_input.value["text"])
            return gr.update(value="")
        tool_deny_btn.click(deny_tool, msg_input, msg_input)

        def autotool(value):
            if value:
                chat_session.session.allow_tool_action = "always"
                logging.info("Always Allow ON")
            else:
                chat_session.session.allow_tool_action = None
                logging.info("Always Allow OFF")
        auto_tool.select(autotool, inputs=auto_tool)

        def close_app():
            logging.info("\n--- Shutdown Complete ---")
            gr.close_all()
        def pre_close_app():
            return gr.update(value="# --- Client Disconnected ---", visible=True), gr.update(interactive=False, placeholder="Session Terminated")
        btn_stop.click(pre_close_app, outputs=[tool_validation_box, msg_input]).then(shutdown_client).then(close_app)

        demo.unload(close_app)

        save_file_root_btn.click(
            save_file_root,
            inputs=file_root_input,
            outputs=file_root_status,
        )

        planner_mode_radio.change(save_planner_mode, inputs=planner_mode_radio, outputs=planner_mode_status)
        
    return demo
    

if __name__ == "__main__":
    app = build_app()
    app.launch(
        theme='Nymbo/Nymbo_Theme', 
        debug=True,
        css=css
    )
