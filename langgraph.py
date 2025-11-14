import os
import webbrowser
import json
import operator
import concurrent.futures
from typing import TypedDict, Annotated, Sequence, Literal
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from PIL import Image
from IPython.display import display, Image as IPImage

from backend import (
    extract_llm_intent,
    action_mapping,
    update_chat_history, 
    resolve_references_in_message,
    speak_response
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
    resolved_input: str
    actions: list
    results: list
    formatted_output: str
    error: str | None
    execution_mode: str 



def preprocess_node(state: AgentState) -> AgentState:
    """Load user input, save it to history, and resolve references."""
    user_input = state["user_input"]
    update_chat_history(user_message=user_input)
    resolved = resolve_references_in_message(user_input)
    return {**state, "resolved_input": resolved}


def intent_extraction_node(state: AgentState) -> AgentState:
    """Call the LLM to extract a list of actions."""
    resolved_input = state["resolved_input"]
    actions = extract_llm_intent(resolved_input)
    
    if isinstance(actions, dict):
        actions = [actions]
    elif not isinstance(actions, list):
        actions = [{"action": "chat", "message": str(actions)}]
    
    return {**state, "actions": actions}


def single_action_node(state: AgentState) -> AgentState:
    """Execute a single action."""
    action = state["actions"][0]
    action_type = action.get("action", "none")
    
    try:
        handler = action_mapping.get(action_type)
        if handler:
            result = handler({k: v for k, v in action.items() if k != "action"})
        else:
            result = f"❌ Unknown action: {action_type}"
    except Exception as e:
        result = f"❌ Error executing {action_type}: {e}"
    
    return {**state, "results": [result], "execution_mode": "single"}


def sequential_execution_node(state: AgentState) -> AgentState:
    """Execute actions one-by-one in order for tasks with side-effects."""
    results = []
    
    for action in state["actions"]:
        action_type = action.get("action", "none")
        try:
            handler = action_mapping.get(action_type)
            if handler:
                result = handler({k: v for k, v in action.items() if k != "action"})
            else:
                result = f"❌ Unknown action: {action_type}"
            results.append(result)
        except Exception as e:
            results.append(f"❌ Error in {action_type}: {e}")
    
    return {
        **state,
        "results": results,
        "execution_mode": "sequential"
    }


# Node 3c: Parallel execution (PERFORMANCE BOOST!)
def parallel_execution_node(state: AgentState) -> AgentState:
    """
    Execute independent, read-only actions in parallel.
    """
    def execute_action(action):
        action_type = action.get("action", "none")
        try:
            handler = action_mapping.get(action_type)
            if handler:
                return handler({k: v for k, v in action.items() if k != "action"})
            else:
                return f"❌ Unknown action: {action_type}"
        except Exception as e:
            return f"❌ Error in {action_type}: {e}"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(execute_action, state["actions"]))
    
    return {
        **state,
        "results": results,
        "execution_mode": "parallel"
    }


def format_output_node(state: AgentState) -> AgentState:
    """Combine results into user-friendly output."""
    results = state["results"]
    execution_mode = state.get("execution_mode", "unknown")
    
    formatted = f"[{execution_mode.upper()} MODE]\n"
    formatted += "\n".join(results)
    
    return {
        **state,
        "formatted_output": formatted
    }



def route_actions(state: AgentState) -> Literal["single_action", "sequential_execution", "parallel_execution"]:
    """
    Smart routing based on action dependencies:
    - Single: One action only
    - Sequential: Actions with side effects (email, file writes, browser)
    - Parallel: Independent queries (weather + wifi + system info)
    """
    actions = state["actions"]
    
    if len(actions) == 1:
        return "single_action"
    
    blocking_actions = {
        "send_email", "send_whatsapp", "create_file", "create_folder", 
        "create_project", "delete_file", "move_file_folder", "fix_code",
        "open_browser", "navigate_to", "search_website", "trash_files",
        "change_wallpaper", "save_note", "rename_file", "setup_python_environment"
    }
    
    action_types = [a.get("action") for a in actions]
    
    if any(act in blocking_actions for act in action_types):
        return "sequential_execution"
    
    return "parallel_execution"


def create_langgraph_agent():
    """
    Creates the optimized LangGraph agent with smart routing
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("extract_intent", intent_extraction_node)
    workflow.add_node("single_action", single_action_node)
    workflow.add_node("sequential", sequential_execution_node)
    workflow.add_node("parallel", parallel_execution_node)
    workflow.add_node("format_output", format_output_node)
    
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "extract_intent")
    
    workflow.add_conditional_edges(
        "extract_intent",
        route_actions,
        {
            "single_action": "single_action",
            "sequential_execution": "sequential",
            "parallel_execution": "parallel"
        }
    )
    
    workflow.add_edge("single_action", "format_output")
    workflow.add_edge("sequential", "format_output")
    workflow.add_edge("parallel", "format_output")
    workflow.add_edge("format_output", END)
    
    return workflow.compile()


app = create_langgraph_agent()

def handle_intent_langgraph(user_message: str) -> str:
    """
    Drop-in replacement for backend.handle_intent() using LangGraph
    """
    initial_state = {
        "user_input": user_message,
        "messages": [HumanMessage(content=user_message)],
        "actions": [],
        "results": [],
        "resolved_input": "",
        "formatted_output": "",
        "error": None,
        "execution_mode": ""
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    output = final_state.get("formatted_output", "No output generated")
    
    update_chat_history(assistant_message=output)
    
    return output


def visualize_graph_pil():
    """
    Display graph using PIL Image viewer (works in any environment)
    """
    png_data = app.get_graph().draw_mermaid_png()
    
    output_path = os.path.expanduser("~/langgraph_workflow.png")
    with open(output_path, "wb") as f:
        f.write(png_data)
    
    try:
        img = Image.open(output_path)
        img.show()  
        print(f"✅ Graph visualization displayed!")
    except Exception as e:
        print(f"❌ Failed to open image with PIL: {e}")
        print("Trying system viewer...")
        visualize_graph_and_display() 

def visualize_graph_and_display():
    """
    Generate LangGraph visualization as PNG and open it in default system viewer
    """
    png_data = app.get_graph().draw_mermaid_png()
    output_path = os.path.expanduser("~/langgraph_workflow.png")
    with open(output_path, "wb") as f:
        f.write(png_data)
    
    print(f"✅ Graph saved to: {output_path}")
    
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_path)
        elif os.name == 'posix':  # Linux/Mac
            os.system(f'xdg-open "{output_path}"')
    except Exception as e:
        print(f"❌ Could not open image automatically: {e}")

def visualize_graph_jupyter():
    """
    Display graph in Jupyter/IPython notebook
    """
    png_data = app.get_graph().draw_mermaid_png()
    display(IPImage(data=png_data))

def print_graph_ascii():
    """
    Print graph structure as ASCII text in terminal
    """
    print("\n" + "="*60)
    print("LANGGRAPH WORKFLOW STRUCTURE")
    print("="*60)
    print(app.get_graph().draw_ascii())
    print("="*60 + "\n")


def main():
    print("🔥 Lucifer Agent (LangGraph Edition) Ready")
    print("\nCommands:")
    print("  'visualize' - Show graph as PNG image (opens in viewer)")
    print("  'graph' - Print ASCII workflow in terminal")
    print("  'exit' - Quit\n")
    
    while True:
        try:
            user_input = input("\n👿 Command: ").strip()
            
            if user_input.lower() in ("exit", "quit"):
                print("👿 Agent terminated.")
                break
            
            if user_input.lower() == "visualize":
                try:
                    visualize_graph_pil()  # Try PIL first
                except Exception:
                    print("PIL failed, trying fallback system viewer...")
                    visualize_graph_and_display() # Fallback
                continue
            
            if user_input.lower() == "graph":
                print_graph_ascii()
                continue
            
            if not user_input:
                continue
            
            result = handle_intent_langgraph(user_input)
            
            print(f"\n{result}")
            
        
            import re
            spoken_result = re.sub(r'\[\w+\sMODE\]\n', '', result)
            speak_response(spoken_result)
            
        except (KeyboardInterrupt, EOFError):
            print("\n👿 Agent interrupted.")
            break


if __name__ == "__main__":
    main()
