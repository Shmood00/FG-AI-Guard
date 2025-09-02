import asyncio, time
from OpenFGAClass import OpenFGAClass

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage


from tools import read_sensitive_file, get_weather, reverse_input, generate_barchart, duckduckgo_wrapper
import streamlit as st
import pandas as pd

load_dotenv()


def build_recent_chat_history(messages, max_turns=10):
    """
    Build a list of BaseMessage objects containing only the last N conversation turns.
    Each turn = 1 user message + 1 assistant message.
    """
    # Take last max_turns * 2 messages (user + assistant pairs)
    recent_msgs = messages[-(max_turns*2):]
    
    history = []
    for msg in recent_msgs:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

async def main():

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "username" not in st.session_state:
        st.session_state.username = None

    if "title_shown" not in st.session_state:
        st.title("Welcome to a Chat Bot")
        st.session_state.title_shown = True

    # Step 1: Ask for user name if not already provided
    if st.session_state.username is None:
        _name_input = st.text_input("Hi! What's your name?")
        if _name_input:
            st.session_state.username = _name_input.strip().lower()
            st.rerun()
        return  # Stop here until username is entered
    
    username = f"user:{st.session_state.username}"


    fga = OpenFGAClass()
    await fga.connect()

    # Wrap the tool with OpenFGA protection
    protected_read_doc = fga.make_protected_tool(
        tool_name="read_sensitive_doc",
         tool_fn=read_sensitive_file,
        user=username,
        agent="agent:agent_a",
        document=True
    )

    protected_weather = fga.make_protected_tool(
        tool_name="get_current_weather",
        tool_fn=get_weather,
        user=username,
        agent="agent:agent_a",
        document=False
    )

    protected_reverse_input = fga.make_protected_tool(
        tool_name="reverse_user_input",
        tool_fn=reverse_input,
        user=username,
        agent="agent:agent_a",
        document=False
    )

    protected_bar_chart = fga.make_protected_tool(
        tool_name="generate_barchart",
        tool_fn=generate_barchart,
        user=username,
        agent="agent:agent_a",
        document=False
    )

    protected_duck_duck_go = fga.make_protected_tool(
        tool_name="duckduckgo_search",
        tool_fn=duckduckgo_wrapper,
        user=username,
        agent="agent:agent_a",
        document=False
    )


    tools = [
        protected_read_doc,
        protected_weather,
        protected_reverse_input,
        protected_bar_chart,
        protected_duck_duck_go
    ]

    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            System Instructions:
             
            You are an AI assistant with access to the following tools:

            Available Tools:

            read_sensitive_doc – Reads the contents of a sensitive document. Use only when the user explicitly requests it.

            get_current_weather(location) – Returns the current weather for the specified location.

            reverse_user_input(text) – Reverses the text the user provides.
             
            generate_barchart(x_axis, y_axis) - Generates a bar chart of user inputted data.
             
            duckduckgo_search(input: str) - Uses DuckDuckGo to search for information based on the user's query
                         
            Behavior Guidelines:

            Understand the user’s intent fully before acting.

            Automatically decide the most appropriate tool(s) for the task.

            Explain your reasoning when using a tool.

            Summarize results clearly and provide actionable next steps if relevant.

            Avoid unnecessary tool usage.

            Handle multiple steps or tool chains if required.

            Response Procedure:

            Interpret User Input:

            Restate the user request briefly to confirm understanding.

            Identify if it requires a tool, multiple tools, or just an explanation.

            Decide on Actions/Tools:

            Choose the appropriate tool(s).

            Plan steps if multiple tools are needed.

            Execute Tools:

            Use the chosen tool(s) and clearly show the reasoning.

            If multiple steps are needed, execute sequentially.

            Provide Results:

            Present the outcome clearly.

            Include next-step suggestions if applicable.

            Example Flows

            Example 1: Weather Request
            User: “What’s the weather in Tokyo?”
            AI:

            Understand: You want the current weather in Tokyo.

            Plan: Use get_current_weather("Tokyo").

            Act: Tool result → “Currently 28°C, sunny.”

            Summarize: The current weather in Tokyo is 28°C and sunny.

            Example 2: Reverse Text
            User: “Reverse this: AI is amazing!”
            AI:

            Understand: You want me to reverse the text “AI is amazing!”

            Plan: Use reverse_user_input("AI is amazing!").

            Act: Tool result → “!gnizama si IA”

            Summarize: The reversed text is !gnizama si IA.

            Example 3: Sensitive Document
            User: "Please read my financial_report_2025.pdf"
            AI:
            Understand: You want me to read the document "financial_report_2025.pdf".
            Plan: Use read_sensitive_doc(doc_name="financial_report_2025.pdf").
            Act: Tool result → [Document contents]
            Summarize: Here is the content of your requested document: [Content].
            If the OpenFGA check fails, inform the use you cannot read the file.
            
             Example 3: Generate Bar Chart
             User: "Please generate a bar chart of the following data: x_axis - apple, pears, orange, y_axis - 1, 2, 3"
             Act: For this request, you must only call generate_barchart. Do not output any text. Only the chart should appear.
            
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)

    # Streamlit interface
    st.title("Welcome to a Chat Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    _input = st.chat_input("How can I help?")
    
    if _input:

        with st.chat_message("user"):
            
            st.markdown(_input)
        
        st.session_state.messages.append({"role":"user", "content":_input})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            typed_text = ""

            def on_log(msg: str):
                nonlocal typed_text
                typed_text += msg + "\n"
                placeholder.markdown(f"```\n{typed_text}\n```")
                time.sleep(1)
            
            fga.on_log = on_log

            # Keep track of chat history of up to 20 messages (1 user turn and 1 assistant turn)
            chat_history = build_recent_chat_history(st.session_state.get("messages", []), max_turns=10)

            resp = await agent_executor.ainvoke({"query": _input, "chat_history":chat_history})

            if isinstance(resp.get("output"), pd.DataFrame):
                st.bar_chart(resp.get("output"))
                st.session_state.messages.append({"role":"assistant", "content":"[Bar Chart generated]"})
            else:

                typed_text = ""
                for word in resp.get("output").split():
                    typed_text += word+" "
                    placeholder.markdown(typed_text)
                    time.sleep(0.05)
        
        st.session_state.messages.append({"role":"assistant", "content":typed_text})
        
    await fga.close()

if __name__ == "__main__":
    asyncio.run(main())
