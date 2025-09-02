import os
import openfga_sdk
from openfga_sdk.client import OpenFgaClient, ClientCheckRequest
from dotenv import load_dotenv
from langchain.tools import StructuredTool
from typing import Optional
from pydantic import BaseModel, Field

load_dotenv()

universally_allowed_tools = [
    "get_current_weather",
    "reverse_user_input",
    "generate_barchart",
    "duckduckgo_search"
]

class OpenFGAClass:
    
    def __init__(self):
        self.configuration = openfga_sdk.ClientConfiguration(
            api_url=os.getenv("FGA_API_URL", "http://localhost:8080"),
            store_id=os.getenv("FGA_STORE_ID"),
            authorization_model_id=os.getenv("FGA_AUTHORIZATION_MODEL_ID")
        )
        self.client = None

    async def connect(self):
        """Connects to OpenFGA asynchronously."""
        self.client = await OpenFgaClient(self.configuration).__aenter__()

    async def close(self):
        """Closes the OpenFGA client session."""
        if self.client:
            await self.client.__aexit__(None, None, None)
            self.client = None

    async def check_user(self, user, relation, object):
        """Checks if a user is allowed to perform a relation on an object."""
        if not self.client:
            raise RuntimeError("FGA client not connected")
        
        return await self.client.check(
            body=ClientCheckRequest(user=user, relation=relation, object=object),
            options={"authorization_model_id": os.getenv("FGA_AUTHORIZATION_MODEL_ID")}
        )

    async def check_agent(self, agent, relation, object):
        """Checks if an agent is allowed to perform a relation on an object."""
        if not self.client:
            raise RuntimeError("FGA client not connected")

        return await self.client.check(
            body=ClientCheckRequest(user=agent, relation=relation, object=object),
            options={"authorization_model_id": os.getenv("FGA_AUTHORIZATION_MODEL_ID")}
        )
    
    async def check_tool(self, tool, relation, object):
        """ Checks if provided tool is able to reach the document it will be accessing """
        if not self.client:
            raise RuntimeError("FGA client not connected")

        return await self.client.check(
            body=ClientCheckRequest(user=tool, relation=relation, object=object),
            options={"authorization_model_id": os.getenv("FGA_AUTHORIZATION_MODEL_ID")}
        )

    def make_protected_tool(self, tool_name: str, tool_fn, user: str, agent: str, document: Optional[bool] = None) -> StructuredTool:
        """
        Wraps a Python function as a LangChain StructuredTool protected by OpenFGA.
        Supports dynamic document names if document=True.
        """
        import uuid

        class ToolArgs(BaseModel):
            doc_name: Optional[str] = Field(None, description="Name of the document to access") if document else None

        def log(msg: str):
            if hasattr(self, "on_log") and self.on_log:
                self.on_log(msg)
            else:
                print(msg)

        if tool_name in universally_allowed_tools:
            return StructuredTool.from_function(
                func=tool_fn,
                name=tool_name,
                description=f"Protected tool {tool_name}",
                return_direct=True
            )

        async def protected_fn(*args, doc_name: Optional[str] = None, **kwargs):
            log(f"\n🔔 Attempting to invoke tool {tool_name}...\n")

            agent_allowed = await self.check_agent(agent, "invoke", f"tool:{tool_name}")
            log(f"🔍 Can {agent} invoke {tool_name}? {'✅ Yes' if agent_allowed.allowed else '❌ No'}\n")
            
            if document:
                if not doc_name:
                    raise ValueError("doc_name must be provided for document-protected tools.")

                doc_object = f"document:{doc_name}"

                user_allowed = await self.check_user(user, "reader", doc_object)
                log(f"🔍 Can {user} read {doc_object}? {'✅ Yes' if user_allowed.allowed else '❌ No'}\n")

                tool_allowed = await self.check_tool(f"tool:{tool_name}", "reader", doc_object)
                log(f"🔍 Can {tool_name} read {doc_object}? {'✅ Yes' if tool_allowed.allowed else '❌ No'}\n")

                if not (agent_allowed.allowed and user_allowed.allowed and tool_allowed.allowed):
                    await self.close()
                    log("❌ Permissions failed, cannot perform requested action.")
                    return {"error":"Permissions failed, cannot perform requested action."}

            if not agent_allowed.allowed:
                await self.close()
                log("❌ Agent is not allowed to invoke this tool.")
                return {"error":"Agent is not allowed to invoke tool"}

            # Call the original function (sync) with kwargs
            return tool_fn(doc_name=doc_name, **kwargs)

        return StructuredTool.from_function(
            func=tool_fn,
            coroutine=protected_fn,
            name=tool_name,
            description=f"Protected tool {tool_name}",
            args_schema=ToolArgs
        )
