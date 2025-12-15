"""
Minimal Custom Tools - BaseTool subclass with Pydantic v2 fix
"""

import os
import requests
from typing import Type
from pydantic import BaseModel, Field


# Simple BaseTool implementation that works with Pydantic v2
class SimpleTool:
    def __init__(self, name: str, description: str, args_schema: Type[BaseModel] = None):
        self.name = name
        self.description = description
        self.args_schema = args_schema

    def invoke(self, **kwargs):
        # This would be implemented by specific tools
        pass


# Pydantic models
class MultiplyInput(BaseModel):
    x: float = Field(description="First number")
    y: float = Field(description="Second number")


class SearchInput(BaseModel):
    query: str = Field(description="Search query")


# Custom tools
class MultiplyTool(SimpleTool):
    def __init__(self):
        super().__init__(
            name="multiply",
            description="Multiply two numbers",
            args_schema=MultiplyInput
        )

    def invoke(self, **kwargs):
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        return f"{x} × {y} = {x * y}"


class SearchTool(SimpleTool):
    def __init__(self):
        super().__init__(
            name="search",
            description="Search Wikipedia",
            args_schema=SearchInput
        )

    def invoke(self, **kwargs):
        query = kwargs.get('query', '')
        return f"Searching Wikipedia for: {query}"


# Create and test tools
tools = [MultiplyTool(), SearchTool()]

print("Custom Tools Test")
for tool in tools:
    print(f"\n{tool.name}: {tool.description}")

# Test multiplication
result = tools[0].invoke(x=10, y=20)
print(f"\nTest: 10 × 20 = {result}")

# Test search
result = tools[1].invoke(query="artificial intelligence")
print(f"\nTest search: {result}")