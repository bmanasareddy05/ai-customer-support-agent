
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


# Connect Pydantic AI to our local Ollama server
model = OllamaModel(
    "qwen3:4b",
    provider=OllamaProvider(
        base_url="http://localhost:11434/v1"
    ),
)

# Define our customer support agent
support_agent = Agent(
    model,
    system_prompt="""
    You are a professional AI customer support assistant.

    Your responsibilities:
    1. Understand customer questions.
    2. Respond politely and professionally.
    3. Be empathetic towards customer concerns.
    4. Never invent order details or company policies.
    5. Ask for clarification when information is missing.
    6. Keep responses concise and helpful.

    Important:
    You must never claim that an order was refunded,
    shipped, or cancelled without verified information.
    """,
)
