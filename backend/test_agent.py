
import asyncio

from app.agents.support_agent import support_agent


async def main():
    customer_message = """
    My order was supposed to arrive yesterday,
    but I still haven't received it.
    Can you please help?
    """

    result = await support_agent.run(customer_message)

    print("\n--- AI CUSTOMER SUPPORT RESPONSE ---\n")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
