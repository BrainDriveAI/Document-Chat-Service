"""
Comprehensive examples of using the advanced RAG system.
"""

import httpx
import asyncio
from typing import Dict, Any


BASE_URL = "http://localhost:8000"

async def call_api(payload: dict):
    timeout = httpx.Timeout(
        connect=10.0,  # time to establish connection
        read=120.0,    # time to wait for a response
        write=10.0,    # time to send data
        pool=None,     # optional
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(f"{BASE_URL}/search/", json=payload)


# ============================================================================
# EXAMPLE 1: Basic Search (No History)
# ============================================================================

async def example_basic_search():
    """Simple search without conversation history"""
    
    request = {
        "query_text": "What is BrainDrive?",
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": [],
        "config": {
            "top_k": 10,
            "use_hybrid": True,
            "use_intent_classification": True,
            "query_transformation": {
                "enabled": True,
                "methods": ["multi_query"]  # Generate multiple query variations
            }
        }
    }

    response = await call_api(request)
    result = response.json()
        
    print(f"Intent: {result['intent']['type']}")
    print(f"Transformed Queries: {result['transformed_queries']}")
    print(f"Found {len(result['chunks'])} chunks")
    
    for chunk in result['chunks'][:3]:
        print(f"\n--- Chunk {chunk['id']} ---")
        print(chunk['content'][:200])
        


# ============================================================================
# EXAMPLE 2: Conversational Search with History
# ============================================================================

async def example_conversational_search():
    """Search with conversation context"""
    
    request = {
        "query_text": "tell me more about it",  # Ambiguous without context
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": [
            {
                "role": "user",
                "content": "What is BrainDrive?"
            },
            {
                "role": "assistant",
                "content": "BrainDrive is a decentralized, open-source platform designed to empower users and developers to create, manage, and share AI-driven applications while prioritizing ownership, freedom, and sustainability."
            }
        ],
        "config": {
            "use_chat_history": True,
            "max_history_turns": 3,
            "query_transformation": {
                "enabled": True,
                "methods": ["contextualize", "multi_query"]  # Contextualize first, then generate variations
            }
        }
    }
    
    response = await call_api(request)
    result = response.json()
    
    print(f"Original Query: tell me more about it")
    print(f"Contextualized: {result['transformed_queries'][0]}")
    print(f"Found {len(result['chunks'])} relevant chunks")


# ============================================================================
# EXAMPLE 3: HyDE (Hypothetical Document Embeddings)
# ============================================================================

async def example_hyde_search():
    """Using HyDE for question-answering workloads"""
    
    request = {
        "query_text": "How do I implement authentication in a REST API?",
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": [],
        "config": {
            "query_transformation": {
                "enabled": True,
                "methods": ["hyde"]  # Generate hypothetical answer
            },
            "use_hybrid": True
        }
    }
    
    response = await call_api(request)
    result = response.json()
    
    print("HyDE Query (hypothetical document):")
    print(result['transformed_queries'][0])
    print(f"\nFound {len(result['chunks'])} matching chunks")


# ============================================================================
# EXAMPLE 4: Collection Summary
# ============================================================================

async def example_collection_summary():
    """Get a summary of entire collection"""
    
    request = {
        "query_text": "Give me an overview of this collection",
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": [],
        "config": {
            "use_intent_classification": True  # Will detect COLLECTION_SUMMARY intent
        }
    }
    
    response = await call_api(request)
    result = response.json()
    
    print(f"Intent: {result['intent']['type']}")
    print(f"Summary:\n{result['summary']}")


# ============================================================================
# EXAMPLE 5: Complex Conversational Flow
# ============================================================================

async def example_complex_conversation():
    """Simulate a multi-turn conversation with various intents"""
    
    conversation_history = []
    
    # Turn 1: Initial question
    request1 = {
        "query_text": "What are the key features of BrainDrive?",
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": conversation_history,
        "config": {
            "query_transformation": {
                "enabled": True,
                "methods": ["multi_query"]
            }
        }
    }

    response1 = await call_api(request1)
    result1 = response1.json()
    
    print("=== Turn 1 ===")
    print(f"Query: {request1['query_text']}")
    print(f"Intent: {result1['intent']['type']}")
    print(f"Chunks found: {len(result1['chunks'])}")
    
    # Add to history
    conversation_history.append({
        "role": "user",
        "content": request1['query_text']
    })
    conversation_history.append({
        "role": "assistant",
        "content": """
            Key Features
            Decentralized Structure:

            Operates as a federated network of independent, user-owned AI nodes (e.g., businesses, communities) that collaborate while maintaining autonomy.
            Shared resources (templates, tools, and open-source code) foster innovation and reduce duplication of effort.
            Core Components:

            BrainDrive Core: Provides foundational ownership and control mechanisms for users.
            BrainDrive Studio: A drag-and-drop visual editor for building custom AI-powered pages and dashboards.
            BrainDrive Marketplace: A sustainable ecosystem for trading plugins, tools, and services.
            BrainDrive Community: A hub for collaboration, learning, and collective problem-solving.
            BrainDrive LLC: Acts as the first node in the network, demonstrating decentralized business practices and offering managed hosting, support, and development leadership.
        """
    })
    
    # Turn 2: Follow-up (needs contextualization)
    request2 = {
        "query_text": "tell me more marketplace",
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": conversation_history,
        "config": {
            "use_chat_history": True,
            "query_transformation": {
                "enabled": True,
                "methods": ["contextualize", "multi_query"]
            }
        }
    }

    response2 = await call_api(request2)
    result2 = response2.json()
    
    print("\n=== Turn 2 ===")
    print(f"Query: {request2['query_text']}")
    print(f"Intent: {result2['intent']['type']}")
    print(f"Chunks found: {len(result2['chunks'])}")


EXAMPLES = {
    "basic_search": example_basic_search,
    "conversational_search": example_conversational_search,
    "hyde_search": example_hyde_search,
    "collection_summary": example_collection_summary,
    "complex_conversation": example_complex_conversation,
}

# ----------------------------------------------------------------------------
# Option 1: Toggle manually in code
# ----------------------------------------------------------------------------
ENABLED_EXAMPLES = {
    "basic_search": False,
    "conversational_search": False,
    "hyde_search": False,
    "collection_summary": True,
    "complex_conversation": False,
}


async def main():
    # Run only enabled examples
    tasks = [
        func()
        for name, func in EXAMPLES.items()
        if ENABLED_EXAMPLES.get(name, False)
    ]

    if not tasks:
        print("⚠️ No examples enabled. Edit ENABLED_EXAMPLES to choose one.")
        return

    print(f"Running {len(tasks)} enabled example(s)...\n")

    # Run all enabled ones concurrently (not sequentially)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
