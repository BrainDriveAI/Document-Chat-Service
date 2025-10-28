# examples/context_retrieval_usage.py

"""
Examples of using the refactored Context Retrieval system.
The API now returns CONTEXT ONLY - clients handle LLM generation.
"""

import httpx
import asyncio


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
# EXAMPLE 1: Basic Context Retrieval
# ============================================================================

async def example_basic_retrieval():
    """Basic context retrieval without conversation history"""
    
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
                "methods": ["multi_query"]
            }
        }
    }

    response = await call_api(request)
    result = response.json()
    
    print(f"Intent: {result['intent']['type']}")
    print(f"Requires Generation: {result['requires_generation']}")
    print(f"Generation Type: {result['generation_type']}")
    print(f"Found {len(result['chunks'])} chunks")
    
    # Client decides what to do with chunks
    if result['requires_generation']:
        # Build context from chunks
        context = "\n\n".join([chunk['content'] for chunk in result['chunks'][:5]])
        
        # Call your LLM (not shown here)
        # answer = await llm.generate(query=request['query_text'], context=context)
        print(f"\nContext preview:\n{context[:500]}...")


# ============================================================================
# EXAMPLE 2: Conversational Context Retrieval
# ============================================================================

async def example_conversational_retrieval():
    """Context retrieval with conversation history"""
    
    request = {
        "query_text": "tell me more about it",
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
            "top_k": 10,
            "use_hybrid": True,
            "query_transformation": {
                "enabled": True,
                "methods": ["contextualize", "multi_query"]
            }
        }
    }
    
    response = await call_api(request)
    result = response.json()
    
    print(f"Original Query: {request['query_text']}")
    print(f"Transformed Queries: {result['metadata'].get('transformed_queries', [])}")
    print(f"Found {len(result['chunks'])} chunks")


# ============================================================================
# EXAMPLE 3: Collection Summary Context
# ============================================================================

async def example_collection_summary():
    """Get diverse chunks for collection summarization"""
    
    request = {
        "query_text": "Give me an overview of this collection",
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": [],
        "config": {
            "use_intent_classification": True,
            "top_k": 20  # Number of diverse chunks to retrieve
        }
    }
    
    response = await call_api(request)
    result = response.json()
    
    print(f"Intent: {result['intent']['type']}")
    print(f"Generation Type: {result['generation_type']}")  # Should be "collection_summary"
    print(f"Retrieved {len(result['chunks'])} diverse chunks")
    print(f"Is collection-wide: {result['metadata'].get('is_collection_wide', False)}")
    
    # Client builds summary from chunks
    if result['generation_type'] == 'summary':
        # Build context from diverse chunks
        excerpts = []
        for i, chunk in enumerate(result['chunks'], 1):
            excerpts.append(f"[Excerpt {i}]\n{chunk['content']}")
        
        print(f"EXCERPTS: {excerpts}")
        context = "\n\n".join(excerpts)
        
        # Call LLM with summary prompt (not shown)
        # summary = await llm.summarize(context=context)
        print(f"\nContext for summarization ready ({len(context)} chars)")


# ============================================================================
# EXAMPLE 4: Chat Intent (No Retrieval Needed)
# ============================================================================

async def example_chat_intent():
    """Handle general chat that doesn't need document retrieval"""
    
    request = {
        "query_text": "Hi, how are you?",
        "collection_id": "71301135-654f-494a-bef3-aebe1f84aa65",
        "chat_history": [],
        "config": {
            "use_intent_classification": True,
            "use_chat_history": True,
            "top_k": 10,
            "use_hybrid": True,
            "query_transformation": {
                "enabled": True,
                "methods": ["contextualize", "multi_query"]
            }
        }
    }
    
    response = await call_api(request)
    result = response.json()
    
    print(f"Intent: {result['intent']['type']}")  # Should be "chat"
    print(f"Requires Generation: {result['requires_generation']}")  # Should be False
    print(f"Chunks: {len(result['chunks'])}")  # Should be 0
    
    # Client handles as direct chat
    if not result['requires_generation']:
        # Direct LLM call without context
        # response = await llm.chat(query=request['query_text'])
        print("No context needed - handle as direct chat")


# ============================================================================
# EXAMPLE 5: Full Workflow with LLM Generation
# ============================================================================

async def example_full_workflow():
    """Complete workflow: retrieve context → generate response"""
    
    # Step 1: Retrieve context
    request = {
        "query_text": "what is the difference between openwebui tools, functions, actions, filter pipelines, manifold pipelines, etc?",
        "collection_id": "48a8de38-e714-40ad-9744-c5c1b73f9c9f",
        "chat_history": [],
        "config": {
            "top_k": 5,
            "use_hybrid": True,
            "query_transformation": {
                "enabled": True,
                "methods": ["multi_query", "hyde"]
            }
        }
    }
    
    # Get context
    response = await call_api(request)
    result = response.json()
    
    print(f"=== Step 1: Context Retrieved ===")
    print(f"Intent: {result['intent']['type']}")
    print(f"Chunks: {len(result['chunks'])}")
    print(f"Generation Type: {result['generation_type']}")
    
    # Step 2: Generate response based on context
    if result['requires_generation'] and result['chunks']:
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}]\n{chunk['content']}"
            for i, chunk in enumerate(result['chunks'])
        ])
        
        print(f"\n=== Step 2: Generate Response ===")
        print(f"Context size: {len(context)} characters")
        
        # Your LLM generation logic here
        # if result['generation_type'] == 'answer':
        #     response = await generate_answer(query, context)
        # elif result['generation_type'] == 'summary':
        #     response = await generate_summary(context)
        # elif result['generation_type'] == 'comparison':
        #     response = await generate_comparison(query, context)
        
        print("Ready for LLM generation!")


# ============================================================================
# Helper: Build Prompt for LLM
# ============================================================================

def build_llm_prompt(query: str, chunks: list, generation_type: str) -> str:
    """Helper to build appropriate prompt based on generation type"""
    
    # Build context from chunks
    context = "\n\n".join([
        f"[Source {i+1}]\n{chunk['content']}"
        for i, chunk in enumerate(chunks)
    ])
    
    if generation_type == "answer":
        return f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    elif generation_type == "summary":
        return f"""Provide a comprehensive summary of the following information.

Content:
{context}

Summary:"""
    
    elif generation_type == "comparison":
        return f"""Compare and contrast the concepts discussed in the following context, addressing: {query}

Context:
{context}

Comparison:"""
    
    elif generation_type == "listing":
        return f"""Based on the context, provide a clear list addressing: {query}

Context:
{context}

List:"""
    
    else:  # Default to answer
        return f"""Context:
{context}

Question: {query}

Answer:"""


# ============================================================================
# Run Examples
# ============================================================================

async def main():
    print("=" * 80)
    print("Context Retrieval API Examples")
    print("=" * 80)
    
    # await example_basic_retrieval()
    # await example_conversational_retrieval()
    # await example_collection_summary()
    # await example_chat_intent()
    await example_full_workflow()


if __name__ == "__main__":
    asyncio.run(main())
