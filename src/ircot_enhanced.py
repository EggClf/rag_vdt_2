def ircot_enhanced_pipeline(question, llm, retriever, final_answer_function=None, previous_messages=[], max_steps=5):
    """
    Implements the IRCoT pipeline with interleaved reasoning and retrieval for backend service with streaming
    """
    reasoning_steps = []
    retrieved_contexts = []
    
    # Initial retrieval based on the question
    initial_context, ignored_ids = retriever(question)
    retrieved_contexts.append(initial_context)
    
    num_step = 0
    
    yield '\n<think>\n'
    yield 'Starting iterative reasoning and retrieval process...\n\n'
    
    for step in range(max_steps):
        yield f'\n\n========== Step {step+1} =============\n\n'
        
        # Summarize prior context if there are reasoning steps
        if reasoning_steps:
            yield "Summarizing context...\n"
            
            summary_messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that summarizes a set of documents relevant to a question."
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nDocuments:\n{retrieved_contexts[-1]}\n\nSummarize the most relevant facts that can support answering the question."
                }
            ]
            
            summary = ""
            for chunk in llm.stream(summary_messages):
                if isinstance(chunk, str):
                    summary += chunk
            
            yield f"Context summary: {summary}\n\n"
        else:
            summary = ""
            
        # Generate reasoning based on current information
        yield "Generating reasoning step...\n"
        
        reasoning_messages = [
            {
                "role": "system", 
                "content": "You are a careful reasoner. You are trying to answer a complex multi-hop question. Each time, read the summary of earlier context and newly retrieved documents and add one more step to your reasoning."
            },
            {
                "role": "user", 
                "content": f"""Question: {question} Prior Context Summary:{summary} New Documents:\n{retrieved_contexts[-1]}Reasoning Steps So Far:{' '.join(reasoning_steps) if reasoning_steps else 'None'} What is the next step in your reasoning?"""
            }
        ]
        
        reasoning = ""
        yield "\nReasoning:\n"
        for chunk in llm.stream(reasoning_messages):
            if isinstance(chunk, str):
                reasoning += chunk
                yield chunk
                
        # Check if we can answer the question
        yield "\n\nDeciding next action...\n"
        
        current_reasoning = '\n'.join(reasoning_steps + [reasoning]) if reasoning_steps else reasoning
        
        decision_messages = [
            {
                "role": "system", 
                "content": "You are a decision-making assistant for a retrieval-augmented reasoning system. Decide whether to retrieve more information or to answer the question. Only answer with one word: 'retrieve' or 'answer'."
            },
            {
                "role": "user", 
                "content": f"Question: {question}\n\nCurrent Reasoning:\n{current_reasoning}\n\nDecision:"
            }
        ]
        
        decision = ""
        for chunk in llm.stream(decision_messages):
            if isinstance(chunk, str):
                decision += chunk
        
        decision = decision.strip().lower()
        yield f"\nDecision: {decision}\n"
            
        if decision == "answer":
            # Extract the final answer
            yield "\nExtracting final answer...\n"
            
            extract_messages = [
                {
                    "role": "system", 
                    "content": "You are an assistant that extracts a short answer (a word or entity) from a reasoning trace."
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nReasoning:\n{current_reasoning}\n\nThe answer to the question is:"
                }
            ]
            
            final_answer = ""
            for chunk in llm.stream(extract_messages):
                if isinstance(chunk, str):
                    final_answer += chunk
                    yield chunk
            
            break
                
        reasoning_steps.append(f"Step {step+1}: {reasoning}")
        num_step += 1
        
        # Retrieve based on the latest reasoning
        yield f"\n\n## Retrieving new information based on reasoning\n\n"
        new_context, ignored_ids = retriever(reasoning)
        retrieved_contexts.append(new_context)
        
        yield new_context
    
    yield f'\n\n========== Total process end with {num_step} hops =============\n\n'
    
    # Close the think tag before generating the final answer
    yield '</think>\n\n'
    
    if decision != "answer":
        yield "ANSWER: Insufficient Information."
    else:
        yield f"ANSWER: {final_answer.strip()}"