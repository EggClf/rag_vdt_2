def ircot_pipeline(query, llm, retriever, final_answer_function, previous_messages=[], max_steps=5):
    """
    Implements the IRCoT pipeline with interleaved reasoning and retrieval
    """
    reasoning_steps = []
    retrieved_contexts = []
    final_answer = ""
    
    # Initial retrieval based on the question
    initial_context, ignored_ids = retriever(query)
    retrieved_contexts.append(initial_context)
    num_step = 0
    
    yield '\n<think>\n'
    yield 'Starting iterative reasoning and retrieval process...\n\n'
    
    for step in range(max_steps):
        yield f'\n\n========== Step {step+1} =============\n\n'
        
        # Create prompt to check if we have enough evidence
        prompt_content = f"""Original Question: {query}

Retrieved Information: {retrieved_contexts[-1]}
Previous Reasoning: {' '.join(reasoning_steps) if reasoning_steps else 'None'}

Based on the reasoning steps so far and the retrieved information,
do we have enough evidence to answer the original question?
MUST ANSWER with 'YES' or 'NO'. If 'YES', further answer the question (MUST ANSWER 'YES' BEFORE ANSWER, The format answer is 'YES . ANSWER: <your answer>' AND DO NOT EXPLAIN THE ANSWER). If 'NO' provide the information need to know to answer the question (MUST ANSWER 'NO' BEFORE ANSWER and ANSWER: <your answer>' with <your answer> is the information you need to know)."""

        check_messages = [
            {
                'role': 'user',
                'content': prompt_content
            }
        ]
        
        reasoning = ""
        
        for chunk in llm.stream(check_messages):
            if isinstance(chunk, str):
                reasoning += chunk
                yield chunk
        
        # Check if we have an answer
        if "NO" not in reasoning and ("YES" in reasoning or 'ANSWER' in reasoning):
            final_answer = reasoning
            break
        
        # Store reasoning step and continue
        reasoning_steps.append(f"Step {step+1}: {reasoning}")
        num_step += 1
        
        # Extract search query from reasoning
        if "NO" in reasoning:
            search_query = reasoning.replace("NO", "").strip()
            if "ANSWER:" in search_query:
                search_query = search_query.split("ANSWER:")[-1].strip()
        else:
            search_query = reasoning  # Default to full reasoning
        
        # Retrieve based on the latest reasoning
        new_nodes , ignored_ids = retriever(query)
        new_context = "\n".join([node.page_content for node in new_nodes])
        retrieved_contexts.append(new_context)
        
        yield f'\n\n## Retrieved new information based on: {search_query}\n\n'
        yield new_context
    
    yield f'\n\n========== Total process end with {num_step} hops =============\n\n'
    
    # Close the think tag before generating the final answer
    yield '</think>\n\n'
    
    if final_answer == "":
        yield "ANSWER: Insufficient Information."
    else:
        # Extract just the answer part if needed
        if "ANSWER:" in final_answer:
            answer_part = final_answer.split("ANSWER:")[-1].strip()
            yield f"ANSWER: {answer_part}"
        else:
            yield final_answer