def ircot_pipeline_agent(query, llm, retriever, final_answer_function, previous_messages=[], max_steps=5):
    """
    Implements the IRCoT pipeline with interleaved reasoning and retrieval
    """
    reasoning_steps = []
    retrieved_contexts = []
    final_answer = ""
    # Initial retrieval based on the question
    initial_context, ignored_ids = retriever(query)

    # print(initial_context)
    retrieved_contexts.append(initial_context)
    num_step=0
    
    sys_prompt = """Carefully evaluate the preceding question. Determine if you possess all the information required to provide a demonstrably accurate, confident, and complete answer.

If you have any doubt whatsoever or identify any missing piece of information (however small) that prevents you from providing a fully comprehensive and confident answer, you MUST select a 'YES' option below.

Respond with exactly one of the following:

NO: I am absolutely certain I have all necessary information to answer the question confidently and completely.
YES ONE: I need exactly one additional piece of information to answer confidently and completely.
YES MULTIPLE: I need multiple additional pieces of information to answer confidently and completely
"""
    
    
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": f"<context>\n\n{initial_context}\n\n</context>\n\nQuestion: {query}"
        }
    ]
    
    yield '\n<think>\n'
    yield 'First decision:\n\n'
    
    reasoning = ""
    
    reasoning_generator = llm.stream(messages)
    for chunk in reasoning_generator:
        if isinstance(chunk, str):
            reasoning += chunk
            yield chunk
    
    if "NO" in reasoning:
        yield '\n\nNO detected, proceeding to final answer...\n\n'
        
        yield f'========== Total process end with 1 hops =============\n\n' 
        
        # Close the think tag before generating the final answer
        yield '</think>\n\n'
                
        for chunk in final_answer_function(llm, query, initial_context, previous_messages):
            if isinstance(chunk, str):
                final_answer += chunk
                yield chunk
        
    elif "YES ONE" in reasoning:
        yield '\n\nYES ONE detected, proceeding to two-hop retrieval...\n\n'
        
        messages.append({
            "role": "assistant",
            "content": reasoning
        })
        
        messages.append({
            "role": "user",
            "content": f"""Based on the above question, what specific information do you need to answer the question? Respond with exactly one piece of information needed in the format:
REQUIREMENT: [Specific information needed]"""
        })

        num_step += 1

        yield f'\n\n========== Step 1 =============\n\n'

        info_need = ""
        generator = llm.stream(messages)
        for chunk in generator:
            if isinstance(chunk, str):
                info_need += chunk
                yield chunk

        info_need = info_need.split('REQUIREMENT:')[-1].strip()
        
        new_context, _ = retriever(info_need, ignore_ids=ignored_ids)
        retrieved_contexts.append(new_context)
        
        yield f'\n\n========== Total process end with 2 hops =============\n\n'
        
        # Close the think tag before generating the final answer
        yield '</think>\n\n'
        
        for chunk in final_answer_function(llm, query,'\n\n'.join(retrieved_contexts), previous_messages):
            if isinstance(chunk, str):
                final_answer += chunk
                yield chunk 
        
    elif "YES MULTIPLE" in reasoning:
        
        yield '\n\nYES MULTIPLE detected, starting iterative retrieval process...\n\n'
        
        for step in range(max_steps):
            
            cumulative_context = "\n".join(retrieved_contexts) if retrieved_contexts else ""
            
            yield f'\n\n========== Step {step+1} =============\n\n'
            
            yield '## Checking agent:\n\n'
            
            prompt = f"""Original Question: {query}
            
            {"Information Retrieved So Far:" + cumulative_context if cumulative_context else ''}
            
            Based on the original question and any information already retrieved, what specific information do you still need to provide a complete answer?
            
            **Instructions:**
            - If you now have sufficient information, respond with: "SUFFICIENT"
            - If you still need more information, specify exactly what you need
            - Focus only on the most critical missing piece for this iteration
            - Be specific about what type of information, data, or details are required
            
            **Response Format:**
            STATUS: [SUFFICIENT | NEED_MORE]
            REQUIREMENT: [Specific information needed, or None if sufficient]"""          
            check_messages = [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            reasoning = ""
            
            for chunk in llm.stream(check_messages):
                if isinstance(chunk, str):
                    reasoning += chunk
                    yield chunk
            
            reasoning_steps.append(f"Step {step+1}: {reasoning}")
            
            # Only retrieve if more information is needed
            if "SUFFICIENT" not in reasoning.upper():
                num_step += 1
                # Extract the specific requirement for retrieval
                requirement_lines = reasoning.split('\n')
                search_query = reasoning  # Default to full reasoning
                
                # Try to extract more focused search query
                for line in requirement_lines:
                    if 'REQUIREMENT:' in line:
                        search_query = line.split('REQUIREMENT:')[-1].strip()
                   
                # Retrieve based on the specific requirement
                new_context, new_ignored_ids = retriever(search_query, )
                retrieved_contexts.append(new_context)
                
                yield '## Summarizing context:\n\n'
                
                prompt = f"""Summarize those information: {new_context} to make it suitable to answer the {query}"""

                summary_messages = [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]

                new_context = ""
                
                for chunk in llm.stream(summary_messages):
                    if isinstance(chunk, str):
                        new_context += chunk
                        yield chunk
                
                retrieved_contexts.append(new_context)
                ignored_ids.update(new_ignored_ids)
            else:
                break
           
        yield f'\n\n========== Total process end with {num_step+1} hops =============\n\n'
        
        # Close the think tag before generating the final answer
        yield '</think>\n\n'
        
        for chunk in final_answer_function(llm, query, "\n\n".join(retrieved_contexts), previous_messages):            
            if isinstance(chunk, str):
                final_answer += chunk
                yield chunk



