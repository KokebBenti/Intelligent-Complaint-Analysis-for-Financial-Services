def get_user_query():
    query = input("Ask your question: ")
    return query

def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def similarity_search(index, vector, chunks, model, k=5):
    distances, indices = index.search(vector, k)
    results = [chunks[i] for i in indices[0]]
    return results

def build_rag_prompt(query,context):
  prompt = f""" You are a helpful and objective financial data analyst.
 You will be given a user's question and a set of context documents retrieved from internal company data. Your task is to answer the question based **only on the information provided in the context**. 
 Do not use any prior knowledge or make assumptions that are not grounded in the context. If the context does not contain enough information to answer the question, say so clearly.
 Be concise, clear, and insightful in your response.
 ---
 Context: {context}
 ---
 User Question:{query}
 ---
 Answer:"""
  return prompt 

def llm_usage(final_prompt):
  from langchain.llms import HuggingFacePipeline
  from transformers import pipeline
  hf_pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=256)
  llm = HuggingFacePipeline(pipeline=hf_pipeline)      
  response = llm(final_prompt)
  return response

def extract_answer(response_text):
  if "Answer:" in response_text:
    return response_text.split("Answer:")[-1].strip()
  else:
    return response_text.strip() 
     

def rerank_chunks(user_question, retrieved_chunks, top_k=5, batch_size=8):
    from tqdm import tqdm
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(user_question, chunk) for chunk in retrieved_chunks]
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="Re-ranking"):
        batch = pairs[i:i+batch_size]
        batch_scores = reranker.predict(batch)
        scores.extend(batch_scores)
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), reverse=True)]
    return ranked_chunks[:top_k] 