import gradio as gr
def answer_question(question):
  import faiss
  index_path = "/content/drive/MyDrive/Tenx/Week 6/faiss_index.idx"
  index = faiss.read_index(index_path)
  import pandas as pandas
  df= pd.read_csv("/content/drive/MyDrive/Tenx/Week 6/faiss_metadata.csv")
  import RAG
  model = RAG.load_embedding_model()
  embedding = model.encode([question])
  result=RAG.similarity_search(index,embedding,df["chunks"],5)
  context = "\n\n".join(result)
  final_prompt=RAG.build_rag_prompt(query,context)
  response=RAG.llm_usage(final_prompt)
  answer=RAG.extract_answer(response)
  return f"Answer to: '{question}' is {answer}."
  
# Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question here...", label="Your Question"),
    outputs=gr.Textbox(label="AI Answer"),
    title="Insight To Customer Complaint",
    description="Type your question and press Submit to get a response.",
    live=False)

demo.launch(debug=True)