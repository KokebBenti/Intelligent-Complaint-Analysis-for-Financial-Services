def text_chunk(df,column):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  df['chunks'] = df[column].apply(lambda x: splitter.split_text(x))
  return(df) 

def vector_embedding(df,col):
 from sentence_transformers import SentenceTransformer
 model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda') 
 texts = df[col].tolist()
 embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
 df['embedding'] = list(embeddings) 
 return df

def vector_store(df,col):
 path= "/content/drive/MyDrive/Tenx/Week 6/faiss_index.idx"
 import faiss
 import numpy as np
 embeddings = np.vstack(df[col].values).astype('float32')
 dimension = embeddings.shape[1]
 index = faiss.IndexFlatL2(dimension)
 index.add(embeddings)
 faiss.write_index(index, path)