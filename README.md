# Intelligent-Complaint-Analysis-for-Financial-Services
# Introduction  
This week we are trying to help CrediTrust financial, which is a digital financial company that offers services in credit cards, personal loans, Buy Now, Pay Later (BNPL), savings accounts, and money transfers, to find insight in the customer complaints using AI. We are building a chatbot that helps teams in the company understand customer pain points in the 5 product groups.  

# Methodology  
**Exploratory Data Analysis**  
We have complaint data from Consumer Financial Protection Bureau (CFPB) containing customer complaints.   First, we perform exploratory data analysis to understand the data. This includes:
	Analyzing the distribution of complaints across product categories.
	Analyzing the Consumer Complaint Narrative column to understand it better.
	Filtering the dataset to include the five products. 
	Cleaning the text by removing special characteristics and boilerplate text.

**Text Chunking**  
In this process we converted the text to a format best suited for RAG using vector embedding. Vector embedding is a technique used to convert data like text and images into numerical vectors so that machines can understand and work with them. These vectors are designed in a way that they capture the meaning, structure, or relationships within the original data. This process includes:
	Chunking the test using LangChain's RecursiveCharacterTextSplitter function with a chunk size and chunk overlap that optimizes speed and accuracy.

	Using an embedding model all-MiniLM-L6-v2 from SentenceTransformer to create the vectors. The model was chosen because it is fast, lightweight, local, free and open source.
	Embedding the text using FAISS and storing the index with the metadata.  

    

**Building the RAG Core Logic and Evaluation**  
For this task, we built a retrieval and generation pipeline based on the index we built in the previous task. This included:
	Retriever Implementation – take query from user and embed it in a vector using the model we used on the chunks in task 2. Now we can find the most relevant chunks.
	Generator Implementation – use an LLM to generate responses based on the query, prompt, and context we found. We used the model ‘TinyLlama/TinyLlama-1.1B-Chat-v1.0’ as it is simple, fast and local.
	Qualitative Evaluation – Ask the model questions to see if it is working correctly.
 
**Interactive Chat Interface**
Now that we have our RAG pipeline, we can build a simple chatbot that answers questions that the teams have using the RAG pipeline and our original data. We used Gradio to build this chatbot that is pictured below. The chat bot has a text input box for the user to type their question, a "Submit" or "Ask" button and a display area for the AI-generated answer.
	 
	 
# Challenges and Recommendations
•	The data is too big to work on comfortably. For example, trying to count the words of Consumer Complaints Narrative was impossible because it kept making Google Collab crash. Also, the embedding model kept taking too long (above 8 hours). This was solved by using GPU instead of CPU. But the FAISS index was too large to put into github repo.

•	There are very similar categories in the Issue column. This means in the EDA, there are more categories than necessary in the distribution of complaints. These categories were too many to include in the figure above.

•	In the RAG process, Google collab kept saying that GPU access is restricted which made the completion time longer. This meant we needed to settle for a faster but less accurate LLM model. Also, implementing reranking became difficult as GPU usage was limited and CPU needed 41 hours to answer a question. 


# Conclusion 
Our primary goal this week is to build a chatbot that enables teams in CrediTrust to better understand customer frustrations related to the five major financial product lines. To power this system, we’re leveraging language models in combination with vector embedding to transform unstructured complaint narratives into actionable information. We're developing a semantic search pipeline that stores text embedding in FAISS allowing us to retrieve the most relevant content in response to natural-language questions. The chatbot uses these retrieved complaint excerpts as context for generating grounded, intelligent answers, ensuring the responses are both accurate and meaningful. Along the way, we’re gaining experience in managing large-scale, real-world text data, handling noise, and structuring unorganized narratives for analysis.

