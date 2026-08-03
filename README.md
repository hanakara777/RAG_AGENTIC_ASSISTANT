**Arxiv RAG Agent**

I wanted to be able to get quality replies with getting data from one specific source which is Arxiv, where are more than 2 million scientific papers.
With that, I also wanted the model to evaluate its own answers - *is it accurate enough?*, if not, to be able to recognise inaccuracy and improve it
until satisfied.

**USING:**
1. LLM **Deepseek Reasoner** for processing and evaluation
2. **Qdrant** for vector database
3. **Google Generative AI** embeddings for document retrival
4. **Streamlit** for user interface
5. **LangGraph and LangChain** for orchestration

Notice: all prompts for llm in code are made with ai, because of my poor wording. Other than that my knowledge for building this was gained with 
references api's and their examples (then modifying examples based on my wishes), youtube videos, forums, a lot of reading and mostly trial and error.
There were quite a few bugs that happened when deploying online because of environment/keys,
which i handled with researching errors, last case scenario using ai to teach me about error and then trying to fix it myself.

Doing, learning and finishing projects like this is what motivates me.

hana
