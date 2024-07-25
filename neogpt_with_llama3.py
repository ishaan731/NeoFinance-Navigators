##This will be the paid version of our prototype if taken public.

!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install peft
!pip install auto-gptq
!pip install optimum
!pip install bitsandbytes

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

documents = SimpleDirectoryReader("articles").load_data()

# store docs into vector DB
index = VectorStoreIndex.from_documents(documents)

from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
import torch
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
import torch

# Set up the retriever and query engine
top_k = 3

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# Get the query from the user
query = input("Query:")
response = query_engine.query(query)

# Combine the context from the top k source nodes
context = " ".join([n.text for n in response.source_nodes[:top_k]]) + "."

# Define the prompt template with context
prompt_template_w_context = lambda context, comment: f"""NeoGPT, a customer service-based chatbot for all financial-related customer queries. Respond accurately to the person's queries. Always end the whole response with a fullstop.
{context}
Please respond to the following comment. Use the context above if it is helpful.

{comment}
[/INST]
"""

comment = query
prompt = prompt_template_w_context(context, comment)

# Use the Hugging Face Inference API for the Llama 3 model
client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token="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",
)

# Generate the response using the Llama 3 model
response_text = ""
for message in client.chat_completion(
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500,
    stream=True,
):
    response_text += message.choices[0].delta.content

# Process and format the response
def process_response(text):
    text = text.replace('..', '.')
    parts = text.split('[/INST]')
    processed_parts = []
    for part in parts:
        part = part.strip()
        if part:
            lines = part.splitlines()
            formatted_lines = []
            for line in lines:
                sentences = line.split('. ')
                formatted_lines.extend([sentence.strip() + '.' for sentence in sentences if sentence.strip()])
            processed_parts.append("\n".join(formatted_lines))
    return "\n\n".join(processed_parts)

formatted_response = process_response(response_text)
print(formatted_response)
