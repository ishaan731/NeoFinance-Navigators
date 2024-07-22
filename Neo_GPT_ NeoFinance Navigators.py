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

index = VectorStoreIndex.from_documents(documents)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

top_k = 3

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

query = input("Query:")
response = query_engine.query(query)

context = " ".join([n.text for n in response.source_nodes[:top_k]]) + "."

model_name = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_name, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt_template_w_context = lambda context, comment: f"""FinGPT, a customer service-based chatbot for all financial-related customer queries. Respond accurately to the person's queries.
{context}
Please respond to the following comment. Use the context above if it is helpful.

{comment}
[/INST]
"""

comment = query
prompt = prompt_template_w_context(context, comment)

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

outputs = model.generate(input_ids=input_ids, max_new_tokens=280)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

start_index = output_text.find("[/INST]") + len("[/INST]")
response_text = output_text[start_index:].strip()

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