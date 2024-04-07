import os
from apikey import apikey
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response

os.environ["OPENAI_API_KEY"] = apikey

# check if disk storage for indexes already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
# retriever
retriever= VectorIndexRetriever(index=index, similarity_top_k=5)

# postprocessor for similiarity threshold
postprocessor=SimilarityPostprocessor(similarity_cutoff=0.80)

# query index based on the specified retriever & postprocessor
query_engine=RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

# query your data on a given input and save the response
response=query_engine.query(input('Please ask your question: '))

# display the response and show the sources
pprint_response(response, show_source=True)