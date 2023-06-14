from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext
from pathlib import Path
from llama_index import GPTListIndex, LLMPredictor
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig
import os

os.environ["OPENAI_API_KEY"] = "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"

def createIndex(directory):
    file_names = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            file_names.append(file)
            

    UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
    loader = UnstructuredReader()
    doc_set = {} 
    for file in file_names:
        docs = loader.load_data(file=Path(f'./{directory}/{file}'), split_documents=False)
        doc_set[file] = docs

    service_context = ServiceContext.from_defaults(chunk_size_limit=512)
    index_set = {}
    for file in file_names:
        cur_index = GPTSimpleVectorIndex.from_documents(doc_set[file], service_context=service_context)
        index_set[file] = cur_index
        cur_index.save_to_disk(f'index/{file}.json')

# createIndex('./content')

def createGraph(directory):
    file_names = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            file_names.append(file)
            
    index_set = {}
    for file in file_names:
        cur_index = GPTSimpleVectorIndex.load_from_disk(f'index/{file}.json')
        index_set[file] = cur_index
        
    index_summaries = [f"information of {file}" for file in file_names]

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.3, max_tokens=512))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)


    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [index_set[y] for y in file_names], 
        index_summaries=index_summaries,
        service_context=service_context,
    )

    # [optional] save to disk
    graph.save_to_disk('index/GraphIndex.json')


# createGraph('./content')

def askBot(directory):
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    graph = ComposableGraph.load_from_disk(
        'index/GraphIndex.json', 
        service_context=service_context,
    )  
    query_configs = [
        {
            "index_struct_type": "dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 1,
                # "include_summary": True
            }
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "response_mode": "tree_summarize",
            }
        },
    ]
    
    while True:
        text_input = input("User: ")
        response = graph.query(text_input,query_configs=query_configs)
        print(f'Agent: {response}')
    
askBot('./content')