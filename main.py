from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from transformers import BitsAndBytesConfig
import os
import gc
import asyncio
import requests
import re
import contextlib


# define the model names and index path
embeded_model_name = "Qwen/Qwen3-Embedding-0.6B"
llm_model_name = "qwen3:8b"
index_path = "./index_store"


# quantization configuration for the embedding model
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0    
)

# init the embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name=embeded_model_name,
    model_kwargs={
        "quantization_config": bnb_config,  
        "device_map": "auto",      
        "local_files_only": True    
    }
)
# print(f"Embedding model set to {embeded_model_name}")


# init the LLM model
Settings.llm = Ollama(
    model=llm_model_name,
    request_timeout=360.0,
    context_window=8000,
)
# print(f"LLM model set to {llm_model_name}")




def initialize_index():
    documents = SimpleDirectoryReader("books").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_path)
    print(f"索引已建立並儲存到 {index_path}")


# Check if the index already exists
if os.path.isdir(index_path) and os.listdir(index_path):
    pass
else:
    print("索引不存在，正在建立索引...")
    initialize_index()


# Load the index from storage
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

with suppress_output():
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()




# Define the search tool for the agent
async def search_novel(query: str) -> str:
    """可以用來搜尋三國演義相關資料的工具。"""
    response = await query_engine.aquery(query)
    return str(response)


# Define a function to unload the Ollama model
def unload_ollama_model(model_name: str, host: str = "http://127.0.0.1:11434"):

    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": "",       
        "keep_alive": 0,    
        "stream": False
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    print(f"已卸載 Ollama 模型：{model_name}")




# Create an enhanced workflow with both tools
agent = AgentWorkflow.from_tools_or_functions(
    [search_novel],
    llm=Settings.llm,
    system_prompt="""你是一個AI助手。如果問題是關於三國演義相關資料，直接使用search_novel，請用繁體中文回答。""",

)



# Define a function to strip <think> sections from the text
def strip_think_sections(text) -> str:
    if not isinstance(text, (str, bytes)):
        text = str(text)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned).strip()
    return cleaned



async def main():
    print("\n歡迎使用三國演義查詢助手！")
    
    while True:
        print("--------------")
        user_prompt = input("請輸入你要問的內容（輸入 exit 結束）：")
        if user_prompt.strip().lower() == "exit":
            unload_ollama_model(llm_model_name)
            Settings.embed_model = None
            gc.collect()
            print("結束對話，掰掰！")
            break
        response = await agent.run(user_prompt)
        print("輸出: ",strip_think_sections(response))
        print("--------------")

if __name__ == "__main__":
    asyncio.run(main())




