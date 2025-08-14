import os
from dotenv import load_dotenv
from pymilvus import connections, list_collections, Collection
load_dotenv()


from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Milvus
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class RAG():
    def __init__(self,
                 docs_dir: str,
                 n_retrievals: int = 4,
                 chat_max_tokens: int = 3097,
                 model_name = "tiiuae/falcon-7b-instruct",
                 creativeness: float = 0.7):
        self.__model = self.__set_llm_model(model_name, creativeness)
        self.__docs_list = self.__get_docs_list(docs_dir)
        self.__retriever = self.__set_retriever(k=n_retrievals)
        self.__chat_history = self.__set_chat_history(max_token_limit=chat_max_tokens)

    def __set_llm_model(self, model_name="tiiuae/falcon-7b-instruct", temperature: float = 0.7):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=temperature)
        return HuggingFacePipeline(pipeline=pipe)
    
    def __get_docs_list(self, docs_dir: str) -> list:
        print("Loading documents...")
        loader = DirectoryLoader(docs_dir,
                                 recursive=True,
                                 show_progress=True,
                                 use_multithreading=True,
                                 max_concurrency=4)
        docs_list = loader.load_and_split()
       
        return docs_list
    
    def __set_retriever(self, k: int = 4):
        # Local embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"  # Small & fast
        )
        # Commented below statement as i manually started milvus
        # milvus_server.start()
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT")
        )

        # List all collections
        collections = list_collections()
        print("Collections in Milvus:", collections)

        vector_store = Milvus.from_documents(
            self.__docs_list,
            embedding=embeddings,
            connection_args={"host": os.getenv("MILVUS_HOST"), "port": os.getenv("MILVUS_PORT")},
            collection_name="personal_documents",
        )
        print("Collections in Milvus:", collections)

        # Check if exists
        if "personal_documents" in list_collections():
            col = Collection("personal_documents")
            print("Schema:", col.schema)
            print("Description:", col.description)
        else:
            print("Collection not found.")

        
        # Self-Querying Retriever
        metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The directory path where the document is located",
                type="string",
            ),
        ]

        document_content_description = "Personal documents"

        _retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        return _retriever

    def __set_chat_history(self, max_token_limit: int = 3097):
        return ConversationTokenBufferMemory(llm=self.__model, max_token_limit=max_token_limit, return_messages=True)
    
    def ask(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant responsible for answering questions about documents. Respond to the user's question with a reasonable level of detail and based on the following context document(s):\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
       
        output_parser = StrOutputParser()
        chain = prompt | self.__model | output_parser
        answer = chain.invoke({
            "input": question,
            "chat_history": self.__chat_history.load_memory_variables({})['history'],
            "context": self.__retriever.get_relevant_documents(question)
        })

        # Atualização do histórico de conversa
        self.__chat_history.save_context({"input": question}, {"output": answer})
       
        return answer