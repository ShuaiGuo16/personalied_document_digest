# Content: Embedding engine to create doc embeddings
# Author: Shuai Guo
# Email: shuai.guo@ch.abb.com
# Date: August, 2023


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import os


class Embedder:
    """Embedding engine to create doc embeddings."""

    def __init__(self, engine='Azure'):
        """Specify embedding model.

        Args:
        --------------
        engine: the embedding model. 
                For a complete list of supported embedding models in LangChain, 
                see https://python.langchain.com/docs/integrations/text_embedding/
        """
        if engine == 'OpenAI':
            self.embeddings = OpenAIEmbeddings()
        
        elif engine == 'Azure':
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", 
                              deployment="text-embedding-ada-002",
                              openai_api_key=os.environ["OPENAI_API_KEY_AZURE"],
                              openai_api_base="https://abb-chcrc.openai.azure.com/",
                              openai_api_type="azure",
                              chunk_size=1)
        
        else:
            raise KeyError("Currently unsupported chat model type!")
        


    def load_n_process_document(self, issue, page, chunk_size=5000, debug=False):
        """Load and process PDF document.

        Args:
        --------------
        issue: path of the journal issue.
        page: dict, start page number and length of the article in the issue.
        """

        # Load PDF
        loader = PyMuPDFLoader(issue)
        documents = loader.load()[page['start']-1: page['start']+page['length']-1]

        # Process PDF
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        self.documents = text_splitter.split_documents(documents)

        if debug:
            return documents



    def create_vectorstore(self, store_path):
        """Create vector store for doc Q&A.
           For a complete list of vector stores supported by LangChain,
           see: https://python.langchain.com/docs/integrations/vectorstores/

        Args:
        --------------
        store_path: path of the vector store.

        Outputs:
        --------------
        vectorstore: the created vector store for holding embeddings
        """
        if not os.path.exists(store_path):
            print("Embeddings not found! Creating new ones")
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            self.vectorstore.save_local(store_path)

        else:
            print("Embeddings found! Loaded the computed ones")
            self.vectorstore = FAISS.load_local(store_path, self.embeddings)

        return self.vectorstore
    


    def create_summary(self, summary_method='stuff', llm_engine='Azure'):
        """Create paper summary. The summary is created by using LangChain's summarize_chain.

        Args:
        --------------
        llm_engine: backbone large language model.

        Outputs:
        --------------
        summary: the summary of the paper
        """

        if llm_engine == 'OpenAI':
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.8
            )
        
        elif llm_engine == 'Azure':
            llm = AzureChatOpenAI(openai_api_base="https://abb-chcrc.openai.azure.com/",
                openai_api_version="2023-03-15-preview",
                openai_api_key=os.environ["OPENAI_API_KEY_AZURE"],
                openai_api_type="azure",
                deployment_name="gpt-35-turbo-0301",
                temperature=0.8)

        else:
            raise KeyError("Currently unsupported chat model type!")
        
        chain = load_summarize_chain(llm, chain_type=summary_method)
        summary = chain.run(self.documents[:10])

        return summary
    