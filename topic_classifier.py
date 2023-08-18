# Content: Embedding engine to create doc embeddings
# Author: Shuai Guo
# Email: shuai.guo@ch.abb.com
# Date: August, 2023


from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI, AzureOpenAI
import os


class TopicClassifier:
    """Determine relevant topics to the given article."""

    def __init__(self, issue, page_num, engine='Azure'):
        """Specify LLM model.

        Args:
        --------------
        engine: the LLM model. 
        issue: the issue name of the journal.
        page_num: starting & ending page number of the target article
        """

        self.issue = issue
        self.page_num = page_num

        if engine == 'OpenAI':
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.8
            )
        
        elif engine == 'Azure':
            self.llm = AzureOpenAI(
                    deployment_name="deployment-5af509f3323342ee919481751c6f8b7d",
                    model_name="text-davinci-003",
                    openai_api_base="https://abb-chcrc.openai.azure.com/",
                    openai_api_version="2023-03-15-preview",
                    openai_api_key=os.environ["OPENAI_API_KEY"],
                    openai_api_type="azure",
                )

        else:
            raise KeyError("Currently unsupported chat model type!")
    


    def classifier(self, topic_list, verbose=False):
        """Determine relevant topics in the given topic list.

        Args:
        --------------
        topic_list: a list of potential topics.
        verbose: boolean, if classifying progress is shown.
        """

        # Process article
        documents = self._split_article()

        # Loop over individual paragraphs
        llm_response = []
        for i, doc in enumerate(documents):
            if i%2==0 and verbose:
                print(f"Processing {i+1}/{len(documents)}th docs.")
            
            response = self.llm.predict(self.prompt.format(text=doc.page_content, topics=topic_list))
            llm_response.append(response)

        # Majority vote 
        results = {}
        for topic in topic_list[:-1]:
            results[topic] = 0

            for response in llm_response:
                if topic in response:
                    results[topic] += 1

        # Determine relevant topics
        relevant_topics = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

        return relevant_topics



    def _split_article(self):
        """Split the article into multiple paragraphs.
        """

        # Load article
        loader = PyMuPDFLoader("./papers/"+self.issue+".pdf")
        raw_documents = loader.load()[self.page_num[0]:self.page_num[-1]+1]

        # Split article
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_documents)

        return documents



    def _specify_prompt(self):
        """Specify LLM prompt for topic classification.
        """
        template = """Given the following text, output which focal points from the following list are most relevant.

                [text]: {text} \n
                [Focal points]: {topics}
                """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["text", "topics"],
        )


        

    