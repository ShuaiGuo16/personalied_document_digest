# Content: Embedding engine to create doc embeddings
# Author: Shuai Guo
# Email: shuai.guo@ch.abb.com
# Date: August, 2023


from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI
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
                model_name="gpt-4",
                temperature=0.8
            )
        
        elif engine == 'Azure':
            self.llm = AzureOpenAI(
                    deployment_name="deployment-5af509f3323342ee919481751c6f8b7d",
                    model_name="text-davinci-003",
                    openai_api_base="https://abb-chcrc.openai.azure.com/",
                    openai_api_version="2023-03-15-preview",
                    openai_api_key=os.environ["OPENAI_API_KEY_AZURE"],
                    openai_api_type="azure",
                )

        else:
            raise KeyError("Currently unsupported chat model type!")
    


    def classifier(self, topic_list, desired_max=5, verbose=False):
        """Determine relevant topics in the given topic list.

        Args:
        --------------
        topic_list: a list of potential topics (exclude "general review").
        desired_max: desired maximum score for measuring relevance.
        verbose: boolean, if classifying progress is shown.

        Outputs:
        --------
        relevant_topics: relevant topics and their relevancy scores
        """

        # Process article
        documents = self._split_article()

        # Generate prompt for summary LLM
        prompt = self._specify_prompt()

        # Loop over individual paragraphs
        llm_response = []
        for i, doc in enumerate(documents):
            if i%2==0 and verbose:
                print(f"Processing {i+1}/{len(documents)}th docs.")
            
            response = self.llm.predict(prompt.format(text=doc.page_content, topics=topic_list))
            llm_response.append(response)

        
        # Majority vote 
        results = {topic: {'vote': 0, 'reason': []} for topic in topic_list}
        for topic in topic_list:
            for response in llm_response:
                parse_response = response.split('\n')
                for item in parse_response:
                    if topic in item:
                        results[topic]['vote'] += 1
                        results[topic]['reason'].append(item.split(':')[-1].strip())

        # Determine relevant topics
        relevant_topics = dict(sorted(results.items(), key=lambda item: item[1]['vote'], reverse=True))

        # Normalize relevant scores
        relevant_topics = self._normalize_score(relevant_topics, desired_max)


        return relevant_topics
    




    def _split_article(self):
        """Split the article into multiple paragraphs.
    
        Outputs:
        --------
        documents: splitted chunks of the given article.
        """

        # Load article
        loader = PyMuPDFLoader("./papers/"+self.issue)
        raw_documents = loader.load()[self.page_num[0]:self.page_num[-1]+1]

        # Split article
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_documents)

        return documents



    def _specify_prompt(self):
        """Specify LLM prompt for topic classification.

        Outputs:
        --------
        prompt: prompt for the summarization engine.
        """
        template = """Given the following text, identify which focal points from the following list are most relevant and 
        provide a reason for each selection in the format of "topic: reason".

        [text]: {text} \n
        [Focal points]: {topics}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["text", "topics"],
        )

        return prompt
    


    def _normalize_score(self, relevant_topics, desired_max):
        """Scale the relevance scores to the desired range.

        Args:
        --------
        relevant_topics: dict, relevant topics and their associated relevance score.
        desired_max: desired maximum score for measuring relevance.
        """

        # Get min & max
        original_max = max(item['vote'] for item in relevant_topics.values())
        original_min = min(item['vote'] for item in relevant_topics.values())

        # Normalize vote
        normalized_scores = {}
        for key, item in relevant_topics.items():
            normalized_vote = ((item['vote'] - original_min) / (original_max - original_min)) * desired_max
            normalized_scores[key] = {'vote': normalized_vote, 'reason': item['reason']}

        return normalized_scores




        

    