# Content: Class definition of interactive chatbot system 
# Author: Shuai Guo
# Email: shuai.guo@ch.abb.com
# Date: August, 2023


from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from abc import ABC, abstractmethod


class Chatbot(ABC):
    """Class definition for a single chatbot with memory, created with LangChain."""
    
    
    def __init__(self, engine='Azure'):
        """Initialize the large language model and its associated memory.
        The memory can be an LangChain emory object, or a list of chat history.

        Args:
        --------------
        engine: the backbone llm-based chat model.
                "OpenAI" stands for OpenAI chat model;
                Other chat models are also possible in LangChain, 
                see https://python.langchain.com/en/latest/modules/models/chat/integrations.html
        """
        
        # Instantiate llm
        if engine == 'OpenAI':
            self.llm = ChatOpenAI(
                model_name='gpt-4',
                temperature=0.8
            )

        elif engine == 'Azure':
            self.llm = AzureChatOpenAI(openai_api_base="https://abb-chcrc.openai.azure.com/",
                    openai_api_version="2023-03-15-preview",
                    openai_api_key=os.environ["OPENAI_API_KEY_AZURE"],
                    openai_api_type="azure",
                    deployment_name="gpt-35-turbo-0301",
                    temperature=0.8)

        else:
            raise KeyError("Currently unsupported chat model type!")


    @abstractmethod
    def instruct(self):
        """Determine the context of chatbot interaction. 
        """
        pass


    @abstractmethod
    def step(self):
        """Action produced by the chatbot. 
        """
        pass
        

    @abstractmethod
    def _specify_system_message(self):
        """Prompt engineering for chatbot.
        """       
        pass
    



class JournalistBot(Chatbot):
    """Class definition for the journalist bot, created with LangChain."""

    
    def __init__(self, engine):
        """Setup journalist bot.
        
        Args:
        --------------
        engine: the backbone llm-based chat model.
                "OpenAI" stands for OpenAI chat model;
                Other chat models are also possible in LangChain, 
                see https://python.langchain.com/en/latest/modules/models/chat/integrations.html
        """
        
        # Instantiate llm
        super().__init__(engine)
        
        # Instantiate memory
        self.memory = ConversationBufferMemory(return_messages=True)


    def instruct(self, theme, summary, focal_points, audience):
        """Determine the context of journalist chatbot. 
        
        Args:
        ------
        theme: the theme of the article
        summary: the summary of the paper
        focal_points: list, interested aspects to distill
        audience: list, target audience
        """
        
        self.theme = theme
        self.summary = summary
        self.focal_points = focal_points
        self.audience = audience
        
        # Define prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self._specify_system_message()),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("""{input}""")
        ])
        
        # Create conversation chain
        self.conversation = ConversationChain(memory=self.memory, prompt=prompt, 
                                              llm=self.llm, verbose=False)
        

    def step(self, prompt):
        """Journalist chatbot asks question. 
        
        Args:
        ------
        prompt: Previos answer provided by the author bot.
        """
        response = self.conversation.predict(input=prompt)
        
        return response
        


    def _specify_system_message(self):
        """Specify the behavior of the journalist chatbot.


        Outputs:
        --------
        prompt: instructions for the chatbot.
        """       
        
        # Base prompr 
        prompt = f"""You are a journalist interested in the news of the company ABB that are related to {self.theme}.

        Your task is to extract key information from an article in ABB's review journal through 
        an interview with the author, which is played by another chatbot.

        Your objective is to ask insightful questions based on the article's content and the interests and needs of 
        {self.audience}, so that {self.audience} who reads the interview can grasp the article's core without reading it in full.

        You're provided with the article's summary to guide your initial questions.

        You must keep the following guidelines in mind:
        - Always remember your role as the journalist.
        - Avoid general questions about {self.theme}, focusing instead on specifics related to the article.
        - Only ask one question at a time.
        - Do not include any prefixed labels like "Interviewer:" or "Question:" in your question.
        - Keep your questions focused, relevant, and succinct.
        """

        for focal_point in self.focal_points:
            prompt += '\n'
            prompt += focal_point

        prompt += '\n\n'
        prompt += f"""[Summary]: {self.summary}"""
        
        return prompt



class AuthorBot(Chatbot):
    """Class definition for the author bot, created with LangChain."""
    
    def __init__(self, engine, vectorstore, debug=False):
        """Select backbone large language model, as well as instantiate 
        the memory for creating language chain in LangChain.
        
        Args:
        --------------
        engine: the backbone llm-based chat model.
        vectorstore: embedding vectors of the paper.
        """
        
        # Instantiate llm
        super().__init__(engine)
        
        # Instantiate memory
        self.chat_history = []
        
        # Instantiate embedding index
        self.vectorstore = vectorstore

        self.debug = debug
        
        
        
    def instruct(self, theme, audience):
        """Determine the context of author chatbot. 
        
        Args:
        -------
        topic: the topic of the paper.
        audience: list, target audience
        """

        # Specify topic & audience
        self.theme = theme
        self.audience = audience

        
        # Define prompt template
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self._specify_system_message()),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        # Create conversation chain
        self.conversation_qa = ConversationalRetrievalChain.from_llm(llm=self.llm, verbose=self.debug,
                                                                     retriever=self.vectorstore.as_retriever(
                                                                         search_kwargs={"k": 5}),
                                                                    chain_type="stuff", return_source_documents=True,
                                                                    combine_docs_chain_kwargs={'prompt': qa_prompt})

        
        
    def step(self, prompt):
        """Author chatbot answers question. 
        
        Args:
        ------
        prompt: question raised by journalist bot.

        Outputs:
        ------
        answer: the author bot's answer
        source_documents: documents that author bot used to answer questions
        """
        response = self.conversation_qa({"question": prompt, "chat_history": self.chat_history})
        self.chat_history.append((prompt, response["answer"]))
        
        return response["answer"], response["source_documents"]
        
        
        
    def _specify_system_message(self):
        """Specify the behavior of the author chatbot.


        Outputs:
        --------
        prompt: instructions for the chatbot.
        """       
        
        # Compile bot instructions 
        prompt = f"""You are the author of a recently published article from ABB's review journal.
        You are being interviewed by a journalist who is played by another chatbot and
        aiming to extract insights beneficial for {self.audience}.
        Your task is to provide comprehensive, clear, and accurate answers to the journalist's questions.
        Please always consider the target audience, {self.audience}, and tailor your explanations in a manner that 
        is most relevant and understandable to them.

        Please keep the following guidelines in mind:
        - Always prioritize information directly from the article. If a question relates to content not covered in the article, be transparent about this.
        - If a direct answer isn't available in the article, you can draw upon your broader knowledge on the subject. 
        - In cases where even your broad knowledge doesn't cover the question, suggest additional resources or avenues where the answer might be found.
        - Always clarify when you're providing information directly from the article with phrases like 'According to the article...'. 
        - When providing broader context or interpreting the data, use terms like 'Based on general trends in the field...'.
        - Handle one question at a time, ensuring each response is complete before addressing the next inquiry.
        - Remember to always maintain the integrity and accuracy of the article's information, and if you're unsure or speculating, be transparent about it.
        - Do not include any prefixed labels like "Author:", "Interviewee:", Respond:", or "Answer:" in your answer.
        """
        
        prompt += """Given the following context, please answer the question.
        
        {context}"""
        
        return prompt