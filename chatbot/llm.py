import os
import pytz
import nltk
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from .CustomChatModel import CustomChatModel
from .models import ChatHistory
from langchain_core.messages import HumanMessage, AIMessage

nltk.download('averaged_perceptron_tagger')
import requests

# Environment setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ['HUGGINGFACE_API_KEY'] = "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"

class LLM:
    retriever = {}

    def __init__(self, **kwargs):
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="conversational",
            do_sample=False,
            huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"],
            repetition_penalty=1.03,
        )
        self.llm = CustomChatModel(llm=self.llm, verbose=True)
        self.config = {"configurable": {"user": kwargs.get("user")}}

    def get_session_history(self, user):
        chat_history = ChatHistory.objects.filter(user=user).order_by('timestamp')
        formatted_history = InMemoryChatMessageHistory()

        for entry in chat_history:
            if entry.user_message:
                formatted_history.messages.append(HumanMessage(content=entry.user_message))
            if entry.assistant_response:
                formatted_history.messages.append(AIMessage(content=entry.assistant_response))
        return formatted_history

    def get_response(self, message):
        instruction_prompt = """
            Context: {context}
            User Input: {input}
            You are an assistant.
            - When the user greets you (e.g., "Hello"), respond only with "Hello, how can I assist you?" Do not mention any context or additional information.
            - When the user asks a specific question, first check the provided context.
            - If relevant information is found in the context, respond using that information without revealing the existence of the context.
            - If the context does not contain relevant information, simply say you don't know or ask how else you can assist.
            - Avoid providing any additional or unsolicited details unless the user explicitly asks about them.
            - Do not reveal to the user that you have access to additional context.
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", instruction_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        user = self.config['configurable']['user']
        retrievers = self.retriever.get(user, {})
        history_aware_retriever = create_history_aware_retriever(self.llm, retrievers, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        session_history_callable = lambda: self.get_session_history(user)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=session_history_callable,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        try:
            raw_response = conversational_rag_chain.invoke(
                {"input": message, "context": retrievers},
                config=self.config
            )["answer"]
            parser = StrOutputParser()
            answer = parser.parse(raw_response)
            print("answer : ", answer)
            chat_entry = ChatHistory.objects.create(
                user=user,
                user_message=message,
                assistant_response=answer,
            )

            timestamp = chat_entry.timestamp
            utc_timestamp = timestamp.replace(tzinfo=pytz.utc)
            ist_timezone = pytz.timezone('Asia/Kolkata')
            ist_timestamp = utc_timestamp.astimezone(ist_timezone)
            formatted_timestamp = ist_timestamp.strftime('%d-%m-%Y %H:%M:%S')

        except requests.exceptions.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except Exception as e:
            # Log detailed error information for debugging
            print(f"An unexpected error occurred: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"
        
        return {"answer": answer, "timestamp": formatted_timestamp}
