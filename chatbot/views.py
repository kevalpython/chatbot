from django.shortcuts import render, redirect
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from .serializers import MessageSerializer, DummySerializer
from .llm import LLM
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import requests
from bs4 import BeautifulSoup
import random
from concurrent.futures import ThreadPoolExecutor
import os
def generate_20_digit_number():
    return ''.join([str(random.randint(0, 9)) for _ in range(20)])

def extract_urls(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        urls = [link['href'] for link in links if link['href'].startswith(('http', 'https', '/'))]
        return urls
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

def fetch_url(url):
    """Fetch a single URL using SeleniumURLLoader."""
    try:
        loader = SeleniumURLLoader(urls=[url])
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

class URLInputView(GenericAPIView):
    serializer_class = DummySerializer

    @method_decorator(login_required)
    def get(self, request, **kwargs):
        return render(request, 'home.html')

    @method_decorator(login_required)
    def post(self, request, **kwargs):
        url = request.data.get('url')
        if not url:
            return Response({'error': 'No URL provided'}, status=400)

        try:
            all_urls = set(extract_urls(url))
            full_urls = []
            for i in all_urls:
                if i.startswith('/'):
                    i = url + i.replace('/', '')
                full_urls.append(i)

            if url not in all_urls:
                all_urls.add(url)

            all_docs = []

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {executor.submit(fetch_url, link): link for link in full_urls}
                for future in future_to_url:
                    result = future.result()
                    if result:
                        all_docs.extend(result)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=HuggingFaceEndpointEmbeddings(
                    repo_id="sentence-transformers/all-mpnet-base-v2",
                    huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"]
                )
            )
            retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

            LLM.retriever[request.user] = retriever
            return redirect('chatbot', user=request.user.username)
        except Exception as e:
            return Response({'error': str(e)}, status=400)

@method_decorator(csrf_exempt, name='dispatch')
class ChatView(GenericAPIView):
    serializer_class = MessageSerializer

    @method_decorator(login_required)
    def get(self, request, **kwargs):
        return render(request, 'chat.html')

    @method_decorator(login_required)
    def post(self, request, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        message = serializer.validated_data['text']
        llm = LLM(user=request.user)
        response_data = llm.get_response(message)
        return Response(response_data)
