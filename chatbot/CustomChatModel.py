from langchain_huggingface import ChatHuggingFace

class CustomChatModel(ChatHuggingFace):
    def _generate(self,messages,stop,run_manager,**kwargs):
        kwargs['max_tokens']=20000
        return super()._generate(messages,stop,run_manager,**kwargs)