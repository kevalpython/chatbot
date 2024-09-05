from rest_framework import serializers
from .models import ChatHistory
class MessageSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=1000)

class DummySerializer(serializers.Serializer):
    url = serializers.URLField()
    
class ChatHistorySerializer(serializers.Serializer):
    class Meta:
        model = ChatHistory
        fields = (
            "user",
            "user_message",
            "assistant_response",
            "timestamp",
        )