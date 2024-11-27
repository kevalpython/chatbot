from django.urls import path
from chatbot import views
urlpatterns = [
    path('', views.URLInputView.as_view(), name='url_input'),
    path('chatbot/<str:user>', views.ChatView.as_view(), name='chatbot'),

]
