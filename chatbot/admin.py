from django.contrib import admin
from .models import User, ChatHistory
# Register your models here.

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ["username", "first_name", "email"]

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ["user", "user_message", "assistant_response","timestamp"]