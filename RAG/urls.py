from django.urls import path
from . import views

urlpatterns = [
    path("", views.upload_pdf, name="upload"),
    path("upload_document/", views.upload_document, name="upload_document"),
    path("chat/", views.chat, name="chat"),
    path("rag/", views.rag_tool, name="rag_tool"),
]