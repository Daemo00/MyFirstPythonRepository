from django.contrib import admin

from .models.question import Question
from .models.choice import Choice

admin.site.register(Question)
admin.site.register(Choice)
