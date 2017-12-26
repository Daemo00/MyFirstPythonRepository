from django.contrib import admin

from .models.question import Question
from .models.choice import Choice


class ChoiceInline(admin.TabularInline):
    model = Choice
    extra = 3


class QuestionAdmin(admin.ModelAdmin):
    fieldsets = [
        (None,
         {'fields': ['question_text']}),
        ('Date information',
         {'fields': ['pub_date'],
          'classes': ['collapse']})
    ]
    inlines = [ChoiceInline]
    list_display = ('question_text', 'pub_date', 'was_published_recently', 'choices_number')
    list_filter = ['pub_date']
    search_fields = ['question_text']

    @staticmethod
    def choices_number(obj):
        return obj.choice_set.count()

admin.site.register(Question, QuestionAdmin)
