"""Database layout, with additional metadata."""
import datetime

from django.db import models
from django.utils import timezone


class Question(models.Model):
    """A Question has a question and a publication date. """
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    def __str__(self):
        return self.question_text
    def was_published_recently(self):
        """Returns true iff this question was published in the last 24 hours"""
        now = timezone.now()
        return now - datetime.timedelta(days=1) < self.pub_date < now
    was_published_recently.admin_order_field = 'pub_date'
    was_published_recently.boolean = True
    was_published_recently.short_description = 'Published recently?'
    
class Choice(models.Model):
    """A Choice has two fields: the text of the choice and a vote tally.
    Each Choice is associated with a Question."""
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    def __str__(self):
        return self.choice_text
