import datetime

from django.utils import timezone

from ..models.question import Question


def create_question(question_text, days):
    """
    Create a question with the given `question_text` and published the
    given number of `days` offset to now (negative for questions published
    in the past, positive for questions that have yet to be published).
    """
    time = timezone.now() + datetime.timedelta(days=days)
    question = Question.objects.create(
        question_text=question_text, pub_date=time)
    question.choice_set.create(
        choice_text='Choice', votes=0)
    return question
