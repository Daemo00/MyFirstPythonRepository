from django.test import LiveServerTestCase
from selenium import webdriver

from .common import create_question


class SeleniumTests(LiveServerTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.selenium = webdriver.Chrome()

    @classmethod
    def tearDownClass(cls):
        cls.selenium.quit()
        super().tearDownClass()

    def test_navigate_to_question(self):
        question = create_question(question_text='Question', days=-1)
        self.selenium.get('%s%s' % (self.live_server_url, '/polls/'))
        question_link = self.selenium.find_element_by_xpath('//a')
        question_link.click()
        self.assertEqual(self.selenium.title, str(question))
