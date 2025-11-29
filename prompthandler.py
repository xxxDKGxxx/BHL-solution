from abc import ABC
from Database.database import DatabaseContext


class PromptHandler():



    def __init__(self,db_context: DatabaseContext):
        self.db_context = db_context


    def preprocess(self):
        pass


    def generate_answer(self):
        pass

