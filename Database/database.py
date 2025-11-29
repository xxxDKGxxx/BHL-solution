from typing import override

from Database.database_context import DatabaseContext


class Database(DatabaseContext):
    def __init__(self, embedder):
        self.embedder = embedder

    @override
    def get(self):
        pass

    @override
    def insert(self, ):
        pass

