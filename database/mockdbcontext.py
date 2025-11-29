from interface.abstractdatabasecontext import DatabaseContext

class MockDbContext(DatabaseContext):
    def __init__(self):
        self._store: dict[str, str] = [1,2,3,4,5,6,7]

    def get(self, prompt: str) -> tuple[str, float]:
        return self._store[0]

    def insert(self, answer: str) -> bool:
        self._store.append(answer)






