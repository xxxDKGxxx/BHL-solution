from train.reporting.model_interface import ModelInterface
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import numpy as np

class SVMModelWrapper(ModelInterface):
    def __init__(self, vec_name: str, vectorizer, C=1.0):
        self.model = Pipeline([
            (vec_name, vectorizer),
            ('svm', SVC(C=C, random_state=42, probability=True, kernel='linear'))
        ])
        self.C = C
        self.is_fitted = False
        self.vectorizer = vectorizer
        self.vec_name = vec_name

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_loss_history(self):
        return {}

    def get_params(self):
        return self.model.get_params()

    def get_vectorizer(self):
        return self.vectorizer

    def get_new_instance(self):
        return SVMModelWrapper(self.vec_name, clone(self.vectorizer), C=self.C)

    def get_feature_importance(self):
        if not self.is_fitted:
            return {}

        try:
            feature_names = self.model.named_steps[self.vec_name].get_feature_names_out()
            coefs = self.model.named_steps['svm'].coef_.copy()
            avg_coefs = np.mean(np.abs(coefs), axis=0)
            avg_coefs = np.ravel(avg_coefs)

            importance_dict = dict(zip(feature_names, avg_coefs))
            return importance_dict
        except Exception as e:
            print(f"Nie udało się pobrać ważności cech: {e}")
            return {}