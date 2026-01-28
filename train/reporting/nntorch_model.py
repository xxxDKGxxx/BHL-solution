from train.reporting.model_interface import ModelInterface
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Self

class TorchModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TorchModelAdapter(ModelInterface):
    def __init__(self, model: TorchModel, lr: float = 1e-3, epochs: int = 10):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def fit(self, X_train: np.ndarray,
            y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
	        y_val: Optional[np.ndarray] = None) -> None:

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        X_val_t = None
        y_val_t = None

        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long)

        for epoch in range(self.epochs):
            self.model.train()

            preds = self.model(X_train_t)
            loss = self.criterion(preds, y_train_t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.history["train_loss"].append(loss.item())

            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_t)
                    val_loss = self.criterion(val_preds, y_val_t)

                    self.history["val_loss"].append(val_loss.item())

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()

        X_t = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(X_t)
            probs = torch.argmax(output, dim=1)

        return probs.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()

        X_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def get_loss_history(self) -> Dict[str, List[float]]:
        return self.history

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None

    def get_params(self) -> Dict[str, Any]:
        return {
            "lr" : self.lr,
            "epochs" : self.epochs,
            "optimizer" : self.optimizer.__class__.__name__,
            "loss": self.criterion.__class__.__name__,
            "model": self.model.__class__.__name__,
        }

    def get_new_instance(self) -> Self:
        new_model = TorchModel(self.model.input_dim)
        return TorchModelAdapter(
            model=new_model,
            lr=self.lr,
            epochs=self.epochs
        )
