import os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from train.reporting.model_interface import ModelInterface
from train.reporting.model_reporter import ModelReporter

from datetime import datetime


class FactClassifierReporter(ModelReporter):
    def __init__(
            self,
            model_wrapper: ModelInterface,
            X,
            y,
            base_output_dir="reports",
            scoring = None,
    ):
        super().__init__(model_wrapper=model_wrapper, X=X, y=y, base_output_dir=base_output_dir)
        self.scoring = scoring or [
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "roc_auc"
        ]
        self.cv_splits = 5
        self.cv_repeats = 3
        self.random_state = 42

    def evaluate_model_cv(self) -> pd.DataFrame:
        self._log(
            f"\n--- Repeated Stratified CV: "
            f"{self.cv_splits} foldów x {self.cv_repeats} powtórzeń ---"
        )

        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_splits,
            n_repeats=self.cv_repeats,
            random_state=self.random_state,
        )

        estimator = self.wrapper.model

        res = cross_validate(
            estimator=estimator,
            X=self.X,
            y=self.y,
            scoring=self.scoring,
            cv=cv,
            return_train_score=True,
            error_score="raise",
        )

        res_df = pd.DataFrame(res)

        summary_df = res_df.agg(['mean', 'std'])

        final_df = pd.concat([res_df, summary_df]).T

        self._log("\n--- Wyniki walidacji krzyżowej ---")

        for metric_name in final_df.index:
            if 'time' in metric_name:
                continue

            mean_val = final_df.loc[metric_name, 'mean']
            std_val = final_df.loc[metric_name, 'std']

            self._log(f"{metric_name}: {mean_val:.4f} (+/- {std_val:.4f})")

        if self.current_report_dir:
            save_path = os.path.join(
                self.current_report_dir,
                'fact_classifier_results.csv',
            )
            final_df.to_csv(save_path)
            self._log(f'Zapisano wyniki CV: {save_path}')

        return final_df

    def generate_report(self) -> pd.DataFrame:
        self._setup_directories()

        self._log(f"Raport wygenerowany: {datetime.now()}")
        self._log(f"Model Wrapper: {self.wrapper.__class__.__name__}")
        self._log("-" * 30)

        self.run_training()
        self.plot_loss_history()
        self.plot_confusion_matrix()

        self.evaluate_model_cv()

        self.save_feature_importance()
        self.plot_top_2_features_boundary()
        self.save_model_and_datasets()

        print(f"\n[SUKCES] Cały raport zapisany w folderze: {self.current_report_dir}")
