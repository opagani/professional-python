import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, space_eval, Trials


class StepwiseHyperoptOptimizer(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        model,
        param_space_sequence,
        max_evals_per_step=100,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    ):
        self.model = model
        self.param_space_sequence = param_space_sequence
        self.max_evals_per_step = max_evals_per_step
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = {}
        self.best_score_ = None

    def clean_int_params(self, params):
        int_vals = ["max_depth", "reg_alpha"]
        return {k: int(v) if k in int_vals else v for k, v in params.items()}

    def objective(self, params):
        # I added this
        params = self.clean_int_params(params)
        # END
        current_params = {**self.best_params_, **params}
        self.model.set_params(**current_params)
        score = cross_val_score(
            self.model, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=-1
        )
        return -np.mean(score)

    def fit(self, X, y):
        self.X = X
        self.y = y

        for step, param_space in enumerate(self.param_space_sequence):
            print(f"Optimizing step {step + 1}/{len(self.param_space_sequence)}")

            trials = Trials()
            best = fmin(
                fn=self.objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=self.max_evals_per_step,
                trials=trials,
                # rstate=np.random.RandomState(self.random_state)
            )

            step_best_params = space_eval(param_space, best)
            # I added this
            step_best_params = self.clean_int_params(step_best_params)
            # END
            self.best_params_.update(step_best_params)
            self.best_score_ = -min(trials.losses())

            print(f"Best parameters after step {step + 1}: {self.best_params_}")
            print(f"Best score after step {step + 1}: {self.best_score_}")

        # Fit the model with the best parameters
        self.model.set_params(**self.best_params_)
        self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
