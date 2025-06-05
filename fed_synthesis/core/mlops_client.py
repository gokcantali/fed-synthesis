import mlflow
from codecarbon import EmissionsTracker as CodeCarbonEmissionsTracker


class MLOpsClient:
    def __init__(self, backend="MLFlow", params: dict = {}):
        if backend not in ["MLFlow"]:
            raise ValueError("Unsupported backend. Currently, only 'MLFlow' is supported.")

        if (url := params.get("url")) is None:
            raise ValueError("URL must be provided in params.")

        if (experiment_name := params.get("experiment_name")) is None:
            raise ValueError("Experiment name must be provided in params.")

        # Default to 1st Run if not specified
        run_name = params.get("run_name", "1st Run")

        self.mlops_api = mlflow
        self.url = url
        self.experiment_name = experiment_name
        self.run_name = run_name

        self.experiment_id = None
        self.run_id = None

    def start_experiment(self):
        """Start an MLOps experiment"""
        self.mlops_api.set_tracking_uri(self.url)

        self.experiment_id = self.mlops_api.set_experiment(
            experiment_name=self.experiment_name
        ).experiment_id

        self.run_id = self.mlops_api.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name
        ).info.run_id

        self.mlops_api.end_run()

    def start_run(self):
        """Start an MLOps run"""
        return self.mlops_api.start_run(run_id=self.run_id)

    def log_metrics(self, metrics: dict, round: int):
        """Log metrics to MLOps"""
        # if self.run_id is None:
        #     raise ValueError("Run ID is not set. Please start an experiment first.")
        #
        # with self.mlops_api.start_run(run_id=self.run_id):

        self.mlops_api.log_metrics(metrics, step=round)

    def log_hyperparams(self, hyperparams: dict):
        """Log hyperparameters to MLOps"""
        self.mlops_api.log_params(hyperparams)

    def log_model(self, model, backend="torch"):
        """Log the ML model to MLOps"""
        if backend not in ["torch", "sklearn"]:
            raise ValueError("Unsupported backend. Currently, only 'torch' and 'sklearn' are supported.")

        model_name = f"{self.experiment_name}-{self.run_name}"
        if backend == "torch":
            self.mlops_api.pytorch.log_model(
                pytorch_model=model,
                artifact_path=self.experiment_name,
                registered_model_name=model_name
            )
        elif backend == "sklearn":
            self.mlops_api.sklearn.log_model(
                sk_model=model,
                artifact_path=self.experiment_name,
                registered_model_name=model_name
            )
