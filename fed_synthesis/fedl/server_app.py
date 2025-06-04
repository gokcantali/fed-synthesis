from typing import List, Tuple, Optional

from flwr.common import Context, Metrics, Scalar, NDArrays
from flwr.server import ServerAppComponents, ServerConfig, ServerApp

from fed_synthesis.core.agg_strategy import FedAvgCF, CF_METHODS
from fed_synthesis.core.client_selection import SimpleClientManagerWithPrioritizedSampling
from fed_synthesis.core.mlops_client import MLOpsClient
from fed_synthesis.core.util import initialize_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    aggregated_metrics = {}
    total_examples = 0

    for num_examples, m in metrics:
        total_examples += num_examples
        for key, value in m.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0

            # multiply metric of each client by number of examples used
            aggregated_metrics[key] += num_examples * value

    # divide by total number of examples to get weighted average
    for metric in aggregated_metrics:
        aggregated_metrics[metric] /= total_examples

    # return aggregated metrics in the weighted average form
    return aggregated_metrics


NUM_ROUNDS = 10
METHOD = "simple_avg"
RUN_NAME = "1st Run"
EXPERIMENT_NAME = "TEST-EXP"

PARAMS = {
    "url": "http://localhost:8080",
    "experiment_name": EXPERIMENT_NAME,
    "run_name": RUN_NAME
}

mlops_client = MLOpsClient(backend="MLFlow", params=PARAMS)
mlops_client.start_experiment()

current_training_round = 0

def training_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global current_training_round
    current_training_round += 1

    total_emission = 0.0
    total_samples = 0
    other_metrics_weighted_total = {}

    for num_examples, m in metrics:
        total_emission += m["carbon"]
        for field in m:
            if field != "carbon":
                if field not in other_metrics_weighted_total:
                    other_metrics_weighted_total[field] = 0.0
                other_metrics_weighted_total[field] += num_examples * m[field]
        total_samples += num_examples

    with open("carbon_emissions.txt", "a") as f:
        f.write(f"{total_emission}\n")

    other_metrics_weighted_average = {
        field: other_metrics_weighted_total[field] / total_samples
        for field in other_metrics_weighted_total
    }

    return {"total_emission": total_emission, **other_metrics_weighted_average}


def log_model_params_and_metrics_to_mlops(
    params: Optional[NDArrays] = None,
    metrics_plus_hyperparams: Optional[dict[str, Scalar]] = None
):
    global EXPERIMENT_NAME, RUN_NAME, NUM_ROUNDS, current_training_round
    with mlops_client.start_run():
        if metrics_plus_hyperparams:
            # record hyperparams along with metrics
            hyperparam_prefix = "hp:"

            hyperparams = {}
            metrics = {}
            for metric, value in metrics_plus_hyperparams.items():
                if metric.startswith(hyperparam_prefix):
                    hyperparams[metric[len(hyperparam_prefix):]] = value
                else:
                    metrics[metric] = value

            mlops_client.log_metrics(metrics, round=current_training_round)
            if current_training_round == 1:
                # log the hyperparams only once at the beginning
                mlops_client.log_hyperparams(hyperparams)

        # if params and current_training_round == NUM_ROUNDS:
        #     # log the model only once at the end
        #     model = initialize_model()
        #     model.set_parameters(params, None, False)
        #     mlflow.pytorch.log_model(
        #         pytorch_model=model,
        #         artifact_path=EXPERIMENT_NAME,
        #         registered_model_name=f"{EXPERIMENT_NAME}-{RUN_NAME}"
        #     )

strategy = FedAvgCF(
    fraction_fit=0.6,  # Sample 60% of available clients for training
    fraction_evaluate=1,  # Sample 100% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 3 clients for training
    min_evaluate_clients=3,  # Never sample less than 3 clients for evaluation
    min_available_clients=3,  # Wait until all 3 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # Use weighted average as custom metric evaluation function
    fit_metrics_aggregation_fn=training_metrics_aggregation,  # Use custom function to report carbon emissions
    alpha=0.5,
    window=5,
    method=METHOD,
    total_rounds=NUM_ROUNDS,
    log_params_and_metrics_fn=log_model_params_and_metrics_to_mlops
)

# Configure the server for <NUM_ROUNDS> rounds of training
config = ServerConfig(num_rounds=NUM_ROUNDS)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g. the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    global NUM_ROUNDS, RUN_NAME, METHOD

    # add a new subheader in the carbon emission file
    # before the next simulation starts
    with open("carbon_emissions.txt", "a") as f:
        subheader = "\n====== "
        subheader += "WITHOUT OPTIMIZATION" if METHOD == 'non_cf' else "WITH OPTIMIZATION"
        subheader += f" - {NUM_ROUNDS} Rounds - {CF_METHODS[METHOD]} Algorithm - {RUN_NAME}"
        subheader += " - CLOUD ======\n"
        f.write(subheader)

    return ServerAppComponents(
        config=config,
        strategy=strategy,
        client_manager=SimpleClientManagerWithPrioritizedSampling()
    )


# Create the ServerApp
app = ServerApp(server_fn=server_fn)
