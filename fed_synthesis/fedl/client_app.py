from dataclasses import asdict

from flwr.client import NumPyClient, Client, ClientApp
from flwr.common import Context, Config, Scalar

from fed_synthesis.core.carbon_tracker_client import CarbonTrackerClient
from gnn_example.fedl_setup import initialize_model, get_dataset_splits


class FlowerClient(NumPyClient):
    def __init__(self, net, train_set, val_set, test_set, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        return self.get_context().node_config

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
        tracker = CarbonTrackerClient(backend="CodeCarbon")
        self.net.set_parameters(parameters, config, is_evaluate=False)
        tracker.start_tracking()
        metrics = self.net.train_model(self.train_set, self.val_set, batch_mode=True, epochs=1)
        emissions = tracker.stop_tracking()
        # self.emissions = emissions if not math.isnan(emissions) else emissions

        metrics_to_aggregate = {
            "carbon": emissions
        }
        for metric, values in asdict(metrics).items():
            metrics_to_aggregate[metric] = values[-1]

        # Include hyperparameters in the metrics to aggregate
        metrics_to_aggregate["learning_rate"] = self.net.scheduler.get_last_lr()[0]
        for initial_hp, hp_value in self.net.hyperparams.items():
            metrics_to_aggregate[f"hp:{initial_hp}"] = hp_value
        metrics_to_aggregate["hp:epochs"] = 1

        return (
            self.net.get_parameters(),
            len(self.train_set),
            metrics_to_aggregate
        )

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters, config, is_evaluate=True)
        _, loss, perf_metrics = self.net.test_model_batch_mode(self.test_set)
        print("METRICS OF CLIENT:")
        print(perf_metrics)
        return loss, len(self.test_set), perf_metrics


def construct_flower_client(client_id, context):
    # Load model
    net = initialize_model()

    # Note: each client gets a different train/validation/test datasets,
    # so each client will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    train_set, validation_set, test_set = get_dataset_splits(
        client_id
    )

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    flower_client = FlowerClient(
        net, train_set, validation_set, test_set,
    )
    flower_client.set_context(context)
    return flower_client.to_client()


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    partition_id = context.node_config["partition-id"]

    # Construct the client
    flower_client = construct_flower_client(
        client_id=partition_id, context=context
    )
    return flower_client


# Create the ClientApp
app = ClientApp(
    client_fn=client_fn,
)
