from logging import log, WARNING, INFO
from typing import Optional

import numpy as np
from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class SimpleClientManagerWithPrioritizedSampling(SimpleClientManager):
    def sample_with_priority(
        self,
        num_clients: int,
        priorities: dict[str, float],
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        log(WARNING, f"Priorities: {priorities}")

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        prior_weights = []
        sum_weights = 0.0
        for cid in available_cids:
            priority = priorities.get(cid, -1)
            prior_weights.append(priority)
            if priority >= 0:
                sum_weights += priority
        for ind in range(len(prior_weights)):
            if prior_weights[ind] <= 0.0:
                prior_weights[ind] = sum_weights / len(prior_weights)

        prior_weights /= np.sum(prior_weights)

        sampled_cids = np.random.choice(
            available_cids, size=num_clients, replace=False, p=prior_weights
        )

        log(WARNING, f"Priority weights: {prior_weights}")
        log(WARNING, "Here are the selected clients:")
        for cid in sampled_cids:
            log(WARNING, cid)
        return [self.clients[cid] for cid in sampled_cids]
