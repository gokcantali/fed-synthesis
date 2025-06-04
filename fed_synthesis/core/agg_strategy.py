import math
import traceback
from logging import log, INFO, WARNING, ERROR
from typing import Optional, Union

import numpy as np
import pandas as pd
from flwr.common import EvaluateRes, Scalar, Parameters, FitIns, FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.linear_model import LinearRegression

from fed_synthesis.core.client_selection import SimpleClientManagerWithPrioritizedSampling
from fed_synthesis.core.util import exponential_smoothing

CF_METHODS = {
    "simple_avg": "CF_SimpleAvg",
    "exp_smooth": "CF_ExpSmooth",
    "lin_reg": "CF_LinRegress",
    "non_cf": "NON_CF"
}

class FedAvgCF(FedAvg):
    def __init__(
        self, alpha: float, window: int,
        method: str = "lin_reg", total_rounds: int = 60,
        log_params_and_metrics_fn: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.window = window

        if method not in CF_METHODS:
            raise Exception(f"Method: {method} not implemented!")
        self.method = method

        self.total_rounds = total_rounds
        self.log_params_and_metrics_fn = log_params_and_metrics_fn

        self.emission_mapping: dict[str, dict[int, float]] = {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        loss_agg, metrics_agg = super().aggregate_evaluate(
            server_round=server_round, results=results, failures=failures
        )

        if self.log_params_and_metrics_fn:
            self.log_params_and_metrics_fn(None, metrics_agg)

        return loss_agg, metrics_agg

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: SimpleClientManagerWithPrioritizedSampling,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        else:
            # Default fit config function
            config = {
                "cf_method": CF_METHODS[self.method],
                "total_rounds": self.total_rounds,
            }
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        if server_round <= self.window:
            log(INFO, f"Round: {server_round} <= Window: {self.window} - Skipping...")
            clients = client_manager.sample(
                num_clients=sample_size,
                min_num_clients=min_num_clients,
            )
        else:
            if self.method == "lin_reg":
                priorities = self._calculate_carbon_based_priorities_using_linear_regression(
                    server_round=server_round
                )
            elif self.method == "exp_smooth":
                priorities = self._calculate_carbon_based_priorities_using_smoothing(
                    server_round=server_round
                )
            elif self.method == "simple_avg":
                priorities = self._calculate_carbon_based_priorities()
            else:  # NON_CF
                log(WARNING, f"Carbon Reduction disabled! Fallback to standard sampling...")
                priorities = {}

            # if priorities are empty, fallback to standard sampling
            if len(priorities.keys()) == 0 or np.sum(list(priorities.values())) <= 0.0:
                clients = client_manager.sample(
                    num_clients=sample_size,
                    min_num_clients=min_num_clients,
                )
            else:
                try:
                    # handles errors that might occur during the prioritized sampling
                    clients = client_manager.sample_with_priority(
                        num_clients=sample_size,
                        priorities=priorities,
                        min_num_clients=min_num_clients,
                    )
                except Exception as _:
                    log(ERROR, traceback.format_exc())

                    # fallback to the simple sampling
                    clients = client_manager.sample(
                        num_clients=sample_size,
                        min_num_clients=min_num_clients,
                    )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid not in self.emission_mapping:
                self.emission_mapping[cid] = {}
            carbon_emission = fit_res.metrics.get("carbon", -1.0)
            self.emission_mapping[cid][server_round] = carbon_emission

        params_agg, metrics_agg = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        if self.log_params_and_metrics_fn:
            self.log_params_and_metrics_fn(
                parameters_to_ndarrays(params_agg), metrics_agg
            )

        return params_agg, metrics_agg

    def _calculate_carbon_based_priorities(self):
        priorities = {}
        mean_emission_total = 0.0
        for cid, emissions in self.emission_mapping.items():
            if np.mean(list(emissions.values())) < 0:
                priorities[cid] = -1
            else:
                priorities[cid] = np.mean(
                    list(filter(lambda e: e > 0, list(emissions.values())))
                )
            mean_emission_total += priorities[cid]

        if mean_emission_total <= 0.0:
            for cid in priorities:
                priorities[cid] = 1 / len(self.emission_mapping.keys())
        else:
            for cid in priorities:
                if priorities[cid] > 0:
                    priorities[cid] = mean_emission_total / priorities[cid]
                else:
                    priorities[cid] = mean_emission_total / len(
                        self.emission_mapping.keys()
                    )

        return priorities

    def _calculate_carbon_based_priorities_using_smoothing(
        self,
        server_round: int,
    ):
        """uses a simple exponential smoothing logic for estimating
        the next round carbon emission and determine the priorities"""

        # calculate the mean emission per round across clients
        mean_emission_per_round = {}
        for fl_round in range(server_round - self.window, server_round, 1):
            measurement_count = 0
            total_emission_in_round = 0.0
            for _, emissions in self.emission_mapping.items():
                emission = emissions.get(fl_round, -1)
                if not emission or emission <= 0 or math.isnan(emission):
                    continue
                measurement_count += 1
                total_emission_in_round += emission
            if measurement_count != 0:
                mean_emission_per_round[fl_round] = (
                    total_emission_in_round / measurement_count
                )

        # if no measurement across any rounds of the time window,
        # return equal priorities
        if len(mean_emission_per_round.keys()) == 0:
            return {cid: 1 for cid in self.emission_mapping.keys()}

        # if there is non-measured rounds, take the mean across
        # measurements across available rounds
        for fl_round in range(server_round - self.window, server_round, 1):
            if fl_round not in mean_emission_per_round:
                mean_emission_per_round[fl_round] = np.mean(
                    list(mean_emission_per_round.values())
                )

        # compute the emission estimations for the next round
        next_round_emission_estimations = {}
        for cid, emissions in self.emission_mapping.items():
            emission_list = []

            for fl_round in range(server_round - self.window, server_round, 1):
                emission = emissions.get(fl_round, -1)
                if not emission or emission <= 0 or math.isnan(emission):
                    emission = mean_emission_per_round[fl_round]
                emission_list.append(emission)

            smoothed_emissions = exponential_smoothing(emission_list, self.alpha)

            next_round_emission_estimations[cid] = smoothed_emissions[-1]

        log(WARNING, f"Next Round Estimations: {next_round_emission_estimations}")
        mean_estimation_for_next_round = np.mean(
            list(next_round_emission_estimations.values())
        )

        # calculate and return the priorities based on the proportion of the mean estimation
        # to each of the client's estimation
        return {
            cid: mean_estimation_for_next_round / next_round_emission_estimations[cid]
            for cid in self.emission_mapping.keys()
        }

    def _calculate_carbon_based_priorities_using_linear_regression(
        self,
        server_round: int,
    ):
        """uses a simple Linear Regression model for estimating
        the next round carbon emission and determine the priorities"""

        # calculate the mean emission per round across clients
        mean_emission_per_round = {}
        for fl_round in range(1, server_round, 1):
            measurement_count = 0
            total_emission_in_round = 0.0
            for _, emissions in self.emission_mapping.items():
                emission = emissions.get(fl_round, -1)
                if not emission or emission <= 0 or math.isnan(emission):
                    continue
                measurement_count += 1
                total_emission_in_round += emission
            if measurement_count != 0:
                mean_emission_per_round[fl_round] = (
                    total_emission_in_round / measurement_count
                )

        # if no measurement across any rounds of the time window,
        # return equal priorities
        if len(mean_emission_per_round.keys()) == 0:
            return {cid: 1 for cid in self.emission_mapping.keys()}

        # if there is non-measured rounds, take the mean across
        # measurements across available rounds
        for fl_round in range(1, server_round, 1):
            if fl_round not in mean_emission_per_round:
                mean_emission_per_round[fl_round] = np.mean(
                    list(mean_emission_per_round.values())
                )

        # compute the emission estimations for the next round
        next_round_emission_estimations = {}
        for cid, emissions in self.emission_mapping.items():
            emission_values_for_regression = []
            rounds_for_regression = []

            for fl_round in range(1, server_round, 1):
                emission = emissions.get(fl_round, 0)
                if not emission or emission <= 0 or math.isnan(emission):
                    continue
                rounds_for_regression.append(fl_round)
                emission_values_for_regression.append(emission)

            if len(rounds_for_regression) < self.window:
                next_round_emission_estimations[cid] = mean_emission_per_round[server_round-1]
            else:
                lin_reg_model = LinearRegression()
                lin_reg_model.fit(
                    pd.DataFrame(rounds_for_regression), emission_values_for_regression
                )
                next_round_emission_estimations[cid] = (
                    lin_reg_model.predict(pd.DataFrame([server_round]))[0]
                )

        log(WARNING, f"Next Round Estimations: {next_round_emission_estimations}")
        mean_estimation_for_next_round = np.mean(
            list(next_round_emission_estimations.values())
        )

        # calculate and return the priorities based on the proportion of the mean estimation
        # to each of the client's estimation
        return {
            cid: mean_estimation_for_next_round / next_round_emission_estimations[cid]
            for cid in self.emission_mapping.keys()
        }
