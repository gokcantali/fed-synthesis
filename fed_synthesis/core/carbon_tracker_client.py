from codecarbon import EmissionsTracker as CodeCarbonEmissionsTracker


class CarbonTrackerClient:
    def __init__(self, backend="CodeCarbon"):
        if backend not in ["CodeCarbon"]:
            raise ValueError("Unsupported backend. Currently, only 'CodeCarbon' is supported.")

        self.tracker = CodeCarbonEmissionsTracker(
            measure_power_secs=10,
            experiment_id="2ef8bb00-570a-4110-b59f-68f8b5e5fd2a",
            save_to_api=False,
            allow_multiple_runs=True
        )

    def start_tracking(self):
        """Start tracking carbon emissions."""
        self.tracker.start()

    def stop_tracking(self):
        """Stop tracking carbon emissions."""
        return self.tracker.stop()
