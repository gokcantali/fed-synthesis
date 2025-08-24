class CodeCarbonBackend:
    def __init__(self):
        try:
            from codecarbon import EmissionsTracker
            self.EmissionsTracker = EmissionsTracker
        except ImportError:
            raise ImportError("CodeCarbon is not installed. Please install it with 'pip install codecarbon'.")

        self.tracker = None

    def start(self, experiment_id="2ef8bb00-570a-4110-b59f-68f8b5e5fd2a"):
        self.tracker = self.EmissionsTracker(
            measure_power_secs=10,
            experiment_id=experiment_id,
            save_to_api=False,
            allow_multiple_runs=True
        )
        self.tracker.start()

    def stop(self):
        if self.tracker:
            emissions = self.tracker.stop()
            return emissions
        else:
            raise RuntimeError("Tracker has not been started.")
