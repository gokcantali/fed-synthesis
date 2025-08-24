from fed_synthesis.infra.carbontracker_backend import CarbontrackerBackend
from fed_synthesis.infra.code_carbon_backend import CodeCarbonBackend


class CarbonTrackerClient:
    def __init__(self, backend="CodeCarbon"):
        if backend not in ["CodeCarbon", "carbontracker"]:
            raise ValueError("Unsupported backend. Currently, only 'CodeCarbon' and 'carbontracker' are supported.")

        if backend == "CodeCarbon":
            self.tracker_backend = CodeCarbonBackend()
        if backend == "carbontracker":
            self.tracker_backend = CarbontrackerBackend()

    def start_tracking(self):
        """Start tracking carbon emissions."""
        self.tracker_backend.start()

    def stop_tracking(self):
        """Stop tracking carbon emissions."""
        return self.tracker_backend.stop()
