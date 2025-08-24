from os import mkdir
import uuid


class CarbontrackerBackend:
    def __init__(self):
        try:
            from carbontracker.tracker import CarbonTracker
            self.CarbonTracker = CarbonTracker

            log_folder_name = str(uuid.uuid4())
            self.log_folder_path = f"./carbon_logs/{log_folder_name}"
        except ImportError:
            raise ImportError("CodeCarbon is not installed. Please install it with 'pip install codecarbon'.")

        self.tracker = None

    def start(self, max_epochs=1):
        mkdir(self.log_folder_path)
        self.tracker = self.CarbonTracker(
            epochs=max_epochs,
            verbose=True,
            update_interval=1.0,
            log_dir=self.log_folder_path,
        )
        self.tracker.epoch_start()

    def stop(self):
        from carbontracker import parser
        if self.tracker:
            self.tracker.epoch_end()
            self.tracker.stop()

            logs = parser.parse_all_logs(log_dir=self.log_folder_path)
            emission_log = logs[0]
            emission_details = emission_log.get('actual', {})
            return emission_details.get('co2eq (g)', 0.0) / 1000  # in kg
        else:
            raise RuntimeError("Tracker has not been started.")
