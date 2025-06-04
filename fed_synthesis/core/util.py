def exponential_smoothing(data, alpha):
    """
    Apply simple exponential smoothing to a time-series data.

    Parameters:
    - data: List of numerical values representing the time-series.
    - alpha: Smoothing factor, a float between 0 and 1.

    Returns:
    - smoothed_data: List of smoothed values.
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be a value between 0 and 1.")

    if len(data) == 0:
        raise ValueError("Input data must not be empty.")

    smoothed_data = [data[0]]  # First value is same as the original

    for t in range(1, len(data)):
        smoothed_value = alpha * data[t] + (1 - alpha) * smoothed_data[t-1]
        smoothed_data.append(smoothed_value)

    return smoothed_data


def get_dataset_splits(client_id):
    # This function should be provided by the client ML code
    return [], [], []


def initialize_model():
    # This function should be provided by the client ML code
    return None
