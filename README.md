# FedSynthesis: A Carbon-aware Federated Learning Framework based on Flower

This repository provides a framework for implementing federated learning (FL) tasks using Flower,
with a focus on carbon-aware client selection. It includes components for both server and client sides,
allowing users to run FL tasks efficiently and sustainably. There is also an example use-case on how
the framework can be applied to existing ML tasks.

---

## Table of Contents

1. [Description](#description)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the Example](#running-the-example)
6. [Directory Structure](#directory-structure)
7. [Testing](#testing)
8. [Roadmap](#roadmap)
9. [License](#license)
10. [Credits / Acknowledgements](#credits--acknowledgements)
11. [Contact](#contact)

---

## Description

**FedSynthesis** aims to provide a practical framework for FL applications to measure and reduce the carbon footprint
of the FL process. It leverages Flower's model-agnostic capabilities to allow users to federate existing
Machine Learning (ML) tasks while incorporating carbon-aware client selection strategies. With this approach, we aim to
reduce the environmental impact of federated learning without compromising the performance of ML models.

FedSynthesis utilizes a carbon emission tracker to select clients based on their carbon emissions. For now, only **CodeCarbon**
is supported as the carbon emission tracker tool, but the framework is designed to be flexible, allowing extensions for other tools in the future.
Users of the framework can use other tools for carbon emission tracking, as long as they implement a backend class that adheres to the provided interface.

The framework allows users to visualize carbon emissions and performance metrics across FL rounds, via an MLOps tool.
For now, only **MLFlow** is supported as the MLOps tool, but similar to the carbon emission tracker, the framework is designed to be extensible
for other tools and users can implement their own backend class to integrate with different MLOps tools.

---

## Features

The following features are provided by FedSynthesis:
- ✅ Federated learning (FL) framework based on Flower, agnostic to the underlying ML framework
- ✅ Carbon-aware client selection based on measurements from a carbon emission tracker
- ✅ Visualization of carbon emission values, along with performance metrics, across FL rounds
- ✅ Easily applicable to existing ML tasks
- ✅ Flexible design for incorporating additional carbon estimation methods 

---

## Requirements

The built-in stack provided by FedSynthesis, including _CodeCarbon_ and _MLFlow_,has been tested
on the following operating systems so far:
- macOS 15.4+
- Linux (Ubuntu 22.04+)

Although the framework has not been tested on Windows, it is expected to work on Windows 10+ as well, 
given that underlying dependencies (e.g., CodeCarbon and MLFlow) are cross-platform.

### CodeCarbon Configuration

> Note that for the CodeCarbon to interact with CPU, GUP, and RAM properly, platform-specific
configurations might be required. The reader is encouraged to refer to the official documentation
of CodeCarbon for further details: [CodeCarbon Documentation](https://mlco2.github.io/codecarbon/methodology.html).

### System Requirements

FedSynthesis has the following main requirements:
- **Python**: `>= 3.11`
- **pip**: `>= 24.0`
- **Flower**: `== 1.12`
- **numpy**: `>= 1.26.5 < 2.0`

If the built-in stack is used, the following additional requirements apply:
- **CodeCarbon**: `>= 2.8.0`
- **MLFlow**: `>= 2.20.0`

For dependency management, we recommend using either `poetry` or `pip`. There are **pyproject.toml** and **poetry.lock** files in the root directory
for poetry installation, and a **requirements.txt** file for pip usage. There is also another pyproject.toml file  in the `gnn_example` directory,
which is specifically used for the example use-case, including the required dependencies for the GNN model such as _torch_, _torch-geometric_, etc.

---

## Installation

Step‐by‐step instructions to get the project running locally.

### 1. Clone the repository

```bash
git clone git@github.com:gokcantali/fed-synthesis.git
cd fed-synthesis
```

### 2. Create a virtual environment (Optional but highly recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
For poetry, you can use:
```bash
poetry install
```

And for pip, you can use:
```bash
pip install -r requirements.txt
```
> NOTE: If you use **Pip** with the requirements.txt file, you will also install the use-case specific
> dependencies such as _torch_, _torch-geometric_, etc.

### 4. Check CodeCarbon configuration (Optional)
If you intend to use the built-in support with CodeCarbon,
you may need to configure it according to your system specifications, as mentioned before.
You can check the CodeCarbon documentation for more details:
[CodeCarbon Documentation](https://mlco2.github.io/codecarbon/methodology.html).

---

## Running the Example

For running the example use-case, which is a Graph Neural Network (GNN) model,
you can follow these steps:

### 1. Install the example dependencies

If you used the 'requirements.txt' file with **Pip** before, your environment is already set up for the example.

If you used **Poetry** for the main project, you can change to the `gnn_example` directory 
and install the use-case specific dependencies:
```bash
cd gnn_example
poetry install
```

> NOTE: From this point onwards, if you use **Poetry**,
> you may need to add `poetry run` prefix to the following commands.

### 2. Start the SuperLink on the Server Side

In a terminal, start the SuperLink server process:
```bash
flower-superlink --insecure
```

### 3. Start Supernode instances on the Client Side

The example use-case supports up to 5 FL clients, so you can open
a separate terminal for each client (up to 5) and run the following command:
```bash
flower-supernode fed_synthesis --server 0.0.0.0:9092 --insecure --node-config 'partition-id=<node-id>'
```
where `<node-id>` is the ID of the client (0 to 4).

### 4. Start MLFlow Tracking Server

In another terminal, start the MLFlow server:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 5. Start the Aggregator on the Server Side

Finally, in another terminal, start the aggregator:
```bash
poetry run flower-server-app fed_synthesis --superlink 0.0.0.0:9091 --insecure
```
At this point, you should be able to see the logs on each running client app, 
and also on the aggregator terminal.

---

## Directory Structure

The directory structure of the project is as follows:

```
<RepoRootFolder>/
├── fed_synthesis/            # Main source code for the framework
├── gnn_example/              # Example use-case with GNN model
│   ├── data/                 # Example data files
│   │   └── graph/            # Example graph data files
│   ├── preprocessing/        # Preprocessing scripts and functions
│   │   ├── converter.py
│   │   ├── encoder.py
│   │   └── preprocesser.py
│   ├── util/                 # Util functions
│   │   ├── constants.py
│   │   ├── data_models.py
│   │   └── load_data.py
│   ├── fedl_setup.py         # FL preparetion scripts for the GNN example
│   ├── gcn.py                # GNN model implementation
│   ├── poetry.lock           # Poetry lock file for the example
│   └── pyproject.toml        # Poetry project file for the example
├── your_package/             # Main library code
│   ├── __init__.py
│   ├── processor.py
│   └── utils.py
├── .gitignore
├── poetry.lock               # Poetry lock file
├── poetry.toml
├── pyproject.toml            # Poetry project file
├── README.md                 # ← This file
└── requirements.txt
```

---

## Testing

Unfortunately, there are no tests implemented yet for the framework.
But hopefully, we will add tests in the near future to ensure the correctness of the framework.

---

## Roadmap

The following features are planned for future releases:
- [ ] Add support for additional carbon emission trackers
- [ ] Add support for additional MLOps tools
- [ ] Apply Hexagonal Architecture to the codebase for better modularity
- [ ] Implement unit tests and integration tests for the framework
- [ ] Incorporate more advanced client selection strategies
- [ ] Include more example use-cases with different ML models

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Credits / Acknowledgements

This work has received co-funding from the Smart Networks
and Services Joint Undertaking (SNS JU), the Swiss State
Secretariat for Education, Research and Innovation (SERI),
and the UK Research and Innovation (UKRI) under the
European Union’s Horizon Europe research and innovation
programme, in the frame of the NATWORK project (Net-Zero
self-adaptive activation of distributed self-resilient augmented
services) under Grant Agreement No 101139285.

## Contact

For any questions or issues regarding the framework,
please feel free to contact the project maintainer(s):

- Maintainer: Gökcan Çantali (gokcantali@gmail.com)
- Issues: https://github.com/gokcantali/fed-synthesis/issues

