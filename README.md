# STL-10 Image Classifier & Streamlit App

## Overview

This project is an image classifier trained on the STL-10 dataset. The model is capable of identifying various object classes in images. Additionally, the repository includes a Streamlit app that allows users to upload photos and classify them in real-time. There is a file with saved parameters if you wish to try it out of the box.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites

- Python 3.10
- pip
- Virtual environment (optional)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/STL10-Image-Classifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd STL10-Image-Classifier
    ```
3. Install the required packages(Visit [Pytorch](https://pytorch.org) for installation guide):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Model Training

To train the model, run the following command:

```bash
python unv_model.py
```

### Streamlit App

To launch the Streamlit app, run:

```bash
streamlit run app.py
```

Open the displayed URL in your browser to interact with the application.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- STL-10 dataset creators
- Streamlit for their amazing framework
