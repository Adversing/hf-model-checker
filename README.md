# hf-model-checker

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Accepted URL Types](#accepted-url-types)
    - [1. `.safetensors` and `.bin` Model URLs](#1-safetensors-and-bin-model-urls)
    - [2. Non-specific GGUF Quantization URLs](#2-non-specific-gguf-quantization-urls)
    - [3. Specific GGUF Quantized Model URLs](#3-specific-gguf-quantized-model-urls)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Overview

**hf-model-checker** is a command-line tool designed to analyze Hugging Face model URLs and recommend the most suitable quantization options based on your system's available resources. By evaluating your system's RAM and VRAM, the tool ensures optimal performance and compatibility when loading large machine learning models.

## Features

- **System Resource Analysis:** Detects available RAM and VRAM to determine feasible model quantizations.
- **Quantization Recommendations:** Suggests the best quantization method from predefined multipliers to balance performance and memory usage.
- **Comprehensive Reporting:** Provides detailed information about the model size, required memory, and recommended quantization in a user-friendly format.
- **Supports Multiple Model Formats:** Handles `.safetensors`, `.bin` and GGUF quantized models, both specific and non-specific versions.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Adversing/hf-model-checker.git
   cd hf-model-checker
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Ensure you have Python 3.8 or higher installed.

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

- **`quant_multipliers.json`:** This JSON file defines multipliers for different quantization methods, influencing the estimated RAM required for each quantization type. 

## Usage

1. **Run the Script:**

   Navigate to the project directory and execute the script:

   ```bash
   python hf_model_checker.py
   ```

2. **Enter a Hugging Face Model URL:**

   When prompted, input the Hugging Face model URL you wish to analyze.

   ```plaintext
   Enter a Hugging Face model URL (or 'exit' to quit):
   ```

3. **Receive Analysis:**

   The tool will display an analysis of the model, including size, required memory, and recommended quantization.

### Accepted URL Types

**hf-model-checker** accepts three types of Hugging Face model URLs. Each type corresponds to different model formats and quantization methods.

#### 1. `.safetensors` and `.bin` Model URLs

- **Description:** URLs that directly point to a directory containing `.safetensors` or `.bin` files, which are optimized tensor formats for efficient storage and loading.
- **Usage Scenario:** When you have a standard model file without any specific quantization applied.
- **Example URL:**

  ```
  https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
  ```

- **Behavior:** The tool will analyze the directory files and estimate memory requirements based on the model's size.

#### 2. Non-specific GGUF Quantization URLs

- **Description:** URLs that point to a repository containing GGUF quantized versions of models without specifying a particular quantization variant.
- **Usage Scenario:** When the repository includes multiple GGUF quantization options, and you want the tool to evaluate all available quantizations.
- **Example URL:**

  ```
  https://huggingface.co/lmstudio-community/Llama-3.3-70B-Instruct-GGUF
  ```

- **Behavior:** The tool scans the repository for all GGUF files, evaluates each quantization based on system resources, and recommends the most suitable quantization method.

#### 3. Specific GGUF Quantized Model URLs

- **Description:** URLs that point directly to a specific GGUF quantized model file.
- **Usage Scenario:** When you have a particular GGUF quantization variant in mind and want to verify its compatibility with your system.
- **Example URL:**

  ```
  https://huggingface.co/lmstudio-community/Llama-3.3-70B-Instruct-GGUF/blob/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf
  ```

- **Behavior:** The tool analyzes the specified GGUF file, estimates the required memory, and indicates whether your system can efficiently handle the quantized model.

## Example

![image](https://github.com/user-attachments/assets/632bead6-4a1d-449e-940e-ecce31f0f550)

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).
