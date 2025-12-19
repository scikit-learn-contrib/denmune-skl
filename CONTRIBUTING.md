# Contributing to DenMune-Sklearn

First off, thank you for considering contributing to DenMune-Sklearn! Any contribution, from bug reports to new features, is greatly appreciated.

## Getting Started

Before you begin, please take a moment to read through these guidelines. If you have any questions, feel free to open an issue on GitHub.

### 1. Raise an Issue

Before making any changes, please open an issue to discuss your proposed contribution. This allows us to coordinate efforts and ensure that your work is in line with the project's goals.

### 2. Fork the Repository

Once you're ready to start coding, fork the main repository to your own GitHub account.

### 3. Set Up the Development Environment

We use `pixi` to manage the development environment.

1.  **Install Pixi:** Follow the instructions on the [official pixi website](https://pixi.sh/latest/) to install it on your system.

2.  **Install Dependencies:** Navigate to the project's root directory and run:
    ```bash
    pixi install
    ```
    This will create a virtual environment and install all the necessary dependencies for running, testing, and building the documentation.

3.  **Set Up Pre-commit Hooks:** We use pre-commit hooks to ensure code quality and consistency. To install them, run:
    ```bash
    pre-commit install
    ```
    This will run linters and formatters automatically every time you make a commit.

## Development Workflow

1.  **Create a Branch:** Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b your-feature-name
    ```

2.  **Make Changes:** Write your code and make your changes.

3.  **Run Linters and Formatters:** Before committing, ensure your code adheres to the project's style guidelines by running:
    ```bash
    pixi run lint
    ```

4.  **Run Tests:** Make sure all tests pass, including any new tests you've added for your changes:
    ```bash
    pixi run test
    ```

5.  **Commit Your Changes:** Commit your changes with a clear and descriptive commit message.

6.  **Push to Your Fork:** Push your changes to your forked repository:
    ```bash
    git push origin your-feature-name
    ```

7.  **Submit a Pull Request:** Open a pull request from your fork to the main DenMune-Sklearn repository. In the pull request description, please reference the issue you created in the first step.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. We are committed to fostering an open and welcoming environment.
