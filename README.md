# Setup the project.

### 0. Setup

Install the following before setting up the project

`poetry` - To manage requirements

> **Tip**
>
> Install `pyenv` to manage multiple versions of python. This project requires Python 3.9+. We wil be using native type annotations that is not available in veresions below this

### 1. Clone the project

```bash
git clone https://github.com/abhinavkashyap/chataway.git
```

```bash
cd knowme
```

### 2. Install the dependencies

Activate the virtual environment

```bash
poetry shell 
```

Install the dependencies

```bash
poetry install
```


### 3. Run the app 
```bash 
streamlit run app.py
```

This runs a stremalit app which allows the person to log some 
daily measures. 
