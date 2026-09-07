# Python Setup

601.465/665 assumes familiarity with Python. This guide describes the recommended environment setup for completing homework assignments.  It ensures that you're using the same package versions as the course staff.

You should use Python 3.12 for homework assignments. We recommend working locally with an editor such as VS Code and a Python virtual environment.

The CS Linux machines are also available if you prefer to work remotely.

## Working remotely on the CS Linux machines

The CS Linux machines provide a shared Conda environment, `nlp-class`, with the packages required for this course.

First, obtain a CS account by following the instructions in Dr. Eisner's Piazza post. Then, connect over SSH to an available **`ugrad`** or **`grad`** machine ([SSH guide](https://support.cs.jhu.edu/wiki/Connecting_to_CS_Systems_with_SSH)).

After logging in, activate the shared environment:

```bash
conda activate nlp-class
```

You can now run Python on the remote machine.

You can also use VS Code remotely through the [Remote - SSH extension](https://code.visualstudio.com/docs/remote/ssh).


## Working locally on your own machine (recommended)

### 1. Install Python

Install Python 3.12 from the [official Python website](https://www.python.org/downloads/). Then open a new terminal and verify that you can run it:

> Some assignment helper scripts require Perl and a Unix-style shell.
On macOS or Linux, confirm that Perl is installed by running `perl --version`. 
*On Windows, we recommend using [WSL](https://learn.microsoft.com/windows/wsl/install) and following the Linux instructions*.

**macOS or Linux**

```bash
python3.12 --version
```

**Windows (PowerShell)**

```powershell
py -3.12 --version
```


### 2. Set up VS Code (if you haven't already)

Install [Visual Studio Code](https://code.visualstudio.com/) and Microsoft's [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

For an introduction, see VS Code's [Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial).

Open the assignment directory in VS Code with **File > Open Folder**, then use the integrated terminal for the commands below.

### 3. Create a virtual environment for the NLP class

Open a terminal with **Terminal > New Terminal**, then follow these instructions to create a [virtual environment](https://docs.python.org/3/library/venv.html) named `nlp-class`. You will use this environment for all NLP assignments throughout the semester.


**macOS or Linux**

```bash
VENVS=~/.venvs    # or another directory name you prefer
mkdir -p $VENVS
python3.12 -m venv $VENVS/nlp-class
```

**Windows (PowerShell)**

```powershell
$VENVS = ~/venvs    # or another directory name you prefer
New-Item -ItemType Directory -Force $VENVS
py -3.12 -m venv $VENVS/nlp-class
```

> Optional: For larger or longer-lived Python projects, [â uv](https://docs.astral.sh/uv/getting-started/installation/) is a fast, modern alternative for managing Python versions, virtual environments, and dependencies. 

### 4. Tell VS Code to use the environment for your homework

Download the homework materials to a local directory.  Open that directory in VS Code and then select the `nlp-class` environment.

1. Open the Command Palette (`Cmd/Ctrl+Shift+P`).
2. Choose **Python: Select Interpreter**.
3. Select the Python interpreter inside `$VENVS/nlp-class`. If it does not appear in the list, choose **Enter interpreter path...** and browse to `$VENVS/nlp-class/bin/python` on macOS or Linux, or `$VENVS/nlp-class/Scripts/python.exe` on Windows.

VS Code will then associate the `nlp-class` environment with this project, using it for all of the project's Python scripts, notebooks, tests, and tools.

### 5. Tell your shell to use the environment

VS Code will automatically activate `nlp-class` when you open a new integrated terminal. If you want to run Python from a terminal outside VS Code, or if automatic activation does not occur, you need to first activate `nlp-class` within that particular shell session.

**macOS or Linux**

```bash
source $VENVS/nlp-class/bin/activate
```

**Windows (PowerShell)**

```powershell
$VENVS/nlp-class/Scripts/Activate.ps1
```

> PowerShell may initially block activation scripts. 
See the Python [`venv` documentation](https://docs.python.org/3/library/venv.html) for the recommended execution policy setting.

### 6. Populate the `nlp-class` environment

So far, `nlp-class` is an empty environment: no special packages will be available.  So let's add the packages required for the assignment.  With `nlp-class` activated, run:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
```
where `requirements.txt` was provided in the homework directory.  

For your convenience, we are trying to use the same `requirements.txt` for the entire course, so you'll probably only have to do this for HW1 and never again.

**Note:** If your solution needs any additional packages that were not in the starter code, please add them to `requirements.txt` yourself (rerunning the above commands) and include that file in your submitted code.  Then the autograder will be able to find the additional packages and everything will run.

### 7. Run your code

Each time you work on an assignment:

1. Open the assignment's directory in VS Code.  VS Code will remember that you were using the `nlp-class` environment.
2. Open a new terminal. VS Code should automatically activate `nlp-class`.  Then you can run your code:

```bash
python your_program.py
```