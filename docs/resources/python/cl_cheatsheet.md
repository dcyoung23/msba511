
| Command Line                                       | Notes                                                                                                                          |
|:---------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| `conda list`                                       | Uses conda to list out all installed packages, version, build and how it was installed in the current environment.             |
| `conda search --full-name python`                  | Searches conda channels for the current versions of python that are available.                                                 |
| `conda env list`                                   | List out all conda environments on computer.                                                                                   |
| `conda create -n myenv python=3.12.7`              | Create a new environment with the specified version of Python and minimal packages. Replace `myenv` with the desired name of the new environment and `3.12.7` with the desired Python version.                                                 |
| `conda env remove -n myenv`                        | Remove environment from your computer. You cannot have the current environment activated. Replace `myenv` with the name of the environment that you want to remove. ⚠️Warning ⚠️ this deletes the env folder from your computer! |
| `conda deactivate`                                 | Deactivate current environment.                                                                                                |
| `conda activate myenv`                             | Activate specified environment. Replace `myenv` with the name of the environment that you want to activate.                                            |
| `pip list`                                         | Use pip to list installed packages and version. Similar to conda list.                                                         |
| `pip install pandas`                               | Use pip to install the current version of a package in the current environment. Replace `pandas` with the name of the package that you want to install. This will also install all dependencies.           |
| `pip install pandas==2.0`                          | Use pip to install a specific version of a package in the current environment.                                                 |
| `pip install "pandas<2.0"`                         | Use pip to install the version less than a specific version of a package in the current environment. ℹ️ Double Quotes are required.                         |
| `pip install --upgrade pandas`                     | Use pip to upgrade to the current version of a package in the current environment.                                             |
| `pip uninstall pandas`                             | Use pip to uninstall a package from the current environment.                                             |



	
	
