# Low HD Project Repo

## Configuration
This project requires a `config.ini` file to specify paths and directories needed for the program to run. Follow the instructions below to create and configure your `config.ini` file.

1. **Create the `config.ini` file**: In the root directory of your project, create a new file named `config.ini`.

2. **Add the following content to the `config.ini` file**:

       
       [Directories]
       data_dir = your\path\here\000939
       results_dir = your\path\here\data_results
       cell_metrics_dir = your\path\here

       
   
   **What are these paths?**

`data_dir`: Path to the directory where your DANDI data is stored.

`results_dir`: Path to the directory where you want the analysis results to be saved.

`cell_metrics_dir`: Path to the directory where the cell metrics file will be generated.

3. **Customize the paths**: Replace the paths in the example 