#------------------------------------------------------------------------------
# File: analysis.py
# Description: 24-25: 4122 -- PROGRAMMING AND SCRIPTING : Project
# Author: Clyde Watts
# Date: 2025-03-19
# Version: v1.0
#------------------------------------------------------------------------------
# Requirements
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

log_file = "analysis.log"
# configuration dictionary
config = {
    "source_csv_file": "iris.data",
    "source_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    "target_report": "analysis_report.txt",
    "target_histogram": "analysis_histograms.png",
    "target_scatter": "analysis_scatter.png",
}

def setup_logging(log_file = log_file,level = logging.INFO):
    # Set up logging configuration
    # set logging directory to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, log_file)
    # clear the log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)


def load_data(config):
    # Load data from a CSV file into a pandas DataFrame

    # return_code: 0 = success, 1 = failure
    # df: pandas DataFrame , set to None if return_code is 1
    if "source_csv_file" not in config:
        logging.error("source_csv_file not in config")
        return 1, None
    # Check if the file exists
    file_path = config["source_csv_file"]
    logging.info("Loading data from %s", file_path)
    return_code = 0
    # Check if file loaded correctl
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            logging.error("File %s does not exist", file_path)
            return_code = 1
            return return_code, None

        # Load the data into a pandas DataFrame
        df = pd.read_csv(file_path, header=None, names=config["source_columns"])
    # Catch generic exceptions
    except Exception as e:
        logging.error("Error parsing file %s: %s", file_path, e)
        return_code = 1
        return return_code, None
    # Check if the dataframe is empty
    if df.empty:
        logging.error("DataFrame is empty")
        return_code = 1
        return return_code, None
    logging.info("Loaded data from %s", file_path)
    logging.info("DataFrame shape: %s", df.shape)
    logging.info("DataFrame columns: %s", df.columns)
    logging.info("DataFrame head: %s", df.head())
    # add df to the config
    config["df"] = df
    # check if the dataframe is empty
    return return_code , df


def generate_report(config,to_console = False):
    # setup a report file name
    report_file = config["target_report"]
    # check if the report file exists
    if os.path.exists(report_file):
        # remove the file
        os.remove(report_file)
    # open the file for writing
    if 'df' not in config:
        logging.error("DataFrame not in config")
        return -1
    with open(report_file, "w") as f:
        # write the header
        f.write("Analysis Report\n")
        f.write("===============\n")
        # write the dataframe to the file
        f.write("DataFrame\n")
        f.write("=========\n")
        f.write(f"DataFrame Shape: {config['df'].shape}\n")
        f.write(f"DataFrame Info: {config['df'].info}\n")
        f.write(f"DataFrame Columns: {config['df'].columns}\n")
        f.write(f"DataFrame Head: {config['df'].head()}\n")
        f.write("\n")
        # write the summary statistics to the file
        f.write("Summary Statistics\n")
        f.write("==================\n")
        f.write(config["df"].describe().to_string())
    # Print out file - use in jupyter notebook
    if to_console:
        # print the report to the console
        with open(report_file, "r") as f:
            print(f.read())
    return 0

def generate_histogram(config,to_console = False):
    # Generate a histogram of the data
    # check if the dataframe is empty
    if 'df' not in config:
        logging.error("DataFrame not in config")
        return -1
    # check if the histogram file exists
    if os.path.exists(config["target_histogram"]):
        # remove the file
        os.remove(config["target_histogram"])
    # create a histogram of the data
    plt.figure(figsize=(10, 6))
    sns.histplot(data=config['df'], kde=True)
    plt.title("Histogram of Data")
    plt.savefig(config["target_histogram"])
    if to_console:
        # print the histogram to the console
        plt.show()
    plt.close()
    return 0

def generate_scatter_plot(config,to_console = False):
    # Generate a scatter plot of the data
    # check if the dataframe is empty
    if 'df' not in config:
        logging.error("DataFrame not in config")
        return -1
    # check if the scatter plot file exists
    if os.path.exists(config["target_scatter"]):
        # remove the file
        os.remove(config["target_scatter"])
    # create a scatter plot of the data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=config['df'], x='sepal_length', y='sepal_width', hue='species')
    plt.title("Scatter Plot of Data")
    plt.savefig(config["target_scatter"])
    if to_console:
        # print the scatter plot to the console
        plt.show()
    plt.close()
    return 0
def generate_box_plot(config):
    # Generate a box plot of the data
    pass


def main():
    # Set up logging
    setup_logging()
    # TODO: Add yaml config file

    return_code, df = load_data(config)
    # If there is a error loading the data, log it and return
    if return_code == 1:
        logging.error("Failed to load data from %s", file_path)
        return
    
    # Generate the report
    return_code = generate_report(config)
    # If there is a error generating the report, log it and return
    if return_code == -1:
        logging.error("Failed to generate report")
        return
    
    # Generate the histogram
    return_code = generate_histogram(config)
    # If there is a error generating the histogram, log it and return
    if return_code == -1:
        logging.error("Failed to generate histogram")
        return

    # Generate the scatter plot
    return_code = generate_scatter_plot(config)
    # If there is a error generating the scatter plot, log it and return
    if return_code == -1:
        logging.error("Failed to generate scatter plot")
        return

if __name__ == "__main__":
    main()