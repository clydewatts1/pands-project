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
    "target_box": "analysis_box.png",
    "target2_box": "analysis_box_II.png"
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

def convert_to_metrics_df(config):
    # This will convert to a metric dataframe
    # using melt , this is useful for certain types of plots
    # in the datacamp course - I am familiar with the metrics tables
    # from Lloyds Bank and implememnted in primark for SKU/Store/Date
    # use melt to convert the dataframe to long format
    # this is a classic metric table format
    # makes it easier to workout the mean, min, max, std, and median of the species and features
    # id_vars is the column to keep as is - target setosa, versicolor, virginica
    # var_name is the new column name for the features
    # value_name is the new column name for the values
    df_iris_melt = config['df'].melt(id_vars='species', var_name='feature', value_name='value')
    config["df_iris_melt"] = df_iris_melt
    logging.info("Melt DataFrame shape: %s", config["df_iris_melt"].shape)
    logging.info("Melt DataFrame columns: %s", config["df_iris_melt"].columns)
    logging.info("Melt DataFrame head: %s", config["df_iris_melt"].head())
    # check if the dataframe is empty
    if df_iris_melt.empty:
        logging.error("DataFrame is empty")
        return 1
    return 0

def load_summary(config):
    def quantile_25(x):
        return np.quantile(x, 0.25)

    def quantile_75(x):
        return np.quantile(x, 0.75)
    
    # Get dataframe from config
    if 'df' not in config:
        logging.error("DataFrame not in config")
        return -1
    # Get the dataframe from the config

    df = config['df_iris_melt'].groupby(['species', 'feature'])['value'].agg(['mean', 'min', 'max', 'std', 'median',quantile_25 , quantile_75]).reset_index()
    # Group by target and get the mean, min, max, std, median, 25th and 75th quantile

    df = df.rename(columns={'mean': 'Mean', 'min': 'Min', 'max': 'Max', 'std': 'Std', 'median': 'Median', 'quantile_25': 'Q25', 'quantile_75': 'Q75'})
    # Create a new column with the feature name
    # round the values to 2 decimal places
    df['Mean'] = df['Mean'].round(2)
    df['Min'] = df['Min'].round(2)
    df['Max'] = df['Max'].round(2)
    df['Std'] = df['Std'].round(2)
    df['Median'] = df['Median'].round(2)
    df['Q25'] = df['Q25'].round(2)
    df['Q75'] = df['Q75'].round(2)
    # check if the dataframe is empty
    if df.empty:
        logging.error("DataFrame is empty")
        return 1
    config['df_summary'] = df
    logging.info("Summary DataFrame shape: %s", config["df_summary"].shape)
    logging.info("Summary DataFrame columns: %s", config["df_summary"].columns)
    logging.info("Summary DataFrame head: %s", config["df_summary"].head())
    return 0

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
    # create a subplot with 2 rows and 2 columns to hold the historgrams
    fig,ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Iris Dataset Histograms', fontsize=16)
    df = config['df']
    sns.set_palette("pastel")
    sns.histplot(df, x='sepal_length', hue='species', ax=ax[0, 0], alpha=0.2, bins=20, kde=True)
    sns.histplot(df, x='sepal_width', hue='species', ax=ax[0, 1], alpha=0.2, bins=20, kde=True)
    sns.histplot(df, x='petal_length', hue='species', ax=ax[1, 0], alpha=0.2, bins=20, kde=True)
    sns.histplot(df, x='petal_width', hue='species',  ax=ax[1, 1], alpha=0.2, bins=20, kde=True)

    plt.tight_layout()
    if to_console:
        # print the histogram to the console
        plt.show()
    # save the histogram to a file
    plt.savefig(config["target_histogram"])
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
    sns.set_palette("pastel")
    # create a subplot with 2 rows and 2 columns to hold the historgrams
    fig,ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Iris Dataset Scatter Plot', fontsize=16)
    df = config['df']
    # First row is sepal length and width and petal length and width
    # Second row is sepal length and petal length and sepal width and petal width
    sns.scatterplot(data=config['df'], x='sepal_length', y='sepal_width', hue='species', ax=ax[0, 0])
    sns.scatterplot(data=config['df'], x='petal_length', y='petal_width', hue='species', ax=ax[0, 1])
    sns.scatterplot(data=config['df'], x='sepal_length', y='petal_length', hue='species', ax=ax[1, 0])
    sns.scatterplot(data=config['df'], x='sepal_width', y='petal_width', hue='species', ax=ax[1, 1])
    plt.tight_layout()
    if to_console:
        # print the scatter plot to the console
        plt.show()
    # save the scatter plot to a file
    plt.savefig(config["target_scatter"])
    plt.close()
    return 0
    

def generate_box_plot(config,to_console = False):
    # Generate a box plot of the data
    # check if the dataframe is empty
    if 'df' not in config:
        logging.error("DataFrame not in config")
        return -1
    # check if the box plot file exists
    if os.path.exists(config["target_box"]):
        # remove the file
        os.remove(config["target_box"])
    # create a subplot with 2 rows and 2 columns to hold the historgrams
    fig,ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Iris Dataset Box Plot', fontsize=16)
    df = config['df']
    # Like grid lines easier to see
    ax[0,0].grid(True)
    ax[0,1].grid(True)
    ax[1,0].grid(True)
    ax[1,1].grid(True)
    # xlabels stackoverflow
    # TODO : get the x-axis to be the same for all plots

    # https://stackoverflow.com/questions/19509870/how-to-set-x-axis-labels-in-seaborn-boxplot
    sns.boxplot(data=config['df'], x='species', y='sepal_length', ax=ax[0, 0],hue='species', palette="pastel").set(xlabel="Species", ylabel="Sepal Length")
    sns.boxplot(data=config['df'], x='species', y='sepal_width', ax=ax[0, 1], hue='species', palette="pastel").set(xlabel="Species", ylabel="Sepal Width")
    sns.boxplot(data=config['df'], x='species', y='petal_length', ax=ax[1, 0], hue='species', palette="pastel").set(xlabel="Species", ylabel="Petal Length")
    sns.boxplot(data=config['df'], x='species', y='petal_width', ax=ax[1, 1], hue='species', palette="pastel").set(xlabel="Species", ylabel="Petal Width")
    plt.tight_layout()
    if to_console:
        # print the box plot to the console
        plt.show()
    # save the box plot to a file
    plt.savefig(config["target_box"])
    plt.close()
    return 0

def generate_box_plot_II(config,to_console = False):
    # Generate a box plot of the data
    # check if the dataframe is empty
    if 'df' not in config:
        logging.error("DataFrame not in config")
        return -1
    # check if the box plot file exists
    if os.path.exists(config["target2_box"]):
        # remove the file
        os.remove(config["target2_box"])
    df_iris_melt = config['df_iris_melt'] 
    # Plotting the boxplot using seaborn - datacamp seaborn tutoria
    sns.set_context("notebook")
    # Set the style of seaborn
    sns.set_style("darkgrid")
    g = sns.catplot(data=df_iris_melt,kind="box",x="species",y="value",hue='species',col="feature",col_wrap=2,sharex=True)
    # how to set the title for each subplot - github copilot assited with adjustment
    # Adjust the top space for the title and increase spacing between subplots
    plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.4)
    # set grid on each subplot
    for ax in g.axes.flat:
        ax.grid(True)
        # get the title of each subplot
        # This could be done in one line, but for clarity multiple lines are used
        title = ax.get_title()  # this gets the title of each subplot
        # remove the prefix "feature = " from the title
        # and replace "_" with " "
        title = title.replace("feature = ", "").replace("_"," ")
        # now capitalize the first letter of each word
        title = title.title()
        print(title)
        # set the title of each subplot 
        ax.set_title(title)
    plt.suptitle("Boxplot of features by species")
    if to_console:
        # print the box plot to the console
        plt.show()  
    # save the box plot to a file
    plt.savefig(config["target2_box"])
    plt.close()
    return 0

def main():
    # Set up logging
    setup_logging()
    # TODO: Add yaml config file

    return_code, df = load_data(config)
    # If there is a error loading the data, log it and return
    if return_code == 1:
        logging.error("Failed to load data from %s", file_path)
        return
    
    # COnvert data to metrics dataframe
    return_code = convert_to_metrics_df(config)
    # If there is a error converting the data, log it and return
    if return_code == 1:
        logging.error("Failed to convert data to metrics dataframe")
        return
    
    # Generate summary statistics
    return_code = load_summary(config)
    # If there is a error generating the summary statistics, log it and return
    if return_code == 1:
        logging.error("Failed to generate summary statistics")
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
    
    # Generate the box plot
    return_code = generate_box_plot(config)
    # If there is a error generating the box plot, log it and return
    if return_code == -1:
        logging.error("Failed to generate box plot")
        return
    
    # Generate the box plot II
    return_code = generate_box_plot_II(config)
    # If there is a error generating the box plot, log it and return
    if return_code == -1:
        logging.error("Failed to generate box plot II")
        return

if __name__ == "__main__":
    main()