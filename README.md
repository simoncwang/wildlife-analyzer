# Wildlife Analyzer MLOps Dashboard

[💻 Public Demo](https://wildlife-analyzer-dashboard.streamlit.app/) | [🌐 My Website](https://simoncwang.github.io/)

This project leverages the [iNaturalist API](https://api.inaturalist.org/v1/docs/), which is an open source community for sharing high-quality wildlife sightings.
The Wildlife Analyzer consists of a modular MLOps-style pipeline with the following stages:

- **Fetch:** Fetches relevant observation data from the API based on the parameters specified in the settings
- **Pre-Processing:** Uses Pandas and Python to parse the data into a clean dataframe format, saving it as a CSV
- **Modeling:** Depending on parameter settings, can run various ML and GenAI analyses on the data, including:
  - **KMeans Clustering** using Sci-Kit Learn (in which case there will be a **feature engineering** pipeline stage prior to clustering)
  - **LLM Summary** using gpt-4o through OpenAI API
  - **LLM Analysis (WIP)** gpt-4o analysis of the data, giving more directed insights and leveraging structured outputs
  - **LLM QA (WIP)** through a simple chat UI, allowing users to have multi-round conversations about the data
- **Visualization:** Depending on which modeling stages were run, the UI can display various visualizations of the results and metrics
- **Logging/Monitoring:** Following common MLOps practices, various metrics and outputs are logged and saved in a cloud storage system (default a "mock cloud" in a local directory, but AWS S3 logging can be enabled if this repo is run locally)
- **TODO:** While not currently implemented yet, I intend to add periodic monitoring logic to the project, along with automatic parameter updating and pipeline re-deployment based on metrics and performance to follow **CI/CD principles**, continuously testing/monitoring and delivering
  - The current state of the project is meant as a proof of concept, with all of the needed features (logging and cloud storage) to eventually incorporate CI/CD through periodic data fetching, continuous metric monitoring, and managing deployment through tools like GitHub Actions
  - I have also set up and experimented with using **MLflow** for higher quality and efficient logging, however the code is currently commented out because it is meant to be a local tool. Instructions for setup are below.
 
## Running Locally

First, clone the repo with

    git clone https://github.com/simoncwang/wildlife-analyzer.git

Create a conda environment and activate it

    conda create -n "wildlife-analyzer" python=3.12
    conda activate wildlife-analyzer

Then, install the required dependencies with

    pip install -r requirements.txt

**Setup Streamlit** by creating a directory at the root of the project called `.streamlit`. In this directory create a file called `secrets.toml`. In this file put your own OpenAI API key exactly like follows:

    [openai]
    api_key = "YOURKEYHERE"

Then, to **launch the web app** simply run

    streamlit run app/dashboard.py

and the demo should open in your browser automatically.

## Usage

To use the dashboard, follow the instructions in the instruction tab. Essentially, just specify your desired location, species, and ML models to apply in the parameter settings section, then press the "Run" button in the Run Pipeline tab.

**CLI Mode:** If you would like to run the pipeline without the streamlit UI, there is a shell script in the `old_scripts` directory to run it only through command line. In this case, you must manually specify your parameters in the `pipeline_config.yaml` file.

## Setting up AWS S3 logging (optional)

- **Must be running this project cloned locally, not through my public demo**
- Create an [AWS account](https://signin.aws.amazon.com/signup?request_type=register) (if you don't already have one)
- Create an S3 bucket through the AWS console (just search for S3 in services), just follow the recommended settings, but be sure to remember the name of the bucket
- Using IAM, create a user and access key (be sure to save the secret key for later), once created, add a permission for full S3 access (can be found in the AWS managed permissions)
- If not yet installed, be sure to install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on your computer
- Then, in your terminal run `aws configure`
  - follow the prompts, entering the credentials of the user you made previously (access ID, secret key, region (for me us-east-2), default output (I used json))
- Launch dashboard like normal, then in the parameters change "Cloud Backend" to "s3", and enter the name of your s3 bucket. Now logs and outputs should be automatically uploaded to the bucket!

## Using MLflow (optional)

- To use MLflow, look through the `dashboard.py` and `models/` scripts for commented out lines using MLflow logs
- Once uncommented, run `mlflow ui` in your terminal, which will launch a UI at the specified local host URL
- Now, run the pipeline like normal, and refresh the MLflow UI to see the logged metrics and parameters
- I will add more concrete implementations and documentation in the future!

## Resources

- [Streamlit](https://streamlit.io/) for the dashboard interface.
- [iNaturalist API](https://api.inaturalist.org/v1/docs/) for fetching wildlife observations.
- [OpenAI API](https://platform.openai.com/docs/api-reference) for generating summaries.
- [Plotly](https://plotly.com/python/) for interactive visualizations.
- [Pandas](https://pandas.pydata.org/) for data manipulation.
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) from scikit-learn for clustering observations.
- [MLflow](https://mlflow.org/) for tracking runs and logging model metrics.
- [AWS](https://aws.amazon.com/) for cloud storage and services
- [Databricks: MLOps](https://www.databricks.com/glossary/mlops)

