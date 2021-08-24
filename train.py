# !pip install pandas

from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import pandas
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# +
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '510b94ba-e453-4417-988b-fbdc37b55ca7'
resource_group = 'aml-quickstarts-155605'
workspace_name = 'quick-starts-ws-155605'

workspace = Workspace(subscription_id, resource_group, workspace_name)

# dataset = Dataset.get_by_name(workspace, name='HeartFailureDataset')
# df = dataset.to_pandas_dataframe()
# -

url = 'https://github.com/BusolaAnu/CapstoneProject/raw/main/heart_failure_clinical_records_dataset.csv'
df = TabularDatasetFactory.from_delimited_files(url)

df

# TODO: Split data into train and test sets.

# ## YOUR CODE HERE ###a

data = df.to_pandas_dataframe()

run = Run.get_context()

x = data
y = x.pop("DEATH_EVENT")

# +
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=223)


# +
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")


    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
# 'quit'
# -





