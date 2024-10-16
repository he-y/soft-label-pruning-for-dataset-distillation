import pandas as pd
from aim import Repo
from tqdm import tqdm
import os

my_repo = Repo(".") # TOOD: specify the path to your aim repo

def query_data(tag, excel_name='query/supp.xlsx', name_contains=None):
    data = []
    for run in tqdm(my_repo.iter_runs()):
        try:
            # skip archived runs
            if run.archived:
                continue

            if run.tags[0] == f'{tag}':
                # check if the run name contains the string
                if name_contains is not None:
                    if name_contains not in run.name:
                        continue

                # Initialize a dictionary to store run details
                run_data = {'name': run.name, 'accuracy': None, 'experiment': run.experiment}

                for metric in run.collect_sequence_info('metric')['metric']:
                    if (metric['name'] == 'accuracy') and (metric['context']['subset'] == 'test'):
                        run_data['accuracy'] = metric['last_value']['last'] # Update accuracy

                # Add the run data to the list
                data.append(run_data)

        except Exception as e:
            continue

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # post-processing (compute mean and std)
    # Assuming df is your DataFrame after loading the CSV
    df['base_name'] = df['name'].str.replace(r'_re\d+', '', regex=True)

    # Group by the base name and 'experiment' column, and calculate the mean, standard deviation, and count
    summary = df.groupby(['base_name', 'experiment'])['accuracy'].agg(['mean', 'std', 'size']).reset_index()

    # Rename 'size' column to 'run_count' to indicate the number of repeated runs
    summary.rename(columns={'size': 'run_count'}, inplace=True)

    print(summary.head())
    # save the summary to excel with sheet name as tag
    if not os.path.exists(excel_name):
        with pd.ExcelWriter(excel_name) as writer:
            summary.to_excel(writer, sheet_name=tag)
    else:
        with pd.ExcelWriter(excel_name, mode='a') as writer:
            summary.to_excel(writer, sheet_name=tag)

    print(f"Data saved to query/{tag}.csv")

# methods = ['glide_t5', 'glide_llama', 'sd_t5', 'sd_llama']
# pr = ['0.5', '1']

methods = ['glide_t5']
prs = ['1']
name_contains = None
for method in methods:
    for pr in prs:
        # tag = f'{method}_pr{pr_}_supp'
        tag = f'{method}_pr{pr}_table5'
        query_data(tag, excel_name=f'query/{tag}.xlsx', name_contains=name_contains)