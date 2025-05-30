import pandas as pd


for file in ['data/train/train-v4.csv',
             'data/train/train-v5.csv',
             'data/train/train-v6.csv']:
    df = pd.read_csv(file)
    # extract the columns of 'spg', 'wp0', 'wp1', 'wp2', 'wp3', 'wp4', 'wp5', 'wp6', 'wp7'
    columns = ['spg', 'wp0', 'wp1', 'wp2', 'wp3', 'wp4', 'wp5', 'wp6', 'wp7', 'label']#, 'energy']
    df = df[columns]
    # convert the columns to int
    for col in columns: df[col] = df[col].astype(int)

    # Get unique combinations of all columns
    unique_combinations = df.drop_duplicates()
    print(f"Unique combinations found: {len(unique_combinations)}")
    # Count the occurrences of labels
    label_counts = unique_combinations['label'].value_counts()
    print("Label counts:")
    print(len(label_counts[label_counts == 1]))
    print(len(label_counts[label_counts == 2]))
    print(label_counts[label_counts > 10])


    first_rows = pd.DataFrame()
    for label in range(1, 141):
        label_df = unique_combinations[unique_combinations['label'] == label]
        if not label_df.empty:
            first_rows = pd.concat([first_rows, label_df.iloc[[0]]])
    #print(f"First rows with labels from 1 to 140: {len(first_rows)}")
    first_rows.drop(columns=['label'], inplace=True)
    unique_combinations = first_rows.drop_duplicates()
    print(f"Unique combinations found: {len(unique_combinations)}")
    #print(unique_combinations.to_string(index=False))

    df.drop(columns=['label'], inplace=True)
    unique_combinations = df.drop_duplicates()
    print(f"Unique combinations found: {len(unique_combinations)}")
