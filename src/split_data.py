import numpy as np
from pathlib import Path
from src.utils import ecrire_csv

def splitting_phase(dataset_file, train_ratio):
    """
    Splits the dataset into training, validation, and test sets based on the given train ratio.
    
    Splitting phase :
            python mlp.py --dataset <dataset_file> --split <train_ratio>
        return :
            > splitting dataset '<dataset_file>' into train, validation & test sets with train ratio = <train_ratio>
    """

    if not isinstance(dataset_file, np.ndarray) or not isinstance(train_ratio, tuple):
        print("Error: dataset_file should be a numpy array.")
        return None, None, None
    
    np.random.shuffle(dataset_file) # Shuffle the dataset before splitting

    total_samples = dataset_file.shape[0]
    train_size = int(total_samples * train_ratio[0])
    valid_size = int(total_samples * train_ratio[1])

    train_set = dataset_file[:train_size]
    valid_set = dataset_file[train_size:train_size + valid_size]
    test_set = dataset_file[train_size + valid_size:]

    dossier = Path("datasets")
    dossier.mkdir(exist_ok=True) # Create directory if it doesn't exist

    ecrire_csv(dossier / "train_set.csv", train_set.tolist())
    ecrire_csv(dossier / "valid_set.csv", valid_set.tolist())
    ecrire_csv(dossier / "test_set.csv", test_set.tolist())

    print(f"  > train_set shape : {train_set.shape}")
    print(f"  > valid_set shape : {valid_set.shape}")
    print(f"  > test_set  shape : {test_set.shape}")
    print(f"  > total_set shape : ({train_set.shape[0] + valid_set.shape[0] + test_set.shape[0]}, {train_set.shape[1]})")

    return train_set, valid_set, test_set
