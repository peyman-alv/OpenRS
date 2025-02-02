from typing import Tuple, Union
import os

import numpy as np
import h5py
from scipy.sparse import csc_matrix


def load_matlab_file(path: str, field_name: str) -> Union[np.ndarray, csc_matrix]:
    """Load a field from a MATLAB .mat file."""
    with h5py.File(path, "r") as db:
        if field_name not in db:
            raise KeyError(f"Field '{field_name}' not found in the file.")
        ds = db[field_name]

        # Check if the field has the sparse matrix components.
        if isinstance(ds, h5py.Group) and all(
            k in ds.keys() for k in ["data", "ir", "jc"]
        ):
            data = np.asarray(ds["data"])
            ir = np.asarray(ds["ir"])
            jc = np.asarray(ds["jc"])
            return csc_matrix((data, ir, jc)).astype(np.float32)

        # For dense arrays, convert to float32 and transpose.
        return np.asarray(ds).astype(np.float32).T


def build_rating_matrix(data: np.ndarray, n_users: int, n_movies: int) -> np.ndarray:
    """Build a rating matrix from the data."""
    # Initialize the rating matrix with zeros
    rating_matrix = np.zeros((n_movies, n_users), dtype="float32")
    # Convert user and movie indices to 0-based indexing
    users = data[:, 0].astype(int) - 1
    movies = data[:, 1].astype(int) - 1
    ratings = data[:, 2]
    # Fill the rating matrix
    rating_matrix[movies, users] = ratings
    return rating_matrix


def create_mask_matrix(matrix: np.ndarray) -> np.ndarray:
    """Creates a binary mask where ratings exist."""
    return (matrix > 0).astype("float32")


def load_data_100k(
    path: str, delimiter: str = "\t"
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the MovieLens 100k dataset."""
    train = np.loadtxt(
        os.path.join(path, "movielens_100k_u1.base"), skiprows=0, delimiter=delimiter
    ).astype("int32")
    test = np.loadtxt(
        os.path.join(path, "movielens_100k_u1.test"), skiprows=0, delimiter=delimiter
    ).astype("int32")

    total = np.concatenate((train, test), axis=0)
    n_u = np.unique(total[:, 0]).size  # num of users
    n_m = np.unique(total[:, 1]).size  # num of movies

    # Build the rating matrices
    train_r = build_rating_matrix(train, n_u, n_m)
    test_r = build_rating_matrix(test, n_u, n_m)

    # Create mask matrices indicating where ratings are non-zero
    train_m = create_mask_matrix(train_r)
    test_m = create_mask_matrix(test_r)

    # Print the dataset statistics
    print(".~" * 40)
    print("Number of users: {}".format(n_u))
    print("Number of movies: {}".format(n_m))
    print("Number of training ratings: {}".format(train.shape[0]))
    print("Number of test ratings: {}".format(test.shape[0]))
    print(".~" * 40)

    return n_m, n_u, train_r, train_m, test_r, test_m


def load_data_1m(
    path: str, delimiter: str = "::", frac: float = 0.1, seed: int = 1234
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the MovieLens 1M dataset and split into train/test sets."""
    data = data = np.genfromtxt(
        os.path.join(path, "movielens_1m_dataset.dat"),
        delimiter=delimiter,
        dtype="int32",
    )

    n_u = np.unique(data[:, 0]).size  # Number of users
    n_m = np.unique(data[:, 1]).size  # Number of movies
    n_r = data.shape[0]  # Number of ratings

    # Map user and movie IDs to a contiguous 0-based index
    _, user_indices = np.unique(data[:, 0], return_inverse=True)
    _, movie_indices = np.unique(data[:, 1], return_inverse=True)

    # Shuffle the data
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(n_r)
    split_idx = int(frac * n_r)

    # Split the data into train and test sets
    test_data = np.column_stack(
        (
            user_indices[shuffled_indices[:split_idx]],
            movie_indices[shuffled_indices[:split_idx]],
            data[shuffled_indices[:split_idx], 2],
        )
    )

    train_data = np.column_stack(
        (
            user_indices[shuffled_indices[split_idx:]],
            movie_indices[shuffled_indices[split_idx:]],
            data[shuffled_indices[split_idx:], 2],
        )
    )

    # Build the rating matrices
    train_r = build_rating_matrix(train_data, n_u, n_m)
    test_r = build_rating_matrix(test_data, n_u, n_m)

    # Generate the binary mask matrices
    train_m = create_mask_matrix(train_r)
    test_m = create_mask_matrix(test_r)

    # Print the dataset statistics
    print(".~" * 40)
    print("Number of users: {}".format(n_u))
    print("Number of movies: {}".format(n_m))
    print("Number of training ratings: {}".format(train_data.shape[0]))
    print("Number of test ratings: {}".format(test_data.shape[0]))
    print(".~" * 40)

    return n_m, n_u, train_r, train_m, test_r, test_m


def load_data_monti(
    path: str,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Douban Monti dataset."""
    dataset_path = os.path.join(path, "douban_monti_dataset.mat")
    M = load_matlab_file(dataset_path, "M")
    Otraining = load_matlab_file(dataset_path, "Otraining") * M
    Otest = load_matlab_file(dataset_path, "Otest") * M

    n_u, n_m = M.shape  # Number of users and movies
    n_train = np.count_nonzero(Otraining)  # Number of training ratings
    n_test = np.count_nonzero(Otest)  # Number of test ratings

    # Build the rating matrices
    train_r = Otraining.T
    test_r = Otest.T

    # Generate the binary mask matrices
    train_m = create_mask_matrix(train_r)
    test_m = create_mask_matrix(test_r)

    # Print the dataset statistics
    print(".~" * 40)
    print("Number of users: {}".format(n_u))
    print("Number of movies: {}".format(n_m))
    print("Number of training ratings: {}".format(n_train))
    print("Number of test ratings: {}".format(n_test))
    print(".~" * 40)

    return n_m, n_u, train_r, train_m, test_r, test_m


if __name__ == "__main__":
    # Set the project root directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Set the paths to the datasets
    MOVIELENS_100K_PATH = os.path.join(PROJECT_ROOT, "data", "MovieLens_100K")
    MOVIELENS_1M_PATH = os.path.join(PROJECT_ROOT, "data", "MovieLens_1M")
    DOUBAN_MONTI_PATH = os.path.join(PROJECT_ROOT, "data", "DoubanMonti")

    # Load the MovieLens 100k dataset
    n_m, n_u, train_r, train_m, test_r, test_m = load_data_100k(
        path=MOVIELENS_100K_PATH,
    )

    n_m, n_u, train_r, train_m, test_r, test_m = load_data_1m(
        path=MOVIELENS_1M_PATH,
    )

    n_m, n_u, train_r, train_m, test_r, test_m = load_data_monti(
        path=DOUBAN_MONTI_PATH,
    )
