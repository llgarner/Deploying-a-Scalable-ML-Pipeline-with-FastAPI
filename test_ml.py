from typing import Literal
from numpy._typing._array_like import NDArray
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import apply_label
from ml.model import train_model, save_model
import pickle
from unittest.mock import patch, mock_open

@pytest.mark.parametrize("input_inference, expected_output", [
    ([1], ">50K"),  # Test case for input 1
    ([0], "<=50K"), # Test case for input 0
])
def test_apply_label(input_inference: list[int], expected_output: Literal['>50K'] | Literal['<=50K']):
    """
    Tests the apply_label function with different binary inputs.
    """
    assert apply_label(input_inference) == expected_output



def test_train_model():
    """
    Tests that the train_model function trains and returns a RandomForestClassifier.
    """
    # Arrange: Create dummy training data
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])

    # Act: Train the model
    trained_model = train_model(X_train, y_train)

    # Assert: Verify the returned object is a trained RandomForestClassifier
    assert isinstance(trained_model, RandomForestClassifier), "The returned object is not a RandomForestClassifier."
    
    # Assert: Check if the model has been fitted (e.g., check for attributes set during fitting)
    # The 'n_features_in_' attribute is set after fitting in scikit-learn models
    assert hasattr(trained_model, 'n_features_in_'), "The model has not been fitted."
    
    # Assert: Verify the parameters used to create the model
    assert trained_model.n_estimators == 100
    assert trained_model.random_state == 42
    
def test_save_model():
    """
    Tests that the save_model function calls open and pickle.dump correctly
    when saving a model to a file, using mocked file operations.
    
    """
    # Arrange: Prepare mock data and objects
    mock_model = "This is a mock model object" #  A simple object to be pickled
    test_path = "/fake/path/to/model.pkl"

    # Mock the built-in 'open' function to simulate file operations in memory
    # Use mock_open to create a mock file handle that supports context management
    with patch('builtins.open', new_callable=mock_open) as mock_file_open:
        # Mock the pickle.dump function to check if it's called with the correct arguments
        with patch('pickle.dump') as mock_pickle_dump:

            # Act: Call the save_model function
            save_model(mock_model, test_path)
            # Assert: Verify that the dependencies were called correctly
            
            # Assert that 'open' was called to write the model to the specified path
            mock_file_open.assert_called_once_with(test_path, 'wb')

            # Assert that pickle.dump was called with the model and the mock file handle
            mock_pickle_dump.assert_called_once_with(mock_model, mock_file_open())
