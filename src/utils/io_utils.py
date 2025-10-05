import pickle
from typing import Any, List, Tuple, Union


def read_txt_file(filename: str) -> List[Tuple[float, Union[float, List[float]]]]:
    """
    Read a text file with comma or space-separated values.
    
    Each line should contain a timestamp followed by one or more values.
    
    Args:
        filename: Path to the text file.
        
    Returns:
        List of tuples where each tuple contains (timestamp, value(s)).
        If a single value follows the timestamp, it's returned as a float.
        If multiple values follow, they're returned as a list of floats.
    """
    data = []
    
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().replace(",", " ").split()
            timestamp = float(values[0].strip())
            values_list = [float(value.strip()) for value in values[1:]]
            values_list = values_list[0] if len(values_list) == 1 else values_list

            data.append((timestamp, values_list))
    
    return data


def save_pickle(data: Any, filename: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data object to be pickled and saved.
        filename: Path to the output pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_pickle(filename: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filename: Path to the input pickle file.

    Returns:
        The unpickled data object.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)
    