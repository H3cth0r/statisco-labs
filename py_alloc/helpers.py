import os

def getenv(key: str, default=0): 
    """
    Retrieves an environment variables value, casting it to the same type as
    default.

    Parameters:
    - Key (str): The name of the environment variable.
    - default (Any): default value to return if the variable is not set.

    Returns:
    - Any: Value of the environment variable of the default.
    """
    return type(default)(os.getenv(key, default))
