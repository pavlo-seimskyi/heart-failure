import os


def get_base_path():
    """Get the absolute path to the project folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
