import os
import yaml
import requests
from pathlib import Path
from tqdm import tqdm


def get_model_urls(config_file: str) -> dict:
    """Returns the model urls.

    Parameters
    ----------
        config_file : Path the configuration file

    Returns
    -------
        config_data (dict): the configuration data

    Usage
    -----
        >>> config_data = get_config_data(config_file)
    """
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data['urls']


def create_directory_if_not_exists(path: str):
    """Creates a directory if it does not exist.

    Parameters
    ----------
        path: the path to the directory

    Usage
    -----
        >>> path = 'path/to/directory'
        >>> create_directory_if_not_exists(path)
    """
    directory = Path(path)
    if not directory.exists():
        directory.mkdir(parents=True)

def download_models(config_file: str, destination: str="models"):
    """Downloads modesl files.

    Parameters
    ----------
        config_file: the path to the configuration file
        destination: the folder to save the file to.
                     defaults to 'models'.

    Usage
    -----
        >>> config_file = 'path/to/configs.yaml'
        >>> destination = 'my-path'
        >>> download_file(url, destination)
    """
    urls = get_model_urls(config_file)
    for model_name, model_url in urls.items():
        file_name = Path(model_url).name
        target_directory = Path(destination) / model_name

        create_directory_if_not_exists(target_directory)
        destination_path = Path(target_directory) / file_name
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        file_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name)
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()




if __name__ == '__main__':
    config_file = Path('../../configs.yaml')
    download_models(config_file)
