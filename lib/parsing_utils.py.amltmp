import json


def parse_config_file(file_path: str) -> dict:
    """
    Parse a configuration file in JSON format.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    try:
        # Read the contents of the config.js file
        with open(file_path, 'r') as file:
            content = file.read()

        # Parse the content from JSON to dict
        config = json.loads(content)

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to parse JSON: {e}", e.doc, e.pos)