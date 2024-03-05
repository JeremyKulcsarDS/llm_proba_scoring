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

    Config File Format:
        The configuration file should be in JSON format and include the following keys:

        For config.json file:
            - "key_vault_name": Name of the key vault.
            - "managed_identity_client_id": Client ID of the managed identity.
            - "secret_key": Secret key for accessing the API.
            - "api_type": Type of the API.
            - "api_base": Base URL of the API.
            - "api_version": Version of the API.

        For model.json file:
            - "gpt_model": GPT model information.
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