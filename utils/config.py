# paper_agent/utils/config.py

import os
import json


class Config:
    """
    Manages application configuration.
    Loads settings from a 'config.json' file.
    """

    _instance = None
    _config_data = {}
    _config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        Loads configuration from the config.json file.
        If the file doesn't exist, it creates a default one.
        """
        if not os.path.exists(self._config_path):
            self._create_default_config()

        try:
            with open(self._config_path, "r") as f:
                self._config_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: config.json is malformed. Creating a default config at {self._config_path}")
            self._create_default_config()
            self._load_config()  # Try loading again

    def _create_default_config(self):
        """
        Creates a default config.json file with placeholders.
        """
        default_config = {
            "DATA_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
            "PAPERS_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "papers"),
            "DB_DIR": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "db"),
            "DATABASE_NAME": "paper_agent.db",
            "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",  # A good default for local embeddings
            "LLM_API_KEY": "YOUR_OPENAI_API_KEY_OR_OTHER_LLM_KEY",  # Placeholder for LLM API key
            "LLM_MODEL_NAME": "gpt-4o-mini",  # Or "gpt-3.5-turbo", "llama3", etc.
            "LOG_LEVEL": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
            "LOG_FILE": os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_agent.log"),
        }
        with open(self._config_path, "w") as f:
            json.dump(default_config, f, indent=4)
        print(f"Default config.json created at {self._config_path}. Please review and update it.")
        self._config_data = default_config  # Update current data

    def get(self, key, default=None):
        """
        Retrieves a configuration value by key.
        """
        return self._config_data.get(key, default)

    def set(self, key, value):
        """
        Sets a configuration value and saves it to the file.
        """
        self._config_data[key] = value
        self._save_config()

    def _save_config(self):
        """
        Saves the current configuration data back to the config.json file.
        """
        try:
            with open(self._config_path, "w") as f:
                json.dump(self._config_data, f, indent=4)
        except IOError as e:
            print(f"Error saving config file: {e}")


# Initialize the config instance for easy access throughout the application
config = Config()

if __name__ == "__main__":
    # Example usage and testing the config module
    print("--- Testing Config Module ---")
    cfg = Config()  # Or just use 'config' directly after import

    print(f"PAPERS_DIR: {cfg.get('PAPERS_DIR')}")
    print(f"DATABASE_NAME: {cfg.get('DATABASE_NAME')}")
    print(f"LLM_API_KEY: {cfg.get('LLM_API_KEY')}")

    # Test setting a value
    cfg.set("TEST_KEY", "test_value")
    print(f"TEST_KEY: {cfg.get('TEST_KEY')}")

    # Verify it's saved by loading a new instance (or restarting)
    # For this script run, it will reference the same singleton,
    # but the file will be updated.
    print("Config file should now be updated with TEST_KEY.")
    print("Please check paper_agent/config.json manually.")

    # Clean up test key if you want
    # del cfg._config_data['TEST_KEY']
    # cfg._save_config()
