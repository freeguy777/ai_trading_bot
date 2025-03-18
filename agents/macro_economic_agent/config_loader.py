# agents/macro_economic_agent/config_loader.py

import os
from typing import Dict
from dotenv import load_dotenv


class ConfigLoader:
    """
    Loads API keys and configuration from a .env file.

    Example:
        config = ConfigLoader.load()
        print(config["FMP_API_KEY"])
    """

    REQUIRED_KEYS = [
        "FMP_API_KEY",
        "NEWS_API_KEY",
        "GEMINI_API_KEY",
        "KIS_API_KEY",
        "KIS_API_SECRET",
    ]

    @classmethod
    def load(cls) -> Dict[str, str]:
        """
        Load environment variables from .env file and validate required keys.

        Returns:
            A dictionary containing API keys and settings.

        Raises:
            EnvironmentError: If any required environment variable is missing.
        """

        load_dotenv(dotenv_path="config/.env")  # 상대 경로 사용

        config = {}
        missing_keys = []

        for key in cls.REQUIRED_KEYS:
            value = os.getenv(key)
            if value is None:
                missing_keys.append(key)
            else:
                config[key] = value

        if missing_keys:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_keys)}"
            )

        return config
