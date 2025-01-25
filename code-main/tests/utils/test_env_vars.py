import os
import unittest

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class TestEnvironmentVariables(unittest.TestCase):
    def test_env_vars_are_set(self):
        required_env_vars = [
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
            "HUGGINGFACE_TOKEN",
            "OPENAI_API_KEY",
            "WANDB_API_TOKEN",
        ]

        for var in required_env_vars:
            with self.subTest(env_var=var):
                value = os.getenv(var)
                self.assertIsNotNone(value, f"Environment variable {var} is not set.")
                self.assertNotEqual(value, "", f"Environment variable {var} is empty.")


if __name__ == "__main__":
    unittest.main()
