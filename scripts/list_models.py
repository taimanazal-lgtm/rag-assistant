"""List the models available from the provider (free model IDs change often)."""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/models", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/models", "OPENROUTER_API_KEY"),
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "groq"
    url, key_var = PROVIDERS[name]
    key = os.getenv(key_var)
    if not key:
        sys.exit(f"{key_var} is missing in .env")

    resp = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    resp.raise_for_status()
    models = resp.json()["data"]

    if name == "openrouter":  # keep only free models
        models = [m for m in models if float(m.get("pricing", {}).get("prompt", 1)) == 0]

    for m in sorted(models, key=lambda m: m["id"]):
        print(m["id"])
    print(f"\n{len(models)} models from {name}")


if __name__ == "__main__":
    main()