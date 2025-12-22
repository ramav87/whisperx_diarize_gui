import os
import argparse
from diarize_gui.openai_provider import OpenAIProvider

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--prompt", default="Return exactly: OK")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment first.")

    client = OpenAIProvider(api_key=api_key, model=args.model)
    out = client.analyze(args.prompt)
    print("\n--- OpenAI output ---\n")
    print(out)
    print("\n---------------------\n")

if __name__ == "__main__":
    main()
