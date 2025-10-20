from pathlib import Path
import shutil

try:
    import kagglehub
except ImportError as e:
    raise SystemExit("Install kagglehub first: pip install kagglehub") from e

SLUG = "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def main():
    path = kagglehub.dataset_download(SLUG)
    print("Downloaded to:", path)
    copied = 0
    for ext in ("*.csv", "*.json", "*.parquet"):
        for f in Path(path).glob(ext):
            shutil.copy2(f, RAW / f.name)
            copied += 1
    print(f"Copied {copied} file(s) into {RAW}")

if __name__ == "__main__":
    main()
