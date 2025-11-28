"""Utility script to fetch the fruits dataset from Kaggle."""
import argparse
import pathlib
import subprocess
import zipfile


def run(cmd):
    result = subprocess.run(cmd, shell=True, check=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="data/fruits", help="Local path for the dataset")
    args = parser.parse_args()

    target = pathlib.Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    zip_path = target / "fruits-images-dataset-object-detection.zip"
    kaggle_cmd = (
        "kaggle datasets download afsananadia/fruits-images-dataset-object-detection "
        f"-p {target.as_posix()} -o"
    )
    run(kaggle_cmd)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target)


if __name__ == "__main__":
    main()
