import argparse
import urllib.request
import coremltools as ct

DEFAULT_URL = "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/aiy/vision/classifier/birds_V1/3.tflite"


def download_model(url, path):
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def convert(tflite_path, output_path):
    print(f"Converting {tflite_path} to {output_path}")
    mlmodel = ct.convert(tflite_path, source="tensorflow")
    mlmodel.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert Google AIY Birds model to CoreML")
    parser.add_argument("--tflite-url", default=DEFAULT_URL, help="URL of the TFLite model")
    parser.add_argument("--output", default="aiy_birds.mlmodel", help="Output CoreML model filename")
    args = parser.parse_args()

    tflite_path = "model.tflite"
    download_model(args.tflite_url, tflite_path)
    convert(tflite_path, args.output)


if __name__ == "__main__":
    main()

