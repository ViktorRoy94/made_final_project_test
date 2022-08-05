import argparse
import requests
import torchaudio


def main(wav_file: str, url: str):
    waveform, sr = torchaudio.load(wav_file)
    request_json = {"data": waveform.tolist(), "sample_rate": sr}
    response = requests.get(
        f"{url}/predict/",
        json=request_json,
    )
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", "-i", required=True, help="Path to wav file")
    parser.add_argument("--url", "-u", required=False, default="http://localhost:8000",
                        help="Url of RestAPI service")
    args = parser.parse_args()

    main(args.data, args.url)
