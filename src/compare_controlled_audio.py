import sys
import librosa
import numpy as np


def analyze(path):

    y, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    rms = float(
        np.sqrt(np.mean(y ** 2))
    )

    peak = float(
        np.max(np.abs(y))
    )

    centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )
    )

    bandwidth = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=y,
                sr=sr
            )
        )
    )

    rolloff = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=y,
                sr=sr
            )
        )
    )

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(y)
        )
    )

    return {
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr
    }


def print_result(name, result):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"RMS              : {result['rms']:.6f}")
    print(f"Peak             : {result['peak']:.6f}")
    print(f"Spectral centroid: {result['centroid']:.2f} Hz")
    print(f"Bandwidth        : {result['bandwidth']:.2f} Hz")
    print(f"Rolloff          : {result['rolloff']:.2f} Hz")
    print(f"ZCR              : {result['zcr']:.6f}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage:"
        )
        print(
            "python src\\compare_controlled_audio.py "
            "\"reference.wav\" \"controlled.wav\""
        )
        sys.exit(1)

    reference = sys.argv[1]
    controlled = sys.argv[2]

    ref_result = analyze(reference)
    controlled_result = analyze(controlled)

    print_result(
        "REFERENCE CAR HORN",
        ref_result
    )

    print_result(
        "CONTROLLED MICROPHONE CAR HORN",
        controlled_result
    )

    print("\n" + "=" * 50)
    print("DIFFERENCES")
    print("=" * 50)

    print(
        f"RMS difference       : "
        f"{controlled_result['rms'] - ref_result['rms']:.6f}"
    )

    print(
        f"Peak difference      : "
        f"{controlled_result['peak'] - ref_result['peak']:.6f}"
    )

    print(
        f"Centroid difference  : "
        f"{controlled_result['centroid'] - ref_result['centroid']:.2f} Hz"
    )

    print(
        f"Bandwidth difference : "
        f"{controlled_result['bandwidth'] - ref_result['bandwidth']:.2f} Hz"
    )

    print(
        f"Rolloff difference    : "
        f"{controlled_result['rolloff'] - ref_result['rolloff']:.2f} Hz"
    )

    print(
        f"ZCR difference       : "
        f"{controlled_result['zcr'] - ref_result['zcr']:.6f}"
    )