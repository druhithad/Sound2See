import librosa
import numpy as np


CAR_HORN = r"C:\Users\D LAHARI\Downloads\ESC-50-master\ESC-50-master\audio\5-179860-A-43.wav"
LIVE = r"data\inference\live_test.wav"


def analyze(name, filename):

    audio, sr = librosa.load(
        filename,
        sr=16000,
        mono=True
    )

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Sample rate        :", sr)
    print("Samples            :", len(audio))
    print("Duration           :", len(audio) / sr)

    print("\nAMPLITUDE")
    print("RMS                :", round(float(rms), 6))
    print("Peak               :", round(float(peak), 6))
    print("Crest factor       :", round(float(peak / (rms + 1e-12)), 4))

    print("\nSPECTRAL")
    print(
        "Spectral centroid  :",
        round(float(np.mean(centroid)), 2),
        "Hz"
    )

    print(
        "Spectral bandwidth :",
        round(float(np.mean(bandwidth)), 2),
        "Hz"
    )

    print(
        "Spectral rolloff   :",
        round(float(np.mean(rolloff)), 2),
        "Hz"
    )

    print("\nTEMPORAL")
    print(
        "Zero crossing rate :",
        round(float(np.mean(zcr)), 6)
    )

    return {
        "rms": rms,
        "peak": peak,
        "centroid": np.mean(centroid),
        "bandwidth": np.mean(bandwidth),
        "rolloff": np.mean(rolloff),
        "zcr": np.mean(zcr)
    }


print("=" * 55)
print(" SOUND2SEE AUDIO PROPERTY COMPARISON")
print("=" * 55)

car = analyze(
    "REFERENCE: CORRECT CAR HORN",
    CAR_HORN
)

live = analyze(
    "LIVE MICROPHONE RECORDING",
    LIVE
)


print("\n" + "=" * 55)
print(" COMPARISON")
print("=" * 55)

print(
    "\nRMS ratio (live/reference):",
    round(live["rms"] / (car["rms"] + 1e-12), 4)
)

print(
    "Peak ratio:",
    round(live["peak"] / (car["peak"] + 1e-12), 4)
)

print(
    "Centroid difference:",
    round(
        abs(live["centroid"] - car["centroid"]),
        2
    ),
    "Hz"
)

print(
    "Bandwidth difference:",
    round(
        abs(live["bandwidth"] - car["bandwidth"]),
        2
    ),
    "Hz"
)

print(
    "Rolloff difference:",
    round(
        abs(live["rolloff"] - car["rolloff"]),
        2
    ),
    "Hz"
)

print(
    "ZCR difference:",
    round(
        abs(live["zcr"] - car["zcr"]),
        6
    )
)

print("\n" + "=" * 55)
print(" COMPARISON COMPLETE")
print("=" * 55)