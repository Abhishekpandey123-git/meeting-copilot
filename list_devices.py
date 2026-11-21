import sounddevice as sd

def main():
    print("Available Audio Input Devices:\n")
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            hostapi = sd.query_hostapis(dev['hostapi'])['name']
            print(
                f"[{idx}] {dev['name']} "
                f"[{hostapi}] "
                f"(Inputs: {dev['max_input_channels']}, "
                f"Rate: {int(dev['default_samplerate'])} Hz)"
            )

if __name__ == "__main__":
    main()
