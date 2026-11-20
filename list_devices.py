import sounddevice as sd

def main():
    print("Available Audio Input Devices:\n")
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            print(
                f"[{idx}] {dev['name']} "
                f"(Inputs: {dev['max_input_channels']}, "
                f"Default Rate: {int(dev['default_samplerate'])} Hz)"
            )

if __name__ == "__main__":
    main()
