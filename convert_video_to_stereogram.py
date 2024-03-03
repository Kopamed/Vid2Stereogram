import argparse


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some input.")

    # Add a positional argument
    parser.add_argument("input_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_path", type=str, help="Path to the output video file.")

    # Parse the arguments
    args = parser.parse_args()

    


if __name__ == "__main__":
    main()
