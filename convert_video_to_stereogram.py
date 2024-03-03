import argparse
import os
from video_to_depth_frames import convert_video_to_depth_frames
from frames_to_stereogram import convert_depth_frames_to_video

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Convert videos to spectograms.")

    # Add a positional argument
    parser.add_argument("input_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_path", type=str, help="Path to the output video file.")

    # Parse the arguments
    args = parser.parse_args()

    temp_folder = "temp"
    #delete folder if it exists
    if os.path.exists(temp_folder):
        os.rmdir(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)

    convert_video_to_depth_frames(args.input_path, temp_folder)

    convert_depth_frames_to_video(temp_folder, args.output_path)

    os.rmdir(temp_folder)
    print("Done!")
    

if __name__ == "__main__":
    main()
