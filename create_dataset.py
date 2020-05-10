import argparse
import os
import os.path
from shutil import rmtree, move
import random

# For parsing commandline arguments
parser = argparse.ArgumentParser()

parser.add_argument("--videos_folder", type=str, required=True, help='path to the folder containing videos')
parser.add_argument("--dataset_folder", type=str, required=True, help='path to the output dataset folder')

parser.add_argument("--img_width", type=int, default=1280, help="output image width")
parser.add_argument("--img_height", type=int, default=720, help="output image height")

args = parser.parse_args()


def extract_frames(videos, inDir, outDir):
    """
    Converts all the videos passed in `videos` list to images.
    Parameters
    ----------
        videos : list
            name of all video files.
        inDir : string
            path to input directory containing videos in `videos` list.
        outDir : string
            path to directory to output the extracted images.
    Returns
    -------
        None
    """

    for video in videos:
        os.mkdir(os.path.join(outDir, os.path.splitext(video)[0]))
        retn = os.system(
            'ffmpeg -i "{}" -vf scale={}:{} -vsync 0 -qscale:v 2 "{}/%08d.png"'.format(os.path.join(inDir, video),
                                                                                       args.img_width, args.img_height,
                                                                                       os.path.join(outDir,
                                                                                                    os.path.splitext(
                                                                                                        video)[
                                                                                                        0])))
        if retn:
            print("Error converting file:{}. Exiting.".format(video))


def create_clips(root, destination):
    """
    Distributes the images extracted by `extract_frames()` in
    clips containing 3 frames each.
    TODO: Extract 12 frames. Need to implement code to interpolate several frames first.
    Parameters
    ----------
        root : string
            path containing extracted image folders.
        destination : string
            path to output clips.
    Returns
    -------
        None
    """

    folderCounter = -1

    files = os.listdir(root)

    # Iterate over each folder containing extracted video frames.
    for file in files:
        images = sorted(os.listdir(os.path.join(root, file)))

        for imageCounter, image in enumerate(images):
            # Bunch images in groups of 12 frames
            if imageCounter % 3 == 0:
                if imageCounter + 3 >= len(images):
                    break

                folderCounter += 1
                os.mkdir("{}/{}".format(destination, folderCounter))
            move("{}/{}/{}".format(root, file, image), "{}/{}/{}".format(destination, folderCounter, image))
        rmtree(os.path.join(root, file))


def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    extractPath = os.path.join(args.dataset_folder, "extracted")
    trainPath = os.path.join(args.dataset_folder, "train")

    os.mkdir(extractPath)
    os.mkdir(trainPath)

    # Create list of video names
    trainVideoNames = os.listdir(args.videos_folder)

    # Create train-test dataset
    extract_frames(trainVideoNames, args.videos_folder, extractPath)
    create_clips(extractPath, trainPath)

    rmtree(extractPath)


main()
