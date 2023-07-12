# Author: Ishaan Ghosh
# File Desc: This is a simple Hexoskin Wav to CSV converter. Users can convert single files, or a
#            directory of files. The whole program runs in paths so if you are asked to input a
#            filename input either the direct filepath or the relative path to this python file.
#
# DISCLAIMER: - You will need this library: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiZ_bC69uf4AhVnDkQIHRicCaEQFnoECBcQAQ&url=https%3A%2F%2Ffiles.hexoskin.com%2FHxConvertSourceFile%2Fpython%2FLATEST%2Fload_hx_data.py&usg=AOvVaw1dS-kMRMc6reclBPmijheQ.
#             - This script uses paths, therefore the PER-FILE conversion only works on the following machines: Windows, MacOS, Linux
#             - The PER-DIRECTORY conversion functions on all machines that run python
#             - Be careful of overwriting csv files, this program replaces csv files of the
#               same name in the working directory.

import sys
import os
import load_hx_data
import platform
import csv


# Global variable the contains the user operating system
uOS = platform.system()


# Helper function that converts a WAV to CSV file, taking the read file path and the
# path to write to as parameters.
def convert(rPath, wPath, startTime):
    file = open(wPath, "w+")
    data = load_hx_data.load_data(rPath, 0, 0, None, False)
    for tup in data:
        fullTime = str(float(tup[0]) + float(startTime))
        file.write(fullTime + "," + ','.join(str(s) for s in tup) + '\n')
    file.close()
    if (os.stat(wPath).st_size == 0):
        os.remove(wPath)
        print(
            "\nERROR:Read File not found or empty, please check filepath: " + rPath + "\n")
    else:
        print("\nFile (" + wPath + ") has been created/replaced\n")

# Helper function the handles converting on a file-by-file basis


def perFile():
    while(True):
        rPath = input(
            "Input WAV File Relative Path, or back to menu (e)\n>> ")
        if (rPath.lower() == "e"):
            print(
                "\n<-------------------------------------------------------------------->\n")
            return
        startTime = input(
            "\nCannot determine start time in file-by-file, please provide the EXACT start timestamp (POSIX-like, not date) from respective statistics.csv file\n>> ")
        startTime = float(float(startTime) / 256)
        wPathList = []
        if (uOS == "Windows"):
            wPathList = rPath.split("\\")
        elif (uOS == "Darwin" or uOS == "Linux"):
            wPathList = rPath.split("/")
        wPathList[-1] = wPathList[-1].split(".")[0] + ".csv"
        if (uOS == "Windows"):
            wPath = "\\".join(wPathList)
        elif (uOS == "Darwin" or uOS == "Linux"):
            wPath = "/".join(wPathList)
        else:
            print("Unknown Operating System, cannot traverse files correctly.\n")
            return
        convert(rPath, wPath, startTime)

# Helper function that converts all the WAV files in a directory to CSV files


def perDir():
    dPath = input("Input Directory Relative Path or return to menu (e)\n>> ")
    if (dPath.lower() == "e"):
        print(
            "\n<-------------------------------------------------------------------->\n")
        return
    for root, dirs, files in os.walk(dPath):
        statsFilePath = os.path.join(root, "statistics.csv")
        startTime = 0
        if (not os.path.exists(statsFilePath)):
            print("Could not find stats file (" + statsFilePath +
                  ") in working directory, continuing to subdirectories\n")
            continue
        with open(statsFilePath, mode='r') as stats:
            statsReader = csv.DictReader(stats)
            line_count = 0
            for row in statsReader:
                if (row["start"] is not None and row["start"].isnumeric):
                    startTime = row["start"]
                line_count += 1
        startTime = float(float(startTime) / 256)
        print("Statistics file used for timestamp: " + statsFilePath)
        for filename in files:
            if ((filename[-3:]) == "wav"):
                filenameNew = os.path.join(root, filename)
                csvName = os.path.join(root, filename.split(".")[0] + ".csv")
                convert(filenameNew, csvName, startTime)
    print(
        "\n<-------------------------------------------------------------------->\n")


def main():
    load_hx_data.set_device_model("hx")

    while (True):
        fOrD = input(
            "Do you want to convert a file (f) or directory of files (d) or exit (e)\n>> ")
        if (fOrD.lower() == "f"):
            print(
                "\n<-------------------------------------------------------------------->\n")
            perFile()
        elif (fOrD.lower() == "d"):
            print(
                "\n<-------------------------------------------------------------------->\n")
            perDir()
        elif (fOrD.lower() == "e"):
            print("\nGoodBye\n")
            print(
                "\n<-------------------------------------------------------------------->\n")
            break
        else:
            print("Input Error, please retry\n")


if __name__ == "__main__":
    sys.exit(main())
