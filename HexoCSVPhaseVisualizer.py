import sys
import os
import csv
import pandas as pd


def indexCreate(retDF):
    index = 0
    indexDict = {}
    for subject in retDF["Subject_ID"]:
        indexDict[subject] = index
        index+=1
    return indexDict

def visualize(dPath):
    retDF = pd.DataFrame(columns=["Subject_ID", "Baseline", "Phase_1", "Phase_2", "Phase_3", "Phase_4"])
    retDF.fillna("NA", inplace=True)
    anCSV = pd.read_csv(dPath, usecols=["Subject_ID", "Data_Phase"])
    retDF["Subject_ID"] = anCSV["Subject_ID"].value_counts().sort_index(ascending=True).keys()
    indexes = indexCreate(retDF)
    
    for index, row in anCSV.iterrows():
        subject = row[0]
        phase = row[1]
        if (phase == "0"):
            retDF.at[indexes[subject], "Baseline"] = "X"
        if(phase == "1"):
            retDF.at[indexes[subject], "Phase_1"] = "X"
        if(phase == "2"):
            retDF.at[indexes[subject], "Phase_2"] = "X"
        if(phase == "3"):
            retDF.at[indexes[subject], "Phase_3"] = "X"
        if(phase == "4"):
            retDF.at[indexes[subject], "Phase_4"] = "X"

    rootPath = os.path.dirname(dPath)
    writePath = os.path.join(rootPath, "pyPhaseVisual.csv")
    retDF.to_csv(writePath, index=False)

    return

def main():
    while (True):
        print(
                "\n<-------------------------------------------------------------------->\n")
        dPath = input("\nInput path to pyAnalysis file for visualization, or exit (e)\n>> ")
        if (dPath.lower() == "e"):
            print("\nGoodBye\n")
            print(
                "\n<-------------------------------------------------------------------->\n")
            break
        elif(os.path.exists(dPath)):
            visualize(dPath)
        else:
            print("\nERROR: Unknown file, please retry\n")

    return


if __name__ == "__main__":
    sys.exit(main())