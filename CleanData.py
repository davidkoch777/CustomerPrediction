import csv
import numpy as np
from numpy.random import choice


laender = range(5)
laenderschluessel = ["" for x in range(len(laender))]
laenderschluessel[0] = "Deutschland"
laenderschluessel[1] = "Frankreich"
laenderschluessel[2] = "Großbritanien"
laenderschluessel[3] = "USA"
laenderschluessel[4] = "China"

laenderdist = [0.4, 0.07, 0.03, 0.2, 0.3]

branchen = range(3)
branchenschluessel = ["" for x in range(len(branchen))]
branchenschluessel[0] = "Industrie"
branchenschluessel[1] = "Bauwesen"
branchenschluessel[2] = "Energie"

branchendistproland = [[0 for x in range(3)] for y in range(5)]
branchendistproland[0] = [0.6, 0.2, 0.2]
branchendistproland[1] = [0.15, 0.15, 0.7]
branchendistproland[2] = [0.5, 0.25, 0.25]
branchendistproland[3] = [0.55, 0.2, 0.25]
branchendistproland[4] = [0.45, 0.1, 0.45]


def clean_data(my_data):
    with open(my_data, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        row_count = sum(1 for x in reader)

        laenderdistribution = choice(laender, row_count, p=laenderdist)

        pairs = []

        i = 0
        file.seek(0)
        for row in reader:
            pos1 = row[1].index('(')
            pos2 = row[1].index(')')
            row[0] = i+1
            row[1] = row[1][:pos1] + row[1][pos2+1:]
            row[1] = row[1].strip()
            row[2] = laenderdistribution[i]

            branche = choice(branchen, 1, p=branchendistproland[laenderdistribution[i]])
            pair = [laenderdistribution[i], branche[0]]
            pairs.append(pair)
 #          newRow = np.array([row, branchendistribution[i], businessdistribution[i]])
            i = i + 1
        return pairs

def check_distribution(pairs):

    counts = [[0 for x in range(3)] for y in range(5)]

    for pair in pairs:
        counts[pair[0]][pair[1]] = counts[pair[0]][pair[1]] + 1

    for i in range(len(counts)):
        print("Info für %s" % laenderschluessel[i])
        landsumme = sum(counts[i])
        print("Landessumme: ", landsumme, "| Anteil: ", round(landsumme / len(pairs), 3), " | Plan: ", laenderdist[i])
        print("Branchen:")
        for j in range(len(counts[i])):
            print("\t", branchenschluessel[j], ": ", counts[i][j], " | Anteil: ", round(counts[i][j] / landsumme, 3)
                  , " | Plan: ", branchendistproland[i][j])
        print()

pairs = clean_data('C:/Users/David/Downloads/company_data.csv')
check_distribution(pairs)

# save = np.array(pairs)
# np.savetxt('C:/Users/David/Downloads/test.txt', pairs, delimiter=',', fmt="%s")

# Numpy add column to 2-dim array:
# x = np.array([[10,20,30], [40,50,60]])
# y = np.array([[100], [200]])
# print(np.append(x, y, axis=1)) <-- axis=1 !!!!
# puts out: [[ 10  20  30 100], [ 40  50  60 200]]
