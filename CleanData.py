import csv
import numpy as np
from numpy.random import choice, randint


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

branchendistproland = [[0 for x in range(len(branchen))] for y in range(len(laender))]
branchendistproland[0] = [0.6, 0.2, 0.2]
branchendistproland[1] = [0.15, 0.15, 0.7]
branchendistproland[2] = [0.5, 0.25, 0.25]
branchendistproland[3] = [0.55, 0.2, 0.25]
branchendistproland[4] = [0.45, 0.1, 0.45]

mitarbeiterrangeprobranche = [[0 for x in range(2)] for y in range(len(branchen))]
mitarbeiterrangeprobranche[0] = [1000, 20000]
mitarbeiterrangeprobranche[1] = [100, 2000]
mitarbeiterrangeprobranche[2] = [250, 5000]


def clean_data(my_data):
    with open(my_data, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        row_count = sum(1 for x in reader)

        laenderdistribution = choice(laender, row_count, p=laenderdist)

        dataset = []

        i = 0
        file.seek(0)
        for row in reader:
            # URL aus dem Namen schneiden
            # pos1 = row[1].index('(')
            # pos2 = row[1].index(')')
            # row[1] = row[1][:pos1] + row[1][pos2+1:]
            # row[1] = row[1].strip()

            branche = choice(branchen, 1, p=branchendistproland[laenderdistribution[i]])
            mitarbeiteranzahl = randint(mitarbeiterrangeprobranche[branche[0]][0]
                                        , high=mitarbeiterrangeprobranche[branche[0]][1]+1)

            if mitarbeiteranzahl == 0:
                print(mitarbeiterrangeprobranche[branche[0]][0])
                print(mitarbeiterrangeprobranche[branche[0]][1]+1)

            row[0] = i + 1
            row[1] = laenderdistribution[i]
            row[2] = branche[0]
            row[3] = mitarbeiteranzahl

            growth = float(row[5].replace("%", ""))
            if(growth >= 1000.0):
                growth = round(growth / 100, 2)
            if (growth >= 100.0):
                growth = round(growth / 10, 2)
            row[5] = growth

            i = i + 1

            dataset.append(row)
#        save = np.array(dataset)
        np.savetxt('C:/Users/David/Downloads/dataset.txt', dataset, delimiter=',', fmt="%s")


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

clean_data('C:/Users/David/Downloads/company_data.csv')

# Numpy add column to 2-dim array:
# x = np.array([[10,20,30], [40,50,60]])
# y = np.array([[100], [200]])
# print(np.append(x, y, axis=1)) <-- axis=1 !!!!
# puts out: [[ 10  20  30 100], [ 40  50  60 200]]
