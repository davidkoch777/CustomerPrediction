import csv
import numpy as np
from numpy.random import choice


laender = ["Deutschland", "Frankreich", "Großbritanien", "USA", "China"]
branchen = ["Lebensmittel", "Industrie", "Bauwesen", "Energie"]
businesses = ["B2B", "B2C"]

branchendistproland = {}
branchendistproland[laender[0]] = [0.1, 0.6, 0.2, 0.1]
branchendistproland[laender[1]] = [0.1, 0.4, 0.4, 0.1]
branchendistproland[laender[2]] = [0.3, 0.3, 0.2, 0.2]
branchendistproland[laender[3]] = [0.3, 0.25, 0.25, 0.2]
branchendistproland[laender[4]] = [0.0, 0.1, 0.5, 0.4]

businessdistprobranche = {}
businessdistprobranche[branchen[0]] = [0.0, 1.0]
businessdistprobranche[branchen[1]] = [0.4, 0.6]
businessdistprobranche[branchen[2]] = [0.35, 0.65]
businessdistprobranche[branchen[3]] = [0.3, 0.7]

def cleanData(my_data):
    with open(my_data, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        row_count = sum(1 for x in reader)

        laenderdistribution = choice(laender, row_count, p=[0.4, 0.07, 0.03, 0.2, 0.3])

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

            branche = choice(branchen, 1, branchendistproland[laenderdistribution[i]])
            business = choice(businesses, 1, businessdistprobranche[branche[0]])
            pair = [branche[0], business[0]]
            pairs.append(pair)
 #          newRow = np.array([row, branchendistribution[i], businessdistribution[i]])
            i = i + 1

        print(pairs)
        save = np.array(pairs)
        np.savetxt('C:/Users/David/Downloads/test.txt', pairs, delimiter=',', fmt="%s")


cleanData('C:/Users/David/Downloads/company_data.csv')

# Numpy add column to 2-dim array:
# x = np.array([[10,20,30], [40,50,60]])
# y = np.array([[100], [200]])
# print(np.append(x, y, axis=1)) <-- axis=1 !!!!
# puts out: [[ 10  20  30 100], [ 40  50  60 200]]
