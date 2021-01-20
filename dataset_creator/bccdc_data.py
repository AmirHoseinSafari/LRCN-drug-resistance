import csv
import pandas as pd



# isolate = ['A', 'B', 'A']
# print(type(isolate))
# print(type(isolate[0]))
# iso_dict = dict()
# for i in range(0, len(isolate)):
#     if isolate[i] in iso_dict:
#         index_pointer = int(iso_dict[isolate[i]])
#     else:
#         iso_dict.update({isolate[i]: str(152)})
#
# print(iso_dict)
# graph={'A':['B','C'],
#    'B':['C','D']}
#
# print('A' in graph)


def load_gene_positions():
    dt = pd.read_csv('../Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_allGenes.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    start = sum(dt[['Start']].values.tolist(),[])
    stop = sum(dt[['Stop']].values.tolist(), [])
    # start = start[0:3981]
    # stop = stop[0:3981]
    # dt = pd.read_csv('../Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_genes_v3.csv')
    # dt.set_index(dt.columns[0], inplace=True, drop=True)
    #
    # start.extend(sum(dt[['Start']].values.tolist(), []))
    # stop.extend(sum(dt[['Stop']].values.tolist(), []))
    return start, stop


def binary_search(start, stop, low, high, x):
    # Check base case
    if high >= low:

        mid = (high + low) // 2

        # If element is present at the middle itself
        if start[mid] <= x <= stop[mid]:
            return mid

            # If element is smaller than mid, then it can only
        # be present in left subarray
        elif start[mid] > x:
            return binary_search(start, stop, low, mid - 1, x)

            # Else the element can only be present in right subarray
        else:
            return binary_search(start, stop, mid + 1, high, x)

    else:
        # Element is not present in the array
        return -1


tsv_file = open("../Data/bccdc_snippy_179_snps_training_data.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

isolate = []
position = []

for row in read_tsv:
    isolate.append(row[0])
    position.append(row[4])

isolate = isolate[1:]
position = position[1:]

for i in range (0, len(position)):
    position[i] = int(position[i])

start, stop = load_gene_positions()

zeros = []

res = []

for i in range(0, len(start)):
    zeros.append(0)


iso_dict = dict()
for i in range(0, len(isolate)):
    if i % 100 == 0:
        print(i)

    index_pointer = -1

    if isolate[i] in iso_dict:
        index_pointer = int(iso_dict[isolate[i]])
    else:
        tmp = [isolate[i]]
        tmp.extend(zeros)
        res.append(tmp)
        iso_dict.update({isolate[i]: str(len(res) - 1)})
        index_pointer = len(res) - 1

    # if len(res) == 0:
    #     tmp = [isolate[i]]
    #     tmp.extend(zeros)
    #     res.append(tmp)
    #     index_pointer = 0
    # else:
    #     for j in range(0, len(res)):
    #         if res[j][0] == isolate[i]:
    #             index_pointer = j
    #             break
    #     if index_pointer == -1:
    #         tmp = [isolate[i]]
    #         tmp.extend(zeros)
    #         res.append(tmp)
    #         index_pointer = len(res) - 1

    gene_index = binary_search(start, stop, 0, len(start) - 1, position[i])

    if gene_index != -1:
        res[index_pointer][gene_index + 1] = res[index_pointer][gene_index + 1] + 1

    # found_gene = -1
    # for j in range(0, len(start)):
    #     passed = 0
    #     if start[j] <= position[i] <= stop[j]:
    #         # print("1111111111111")
    #         # print(j)
    #         # print(len(res[index_pointer]))
    #         res[index_pointer][j+1] = res[index_pointer][j+1] + 1
    #         break
    #     if start[j] > position[i]:
    #         passed = 1
    #     if passed == 1 and position[i] > stop[j]:
    #         break SRR6153157

f = open('bccdc_data.csv', 'w')
for item in res:
    for i in range(len(item)):
        if i == 0:
            f.write(str(item[i]))
        else:
            f.write(',' + str(item[i]))
    f.write('\n')
f.close()

#res : 1 sus : 0
tsv_file = open("../Data/bccdc_phenotype.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")

isolate = []
drug = []
status = []

for row in read_tsv:
    isolate.append(row[0])
    drug.append(row[1])
    status.append(row[2])


isolate = isolate[1:]
drug = drug[1:]
status = status[1:]

zeros_d = []
labels = []

for i in range(0, 5):
    zeros_d.append(-1)

for i in range(0, len(res)):
    tmp = []
    for i in range(0, 5):
        tmp.append(-1)
    # tmp = [res[i][0]]
    labels.append(tmp)

print(labels)
for i in range(0, len(isolate)):
    # print(labels[0:10])
    if isolate[i] in iso_dict:
        index_pointer = int(iso_dict[isolate[i]])
    else:
        # print("error")
        continue
    if drug[i] == "PYRAZINAMIDE":
        if status[i] == "RESISTANT":
            # print(status[i])
            # print(i)
            # print(index_pointer)
            # print("ASdadq")
            # print(labels[index_pointer])
            labels[index_pointer][2] = 1
            # print(labels[index_pointer])
        elif status[i] == "SUSCEPTIBLE":
            # print(status[i])
            # print(i)
            # print(index_pointer)
            labels[index_pointer][2] = 0
        else:
            print("wtf")
    elif drug[i] == "ISONIAZID":
        if status[i] == "RESISTANT":
            # print("ASdadq")
            labels[index_pointer][3] = 1
        elif status[i] == "SUSCEPTIBLE":
            labels[index_pointer][3] = 0
        else:
            print("wtf")
    elif drug[i] == "STREPTOMYCIN":
        if status[i] == "RESISTANT":
            # print("ASdadq")
            labels[index_pointer][0] = 1
        elif status[i] == "SUSCEPTIBLE":
            labels[index_pointer][0] = 0
        else:
            print("wtf")
    elif drug[i] == "RIFAMPICIN":
        if status[i] == "RESISTANT":
            # print("ASdadq")
            labels[index_pointer][1] = 1
        elif status[i] == "SUSCEPTIBLE":
            labels[index_pointer][1] = 0
        else:
            print("wtf")
    elif drug[i] == "ETHAMBUTOL":
        if status[i] == "RESISTANT":
            # print("sdfewf e")
            # print(labels[index_pointer])
            labels[index_pointer][4] = 1
            # print(labels[index_pointer])
        elif status[i] == "SUSCEPTIBLE":
            # print("sdfewf e")
            # print(labels[index_pointer])
            labels[index_pointer][4] = 0
            # print(labels[index_pointer])
        else:
            print("wtf")
    else:
        print("errorrrrrrr")
        continue

f = open('bccdc_data_label.csv', 'w')

for i in range(0, len(labels)):
    for j in range(0, len(labels[i])):
        if labels[i][j] != 0:
            print(labels[i][j] )

for item in labels:
    # print(item)
    for i in range(len(item)):
        if i == 0:
            f.write(str(item[i]))
        else:
            f.write(',' + str(item[i]))
    f.write('\n')
f.close()
