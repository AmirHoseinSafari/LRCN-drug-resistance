import pandas as pd
from loading_data import data_loader


def load_gene_positions():
    dt = pd.read_csv('Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_allGenes.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    start = sum(dt[['Start']].values.tolist(),[])
    stop = sum(dt[['Stop']].values.tolist(), [])
    start = start[0:3981]
    stop = stop[0:3981]
    dt = pd.read_csv('Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_genes_v3.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    start.extend(sum(dt[['Start']].values.tolist(), []))
    stop.extend(sum(dt[['Stop']].values.tolist(), []))
    return start, stop


def load_snps():
    dt = pd.read_csv('Data/sparse_matrix/rows_complete.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    snp_positions = []
    for col in dt.columns:
        col = ''.join(i for i in col if i.isdigit())
        snp_positions.append(col)

    indexNamesArr = dt.index.values
    listOfRowIndexLabels = list(indexNamesArr)
    return snp_positions


def table_creator(start, stop, snps):
    snp_index = 0

    #TODO
    df_train = data_loader.process(38)

    isolates = df_train.index.values
    isolates = list(isolates)

    snp_dataset = df_train.values.tolist()

    print(len(start))
    result = []
    for i in range(0, len(isolates)):
        result.append([isolates[i]])

    debug = 0
    state = 0
    for i in range(0, len(start)):
        for j in range(snp_index, len(snps)):
            # print(str(i) + "___" + str(j))
            # curr_start = start[i]
            # curr_stop = stop[i]
            # curr_snp = snps[j]
            if state == 0:
                if start[i] <= int(snps[j]) and stop[i] > int(snps[j]):
                    snp_index = j
                    state = 1
                elif stop[i] < int(snps[j]):
                    break
            if state == 1:
                if stop[i] < int(snps[j]):
                    debug = debug + 1
                    for k in range(0, len(result)):
                        sum1 = 0
                        if snp_index == j:
                            try:
                                sum1 = snp_dataset[k][j]
                            except:
                                print("errorrrrrrrrrrrrrrrrrrr")
                                continue
                        else:
                            for l in range(snp_index, j):
                                try:
                                    sum1 += snp_dataset[k][l]
                                except:
                                    print("errorrrrrrrrrrrrrrrrrrr")
                                    continue
                        result[k].append(sum1)
                    snp_index = j
                    state = 0
                    break
    print(debug)
    f = open('gene_data.csv', 'w')
    for item in result:
        for i in range(len(item)):
            if i == 0:
                f.write(str(item[i]))
            else:
                f.write(',' + str(item[i]))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    start, stop = load_gene_positions()
    snps = load_snps()
    # table_creator(start, stop, snps)

    ## debug usage
    count = 0
    index = 0
    miss = 0
    arr = []
    arr.append(-1)
    print(len(snps))
    print(len(start))
    for i in range(0, len(snps)):
        a = int(snps[i])
        a1 = start[index]
        a2 = stop[index]
        if int(snps[i]) >= start[index] and  int(snps[i]) < stop[index]:
            if arr[len(arr) - 1] != index:
                arr.append(index)
            count = count + 1
        elif int(snps[i]) < start[index]:
            miss = miss + 1
            continue
        elif int(snps[i]) > stop[index]:
            index = index + 1
            i = i - 1
    print(count)
    print(arr)
    print(len(arr))
    print(miss)