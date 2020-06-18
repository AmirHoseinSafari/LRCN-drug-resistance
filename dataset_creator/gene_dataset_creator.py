import pandas as pd
import data_loader

def load_gene_positions():
    dt = pd.read_csv('../Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_allGenes.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    start = sum(dt[['Start']].values.tolist(),[])
    stop = sum(dt[['Stop']].values.tolist(), [])
    start = start[0:3981]
    stop = stop[0:3981]
    dt = pd.read_csv('../Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_genes_v3.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    start.extend(sum(dt[['Start']].values.tolist(), []))
    stop.extend(sum(dt[['Stop']].values.tolist(), []))
    return start, stop


def load_snps():
    dt = pd.read_csv('../Data/sparse_matrix/rows.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    snp_positions = []
    for col in dt.columns:
        col = ''.join(i for i in col if i.isdigit())
        snp_positions.append(col)

    indexNamesArr = dt.index.values
    listOfRowIndexLabels = list(indexNamesArr)
    return snp_positions, listOfRowIndexLabels


def table_creator(start, stop, snps, isolates):
    snp_index = 0

    df_train = data_loader.process(2)

    arr = df_train.values.tolist()


    result = []
    for i in range(0, len(isolates)):
        result.append([isolates[i]])

    state = 0
    for i in range(0, len(start)):
        for j in range(snp_index, len(snps)):
            if state == 0:
                if start[i] <= int(snps[j]):
                    snp_index = j
                    state = 1
            elif state == 1:
                if stop[i] < int(snps[j]):
                    for k in range(0, len(result)):
                        sum1 = 0
                        for l in range(snp_index, j):
                            print(k)
                            print(l)
                            print(len(arr))
                            print(len(arr[k]))
                            sum1 += arr[k][l]
                        result[k].append(sum1)
                    snp_index = j
                    state = 0
                    break

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
    snps, isolates = load_snps()
    table_creator(start, stop, snps, isolates)