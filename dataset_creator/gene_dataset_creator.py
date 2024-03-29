import pandas as pd
from loading_data import data_loader


def load_gene_positions():
    dt = pd.read_csv('Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_allGenes.csv')
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


def load_imp_gene_positions():
    import csv

    file_name = '../Data/important_genes2.csv'
    data = list(csv.reader(open(file_name)))

    print(data)

    start = []
    stop = []

    for i in range(1, len(data)):
        start.append(data[i][2])
        stop.append(data[i][3])

    return start, stop


def find_index_imp_genes(start, stop, imp_start, imp_stop):
    index = []
    for i in range(0, len(imp_start)):
        for j in range(0, len(start)):
            if start[j] == int(imp_start[i]):
                if stop[j] == int(imp_stop[i]):
                    index.append(j)
                    break
                else:
                    print("error")
            if start[j] > int(imp_start[i]):
                print(imp_start[i])
                print("intergenic")
                break
    print(len(index))
    print(index)
    # index = [4, 5, 17, 52, 123, 206, 207, 294, 363, 364, 429, 510, 514, 598, 678, 679, 680, 689, 690, 691, 715, 716, 724, 725, 726, 730, 752, 1080, 1081, 1256, 1259, 1276, 1296, 1350, 1363, 1405, 1411, 1417, 1419, 1545, 1596, 1597, 1736, 1751, 1789, 1817, 1827, 1911, 1912, 1985, 2041, 2042, 2116, 2126, 2186, 2213, 2276, 2402, 2587, 2600, 2620, 2697, 2716, 2863, 2923, 2947, 2956, 2957, 2973, 2976, 3108, 3129, 3130, 3131, 3132, 3138, 3183, 3216, 3253, 3295, 3384, 3416, 3419, 3486, 3487, 3489, 3491, 3656, 3690, 3782, 3801, 3834, 3840, 3847, 4010, 4051, 4052, 4053, 4054, 4066, 4116, 4117, 4181]


def load_snps_positions():
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

    # TODO - You should change here - This suppose to load your SNP dataset into df_train
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
    indexes = []
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
                    indexes.append(i)
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
    print(len(indexes))
    print(indexes)
    f = open('gene_data.csv', 'w')
    for item in result:
        for i in range(len(item)):
            if i == 0:
                f.write(str(item[i]))
            else:
                f.write(',' + str(item[i]))
        f.write('\n')
    f.close()


def main():
    # imp_start, imp_stop = load_imp_gene_positions()
    start, stop = load_gene_positions()
    # find_index_imp_genes(start, stop, imp_start, imp_stop)
    snps = load_snps_positions()
    table_creator(start, stop, snps)


if __name__ == '__main__':
    # imp_start, imp_stop = load_imp_gene_positions()
    start, stop = load_gene_positions()
    # find_index_imp_genes(start, stop, imp_start, imp_stop)
    snps = load_snps_positions()
    table_creator(start, stop, snps)

    ## debug usage
    # count = 0
    # index = 0
    # miss = 0
    # arr = []
    # arr.append(-1)
    # print(len(snps))
    # print(len(start))
    # for i in range(0, len(snps)):
    #     a = int(snps[i])
    #     a1 = start[index]
    #     a2 = stop[index]
    #     if int(snps[i]) >= start[index] and  int(snps[i]) < stop[index]:
    #         if arr[len(arr) - 1] != index:
    #             arr.append(index)
    #         count = count + 1
    #     elif int(snps[i]) < start[index]:
    #         miss = miss + 1
    #         continue
    #     elif int(snps[i]) > stop[index]:
    #         index = index + 1
    #         if index == len(start):
    #             break
    #         i = i - 1
    # print(count)
    # print(arr)
    # print(len(arr))
    # print(miss)