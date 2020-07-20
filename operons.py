import pandas as pd


def load_gene_positions():
    dt = pd.read_csv('Data/EPFL_Data/Mycobacterium_tuberculosis_H37Rv_allGenes.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)

    names = sum(dt[['Name']].values.tolist(), [])
    return names


def find_operons_index(names):
    cnt = 0
    cnt_unknown = 0
    index = []
    names_op =[]
    marker = []
    for i in range(0, len(names)):
        marker.append(0)

    for i in range(0, len(names)):
        if names[i].startswith('Rv') or names[i].startswith('ncRv'):
            cnt_unknown = cnt_unknown + 1
            continue
        else:
            if marker[i] == 1:
                cnt = cnt+1
                continue
            list = []
            list_names = []
            list.append(i)
            list_names.append(names[i])
            for j in range(i + 1, len(names)):
                if names[i][0:3] == names[j][0:3]:
                    marker[j] = 1
                    list.append(j)
                    list_names.append(names[j])
            if len(list) > 1:
                index.append(list)
                names_op.append(list_names)
                print(list_names)
                print(list)
            # print(names[i])
            cnt = cnt + 1
    print(cnt)
    print(cnt_unknown)
    # print(names_op)


if __name__ == '__main__':
    names = load_gene_positions()
    print(names)
    find_operons_index(names)