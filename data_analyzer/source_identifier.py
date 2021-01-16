f = open("../Data/source/bccdc_phenotype.txt", "r")

res = open("../Data/source/result.txt","a+")

dup = []

def idReader(content, start):
    ids = []
    dup = []
    while content != "":
        for i in range(0, len(content)):
            if content[i:i+3] == start:
                id = ""
                for j in range((i), len(content)):
                    if content[j] != "\t":
                        id += content[j]
                    else:
                        ids.append(id)
                        break
                break
        content = f.readline()

    print(len(ids))
    print(ids)
    counter = 0
    for i in range(0, len(ids)):
        duplicate = False
        for j in range(0, len(dup)):
            if ids[i] == dup[j]:
                duplicate = True
                break
        if duplicate == False:
            counter += 1
            dup.append(ids[i])
            res.write(ids[i])
            res.write("\n")
    print(dup)
    print(counter)
    print("_______")


content = f.readline()

idReader(content, "SRR")

f = open("../Data/source/patric_phenotype.txt", "r")

content = f.readline()

idReader(content, "SRR")

f = open("../Data/source/reseq_phenotype_fastq.txt", "r")

content = f.readline()

idReader(content, "ERR")



