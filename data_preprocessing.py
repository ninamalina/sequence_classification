import random
from itertools import izip

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def represent_as_kmers(k, s, keywords):
    d = [0] * len(keywords)
    for i in range(0, len(s)-k):
        if s[i:i+k] not in keywords:
            return []
        d[keywords.index(s[i:i+k])] += 1
    return d


def compute_stats(data_stats):
    taxa_index_dict = {}

    i = 0
    for p in data_stats:
        if "p_" + p not in taxa_index_dict:
            taxa_index_dict["p_" + p] = i
            i += 1

        n_p = 0

        for c in data_stats[p]:
            if "c_" + c not in taxa_index_dict:
                taxa_index_dict["c_" + c] = i
                i += 1
            n_c = 0

            for o in data_stats[p][c]:
                if "o_" + o not in taxa_index_dict:
                    taxa_index_dict["o_" + o] = i
                    i += 1
                n_o = 0

                for f in data_stats[p][c][o]:
                    if "f_" + f not in taxa_index_dict:
                        taxa_index_dict["f_" + f] = i
                        i += 1
                    n_f = 0

                    for g in data_stats[p][c][o][f]:
                        if "g_" + g not in taxa_index_dict:
                            taxa_index_dict["g_" + g] = i
                            i += 1
                        n_g = data_stats[p][c][o][f][g]

                        print "\t\t\t\t", g, n_g
                        n_f += n_g

                    print "\t\t\t", f, n_f
                    n_o += n_f

                print "\t\t", o, n_o
                n_c += n_o

            print "\t", c, n_c
            n_p += n_c

        # print p, n_p

    return taxa_index_dict


def prepare_data(f_in, f_kmers, f_deep_learning, keywords):
    f_kmers.write("phylum,class,order,family,genus")
    for keyword in keywords:
        f_kmers.write(",")
        f_kmers.write(keyword)
    f_kmers.write("\n")
    f_deep_learning.write("phylum,class,order,family,genus,sequence\n")

    i = 0
    for line in f_in:
        if line.startswith(">"):
            if i > 0 and len(sequence) > 1200:
                k_mers = represent_as_kmers(5, sequence, keywords)
                if k_mers != []:
                    f_kmers.write(s_phylum + "," + s_class + "," + s_order + "," + s_family + "," + s_genus + ",")
                    f_deep_learning.write(s_phylum + "," + s_class + "," + s_order + "," + s_family + "," + s_genus + ",")
                    f_kmers.write(",".join(map(str, k_mers)) + "\n")
                    f_deep_learning.write(sequence + "\n")

            i += 1
            sequence = ""

            l = line.strip().split("domain")[1].split(";")[1:]

            if not ("class" in l and "order" in l and "family" in l and "genus" in l):
                continue

            s_phylum = l[0].split('"')[len(l[0].split('"')) - 2]
            s_class = l[l.index("class") - 1].split('"')[len(l[l.index("class") - 1].split('"')) - 2]
            s_order = l[l.index("order") - 1].split('"')[len(l[l.index("order") - 1].split('"')) - 2]
            s_family = l[l.index("family") - 1].split('"')[len(l[l.index("family") - 1].split('"')) - 2]
            s_genus = l[l.index("genus") - 1].split('"')[len(l[l.index("genus") - 1].split('"')) - 2]

        else:
            sequence += line.strip()


def count_taxas(f_in):
    data_stats = AutoVivification()
    f_in.next()
    count = 0
    for line in f_in:
        count += 1
        l = line.split(",")[0:5]

        s_phylum = l[0]
        s_class = l[1]
        s_order = l[2]
        s_family = l[3]
        s_genus = l[4]

        if data_stats[s_phylum][s_class][s_order][s_family][s_genus] == {}:
            data_stats[s_phylum][s_class][s_order][s_family][s_genus] = 1
        else:
            data_stats[s_phylum][s_class][s_order][s_family][s_genus] += 1

    return count, data_stats


def reduce_data(f_in_kmers, f_in_sequences, f_out_kmers, f_out_sequences, p, n):
    first_line_kmers = f_in_kmers.next()
    f_out_kmers.write(first_line_kmers)

    first_line_sequences = f_in_sequences.next()
    f_out_sequences.write(first_line_sequences)

    random.seed = 42

    counts = {"Actinobacteria":0, "Firmicutes":0, "Proteobacteria":0}
    for line1, line2 in izip(f_in_kmers, f_in_sequences):
        if line1.split(",")[0].strip() in ["Actinobacteria", "Firmicutes", "Proteobacteria"]:
            if random.random() < p*2 and counts[line1.split(",")[0].strip()] < n:
                f_out_kmers.write(line1)
                f_out_sequences.write(line2)
                counts[line1.split(",")[0].strip()] += 1

    print counts


