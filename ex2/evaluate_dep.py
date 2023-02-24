# Imports
import pandas
import spacy
import string
import numpy

spanish_dep_rel = { 
                    # "ROOT",
                    "adpcomp":"acl",
                    # "advcl",
                    "neg":"advmod",
                    # "amod",
                    # "appos",
                    # "aux",
                    # "case",
                    # "cc",
                    # "ccomp",
                    # "compound",
                    # "conj",
                    # "cop",
                    # "csubj",
                    # "dep",
                    "poss":"det",
                    # "expl:impers",
                    # "expl:pass",
                    # "expl:pv",
                    # "fixed",
                    # "flat",
                    "iobj":"obj",
                    "dobj":"obj",
                    # "mark",
                    # "nmod",
                    # "nsubj",
                    # "nummod",
                    # "obl",
                    # "parataxis",
                    "p":"punct",
                    # "xcomp"
                    }

# italian_dep_rel = ["ROOT", "acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "aux:pass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "dep", "det", "det:poss", "det:predet", "discourse", "expl", "expl:impers", "expl:pass", "fixed", "flat", "flat:foreign", "flat:name", "iobj", "mark", "nmod", "nsubj", "nsubj:pass", "nummod", "obj", "obl", "obl:agent", "parataxis", "punct", "vocative", "xcomp"]


#
def calculate_average_accuracy(list):
    return numpy.mean(list)

# Calculate the accuracy between two lists of elements
def calculate_accuracy(list1, list2):
    if len(list1)==(len(list2)):
        list1 = numpy.array(list1)
        list2 = numpy.array(list2)
        correct = list1==list2
        accuracy = correct.sum()/correct.size
    else:
        accuracy = 0.5
    return accuracy

# Calculate the accuracy of each sentence's dependency relations and dependency heads, and also return % of perfect sentences in dataframe
def evaluate_dependencies(dataframe):
    perfect_sentences = 0
    rel_acc_column = []
    head_acc_column = []
    for index, row in dataframe.iterrows():
        if row["Relation_gold"] == row["Relation_prediction"] and row["Head_gold"] == row["Head_prediction"]:
            perfect_sentences += 1
        rel_acc_column.append(calculate_accuracy(row["Relation_gold"], row["Relation_prediction"]))
        head_acc_column.append(calculate_accuracy(row["Head_gold"], row["Head_prediction"]))
    perfect_sentences = perfect_sentences*100/len(dataframe["Sentence"])
    average_rel_acc = calculate_average_accuracy(rel_acc_column)
    average_head_acc = calculate_average_accuracy(head_acc_column)
    return rel_acc_column, head_acc_column, average_rel_acc, average_head_acc, perfect_sentences

# Use spacy to predict the list of dependency relations and heads for each sentence
def predict_dependencies(dataframe, language):
    if language == "es":
        nlp = spacy.load("es_core_news_sm")
    elif language == "it":
        nlp = spacy.load("it_core_news_sm")
    dep_rel_column = []
    dep_head_column = []
    for sentence in dataframe["Sentence"]:
        dep_rel_list = []
        dep_head_list = []
        doc = nlp(sentence)
        for token in doc:
            if token.dep_=="case":
                dep_rel_list.append("adpmod")
            elif "expl:" in token.dep_:
                dep_rel_list.append("prt")
            elif token.dep_=="flat" or token.dep_=="fixed" or token.dep_=="obl":
                dep_rel_list.append("adpobj")
            else:
                dep_rel_list.append(token.dep_)
            dep_head_list.append(token.head.text)
        dep_rel_column.append(dep_rel_list)
        dep_head_column.append(dep_head_list)
    return dep_rel_column, dep_head_column

# Read conll file and save word form, dep_relation and dep_id into gold label dataframe
def read_gold_data(file):
    gold_dataframe = pandas.DataFrame()
    sentence_column = []
    dep_head_column = []
    dep_rel_column = []
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        sentence = ""
        word_form_list = []
        dep_id_list = []
        dep_rel_list = []
        for line in lines:
            if len(line)>1:
                elements = line.split("\t")
                word = elements[1]
                if word in string.punctuation:
                    sentence = sentence[:-1]
                sentence += word + " "
                word_form_list.append(word)
                dep_rel = elements[-3]
                if dep_rel in spanish_dep_rel.keys():
                    dep_rel_list.append(spanish_dep_rel[dep_rel])
                else:
                    dep_rel_list.append(dep_rel)
                dep_id_list.append(elements[6])
            else:
                sentence_column.append(sentence)
                dep_rel_column.append(dep_rel_list)
                dep_head_list = []
                for id in dep_id_list:
                    dep_head_list.append(word_form_list[int(id)-1])
                dep_head_column.append(dep_head_list)
                sentence = ""
                word_form_list = []
                dep_id_list=[]
                dep_rel_list=[]
        sentence_column.append(sentence)
        dep_rel_column.append(dep_rel_list)
        dep_head_column.append(dep_head_list)
        gold_dataframe["Sentence"]=sentence_column
        gold_dataframe["Relation_gold"]=dep_rel_column
        gold_dataframe["Head_gold"]=dep_head_column
        return gold_dataframe

# Main
def main():
    # Declare input file paths
    es_file = "./data/es/es-universal-test.conll"
    it_file = "./data/it/it-universal-test.conll"

    # Build the gold label dataframes
    es_gold_dataframe = read_gold_data(es_file)
    it_gold_dataframe = read_gold_data(it_file)

    # Declare output file paths
    es_gold_output = "./data/gold/es-dataframe-gold.tsv"
    it_gold_output = "./data/gold/it-dataframe-gold.tsv"

    # Write the gold label dataframes onto tsv file
    es_gold_dataframe.to_csv(es_gold_output, sep="\t", encoding="utf-8")
    it_gold_dataframe.to_csv(it_gold_output, sep="\t", encoding="utf-8")
    
    # Use spacy to predict the list of dependency relations and heads
    es_gold_dataframe["Relation_prediction"], es_gold_dataframe["Head_prediction"] = predict_dependencies(es_gold_dataframe, "es")
    it_gold_dataframe["Relation_prediction"], it_gold_dataframe["Head_prediction"] = predict_dependencies(it_gold_dataframe, "it")

    #
    es_gold_dataframe["Relation_accuracy"], es_gold_dataframe["Head_accuracy"], es_average_dep, es_average_head, es_perfect = evaluate_dependencies(es_gold_dataframe)
    it_gold_dataframe["Relation_accuracy"], it_gold_dataframe["Head_accuracy"], it_average_dep, it_average_head, it_perfect = evaluate_dependencies(it_gold_dataframe)

    #
    print("\nSpanish results:\n")
    print("Average dependency relation accuracy:", es_average_dep)
    print("Average dependency head accuracy:", es_average_head)
    # print("% Perfect:", es_perfect)

    print("\nItalian results:\n")
    print("Average dependency relation accuracy:", it_average_dep)
    print("Average dependency head accuracy:", it_average_head)
    # print("% Perfect:", it_perfect)
    print()

    # Declare output file paths
    es_pred_output = "./results/es-dataframe-pred.tsv"
    it_pred_output = "./results/it-dataframe-pred.tsv"

    # Write the prediction dataframes onto tsv file
    es_gold_dataframe.to_csv(es_pred_output, sep="\t", encoding="utf-8")
    it_gold_dataframe.to_csv(it_pred_output, sep="\t", encoding="utf-8")

# Main
main()