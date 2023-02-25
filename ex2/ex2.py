# Imports
import pandas
import spacy
import numpy

# Calculate the average of a list
def calculate_average_accuracy(list):
    return numpy.mean(list)

# Calculate the accuracy between two lists of elements. If they do not have the same length, we append dummy elements on to the list until they are the same length so that we can compare them, even though the appended dummies will never match
def calculate_accuracy(list1, list2):
    # If the lists do not match in length, add dummy elements to the shortest list until it matches
    if len(list1)>len(list2):
        for i in range(len(list1)-len(list2)):
            list2.append("-")
    elif len(list2)>len(list1):
        for i in range(len(list2)-len(list1)):
            list1.append("-")

    # Now that the lists match in length, calculate the accuracy between them and return it
    list1 = numpy.array(list1)
    list2 = numpy.array(list2)
    correct = list1==list2
    accuracy = correct.sum()/correct.size
    return accuracy

# Calculate the accuracy of each sentence's dependency relations and dependency heads, and also return % of perfect sentences in dataframe and average accuracies of all sentences
def evaluate_dependencies(dataframe):
    # Create counter for perfect sentences and empty sentence accuracy columns
    perfect_sentences = 0
    rel_acc_column = []
    head_acc_column = []

    # Iterate through each sentence and add 1 to perfect if every list matches
    for index, row in dataframe.iterrows():
        if row["Relation_gold"] == row["Relation_prediction"] and row["Head_gold"] == row["Head_prediction"]:
            perfect_sentences += 1
        
        # Calculate the accuracy of the lists of dependency relations and dependency heads and append to accuracies columns
        relation_accuracy = calculate_accuracy(row["Relation_gold"], row["Relation_prediction"])
        rel_acc_column.append(relation_accuracy)
        head_accuracy = calculate_accuracy(row["Head_gold"], row["Head_prediction"])
        head_acc_column.append(head_accuracy)
    
    # Convert absolute frequency of perfect sentences into relative % frequency
    perfect_sentences = perfect_sentences*100/len(dataframe["Sentence"])

    # Calculate the averages of sentence accuracy columns for both relations and heads
    average_rel_acc = calculate_average_accuracy(rel_acc_column)
    average_head_acc = calculate_average_accuracy(head_acc_column)

    # Return everything: the two new accuracy columns, the averages and the % perfect sentences
    return rel_acc_column, head_acc_column, average_rel_acc, average_head_acc, perfect_sentences

# Use spacy to predict the list of dependency relations and heads for each sentence. We chose to use the large model because it outputted better results than the small model
def predict_dependencies(dataframe, language):
    # Load either Spanish or Italian spacy model. We chose the large models after testing the small ones and getting slightly worse results
    if language == "es":
        nlp = spacy.load("es_core_news_lg")
    elif language == "it":
        nlp = spacy.load("it_core_news_lg")

    # Create empty columns for the predictions
    dep_rel_column = []
    dep_head_column = []

    # Iterate through every sentence and process it through spacy
    for sentence in dataframe["Sentence"]:
        dep_rel_list = []
        dep_head_list = []
        doc = nlp(sentence)

        # Append each token's dep_ and head.text to the respective lists of dependency relations and dependency heads to compare later on
        for token in doc:
            dep_rel_list.append(token.dep_)
            dep_head_list.append(token.head.text)

        # Append the sentence level lists to the predictions columns
        dep_rel_column.append(dep_rel_list)
        dep_head_column.append(dep_head_list)

    # Return the two new columns of predictions
    return dep_rel_column, dep_head_column

# Read conll file and save word form, dep_relation and dep_id into gold label dataframe
def read_gold_data(file):
    # Create empty dataframe and columns
    gold_dataframe = pandas.DataFrame()
    sentence_column = []
    dep_head_column = []
    dep_rel_column = []

    # Read gold label files per line
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # Create empty lists of word forms, dependency relations and dependency ids
        word_form_list = []
        dep_id_list = []
        dep_rel_list = []

        # Read each line of the conll. If it contains "# text =" we keep the sentence string
        for line in lines:
            if "# text =" in line:
                sentence = line.split("text = ")[1][:-1]

            # If the length is > 1 and it does not start with # it means we found an annotated token of the sentence
            elif len(line)>1 and line[0]!="#":

                # We split by tabs to get each label
                elements = line.split("\t")

                # If the word id contains "-" or "." then it's a pair such as "de" + "el" in Spanish and not a real annotated token, we skip them
                if "-" not in elements[0] and "." not in elements[0]:

                    # We keep the word forms, dep relations and dep ids each in a list
                    word = elements[1]
                    word_form_list.append(word)
                    dep_rel = elements[7]
                    dep_id = elements[6]

                    # In the gold label root is lowercased, we uppercase it for it to match spacy labels
                    if dep_rel == "root":
                        dep_rel = "ROOT"
                    dep_rel_list.append(dep_rel)

                    # If it is the root its dependency id is 0, but the head is gonna be itself, so we keep its own id instead
                    if dep_rel=="ROOT":
                        dep_id = elements[0]
                    dep_id_list.append(dep_id)

            # If we find a line of length 1 it is an empty line (with a "\n"), which means that we finished reading one sentence, so we append every list to the columns
            elif len(line)==1:
                sentence_column.append(sentence)
                dep_rel_column.append(dep_rel_list)

                # We translate here the dependency ids to their respective head, as we are going to compare lists of the .head spacy attribute to know which are the syntactic parents of each element
                dep_head_list = []
                for id in dep_id_list:
                    dep_head_list.append(word_form_list[int(id)-1])
                dep_head_column.append(dep_head_list)

                # We empty the lists as we are gonna start a new sentence in the following iteration
                word_form_list = []
                dep_id_list = []
                dep_rel_list = []
        
        # Once we finish all the lines of the data the info of the last sentence will still be hanging, so it needs to be appended now
        sentence_column.append(sentence)
        dep_rel_column.append(dep_rel_list)
        dep_head_column.append(dep_head_list)

        # We add all the gold labels columns to the dataframe
        gold_dataframe["Sentence"]=sentence_column
        gold_dataframe["Relation_gold"]=dep_rel_column
        gold_dataframe["Head_gold"]=dep_head_column
        
        # Return the dataframe
        return gold_dataframe

# Main
def main():
    # Declare input file paths
    es_file = "./data/es/es_ancora-ud-test.conllu"
    it_file = "./data/it/it_isdt-ud-test.conllu"

    # Build the gold label dataframes
    es_gold_dataframe = read_gold_data(es_file)
    it_gold_dataframe = read_gold_data(it_file)

    # Declare output file paths
    es_gold_output = "./data/gold/es_ancora-ud-gold.tsv"
    it_gold_output = "./data/gold/it_isdt-ud-gold.tsv"

    # Write the gold label dataframes onto tsv file
    es_gold_dataframe.to_csv(es_gold_output, sep="\t", encoding="utf-8")
    it_gold_dataframe.to_csv(it_gold_output, sep="\t", encoding="utf-8")
    
    # Use spacy to predict the list of dependency relations and heads
    es_gold_dataframe["Relation_prediction"], es_gold_dataframe["Head_prediction"] = predict_dependencies(es_gold_dataframe, "es")
    it_gold_dataframe["Relation_prediction"], it_gold_dataframe["Head_prediction"] = predict_dependencies(it_gold_dataframe, "it")

    # Evaluate spacy's predictions against the gold label we had and save the accuracy of each sentence to new columns of relation accuracies and head accuracies
    es_gold_dataframe["Relation_accuracy"], es_gold_dataframe["Head_accuracy"], es_average_dep, es_average_head, es_perfect = evaluate_dependencies(es_gold_dataframe)
    it_gold_dataframe["Relation_accuracy"], it_gold_dataframe["Head_accuracy"], it_average_dep, it_average_head, it_perfect = evaluate_dependencies(it_gold_dataframe)

    # Print the average accuracies of sentence level relation accuracies and head accuracies for Spanish and Italian, then do the average of the two and also print the % of perfect sentences (perfect matches of both relation and head lists)
    print("\nSpanish results:\n")
    print("Average dependency relation accuracy:", es_average_dep)
    print("Average dependency head accuracy:", es_average_head)
    print("Average overall dependency accuracy:", (es_average_dep + es_average_head)/2)
    print("% Perfect:", es_perfect)

    print("\nItalian results:\n")
    print("Average dependency relation accuracy:", it_average_dep)
    print("Average dependency head accuracy:", it_average_head)
    print("Average overall dependency accuracy:", (it_average_dep + it_average_head)/2)
    print("% Perfect:", it_perfect)
    print()

    # Declare output file paths
    es_pred_output = "./results/es_ancora-ud-pred.tsv"
    it_pred_output = "./results/it_isdt-ud-pred.tsv"

    # Write the prediction dataframes onto tsv file
    es_gold_dataframe.to_csv(es_pred_output, sep="\t", encoding="utf-8")
    it_gold_dataframe.to_csv(it_pred_output, sep="\t", encoding="utf-8")

# Main
main()