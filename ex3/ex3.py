# Imports
import os
import pandas
import nltk
import math
import re
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from rouge_score import rouge_scorer
from nltk.corpus import stopwords

### LLM Inference not working :( ###
# from pychatsonic.chat import ChatSonic
# Generate the summary with chatsonic LLM and return it
# def get_chatsonic_summary(text):
#     chat = ChatSonic("95a82416-128f-4019-bb20-b7f80d3f1e87", "en")
#     text = "Summarise the following text: " + text
#     summary = chat.ask(text)
#     return summary

# Generate the TF-IDF summary based on the scores of sentences and the given threshold, return it
def generate_summary(sentences, sentences_scores, threshold):
    i = 0
    summary = ""
    for sentence in sentences:
        if sentence in sentences_scores and sentences_scores[sentence] >= threshold:
            summary += sentence + " "
            i += 1
    return summary

# Calculate the average score of the sentences and return it
def get_average_score(sentences_scores):
    total = 0
    for item in sentences_scores:
        total += sentences_scores[item]
    average = (total/len(sentences_scores))
    return average

# Get the score of each sentence and return the values
def get_sentences_scores(tf_idf):
    sentences_scores = {}
    for sentence, frequency in tf_idf.items():
        sentence_score = 0
        word_count = len(frequency)
        for word, score in frequency.items():
            sentence_score += score
        sentences_scores[sentence] = sentence_score/word_count
    return sentences_scores

# Build TF-IDF matrix based on TF and IDF matrices by multiplying their values, return the matrix
def get_tf_idf(tf, idf):
    tf_idf_all = {}
    for (sentence1, frequency1), (sentence2, frequency2) in zip(tf.items(), idf.items()):
        tf_idf = {}
        for (word1, value1), (word2, value2) in zip(frequency1.items(), frequency2.items()):
            tf_idf[word1] = float(value1 * value2)
        tf_idf_all[sentence1] = tf_idf
    return tf_idf_all

# Build the IDF matrix based on word frequencies and the number of sentences in which each word appears, return the IDF matrix
def get_idf(word_frequencies, document_per_word, documents):
    idf_all = {}
    for sentence, frequency in word_frequencies.items():
        idf = {}
        for word in frequency.keys():
            idf[word] = math.log10(documents/float(document_per_word[word]))
        idf_all[sentence] = idf
    return idf_all

# Count the appearances of each word in the sentences (documents), return the values
def get_document_per_word(word_frequencies):
    document_per_word = {}
    for sentence, frequency in word_frequencies.items():
        for word, count in frequency.items():
            if word in document_per_word:
                document_per_word[word] += 1
            else:
                document_per_word[word] = 1
    return document_per_word

# Build TF matrix based on word frequencies and return it
def get_tf(word_frequencies):
    tf_all = {}
    for sentence, frequency in word_frequencies.items():
        tf = {}
        word_count = len(frequency)
        for word, count in frequency.items():
            tf[word] = count/word_count
        tf_all[sentence] = tf
    return tf_all

# Get the frequencies of the words in the sentences
def get_word_frequencies(sentences):
    frequencies = {}

    # Ignore stop words
    stopword = set(stopwords.words("english"))
    for sentence in sentences:
        frequency = {}
        words = word_tokenize(sentence)
        for word in words:

            # Lower-case the words and stem them so that they match better. return the frequencies
            word = word.lower()
            word = PorterStemmer().stem(word)
            if word not in stopword:
                if word in frequency:
                    frequency[word] += 1
                else:
                    frequency[word] = 1
            frequencies[sentence]=frequency
    return frequencies

# Build the TF-IDF summary given a text. We first calculate the TF, then IDF, then multiply their values
def get_tf_idf_summary(text):

    # Get the sentences, documents (length) and frequency of the words in the sentences
    sentences = nltk.sent_tokenize(text, language='english')
    documents = len(sentences)
    word_frequencies = get_word_frequencies(sentences)

    # Build TF matrix
    tf = get_tf(word_frequencies)

    # Build IDF matrix
    document_per_word = get_document_per_word(word_frequencies)
    idf = get_idf(word_frequencies, document_per_word, documents)

    # Build the TF-IDF matrix
    tf_idf = get_tf_idf(tf, idf)

    # Score the obtained TF-IDF to find the best sentences
    sentences_scores = get_sentences_scores(tf_idf)

    # Setting the average score of the sentences as the threshold for which sentences to pick
    threshold = get_average_score(sentences_scores)

    # Generating the TF-IDF summary based on the sentences scored over 1.1 times the threshold, return the summary
    summary = generate_summary(sentences, sentences_scores, 1.1 * threshold)
    return summary

# Build the LEAD-2 summary given a text
def get_lead_2(text):
    sentences = nltk.sent_tokenize(text, language='english')
    lead2 = sentences[0]+" "+sentences[1]
    return lead2

# Build the whole dataset
def build_dataset(directory):
    # Set the Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
    
    # Iterator to consult only the first 272 files, as there is only 272 files for the second dataset
    i = 0

    # Initialize lists as columns of the dataset
    stories_tokenized = []
    highlights_tokenized = []
    summaries_lead2 = []
    scores_lead2 = []
    summaries_tfidf = []
    scores_tfidf = []

    ### LLM Inference not working :( ###
    # summaries_chatsonic = []
    # scores_chatsonic = []

    # If dealing with the first directory
    if directory == "dm_stories":

        # Iterate through the first 272 files
        for filename in os.listdir(directory):
            file = os.path.join(directory, filename)
            with open(file, "r", encoding="utf-8") as f:

                # Pre-process the text and separate highlights from story
                whole_text = f.read()
                whole_text = whole_text.replace("\n", " ")
                whole_text = whole_text.replace("-RRB-", "")
                whole_text = whole_text.replace("-LRB-", "")
                split_text = whole_text.split("@highlight")
                highlights = ""
                story = split_text[0]
                for highlight in split_text[1:]:
                    highlights += highlight

                # Create the LEAD-2 summary and score it using rouge
                summary_lead2 = get_lead_2(story)          
                score_lead2 = scorer.score(highlights, summary_lead2)

                # Create the TF-IDF summary and score it using rouge
                summary_tfidf = get_tf_idf_summary(story)
                score_tfidf = scorer.score(highlights, summary_tfidf)

                ### LLM Inference not working :( ###
                # Generate the summary with chatsonic LLM and score it using rouge
                # summary_chatsonic = get_chatsonic_summary(story)
                # score_chatsonic = scorer.score(highlights, summary_chatsonic)

            # Append all of the info onto the lists (columns)
            stories_tokenized.append(story)
            highlights_tokenized.append(highlights)
            summaries_lead2.append(summary_lead2)
            scores_lead2.append(score_lead2)
            summaries_tfidf.append(summary_tfidf)
            scores_tfidf.append(score_tfidf)

            ### LLM Inference not working :( ###
            # summaries_chatsonic.append(summary_chatsonic)
            # scores_chatsonic.append(score_chatsonic)

            # Advance iterator and stop at 272 to have the same number of summaries as the other dataset, then return all the columns
            i += 1
            if i==272:
                return stories_tokenized, highlights_tokenized, summaries_lead2, scores_lead2, summaries_tfidf, scores_tfidf
            
    # If dealing with the second directory
    elif directory == "aligned_stories":

        # Define directory for the stories and for the summaries
        directory_stories = directory + "\\stories"
        directory_summaries = directory + "\\extractive_summaries"

        # Save list of all existing summaries to discard the rest of stories
        story_name_list = []
        for filename in os.listdir(directory_summaries):
            story_name_list.append(filename.replace(".union", ""))

        # Save all of the stories which filenames coincide with the list of summaries in the column of stories
        for filename in os.listdir(directory_stories):
            if filename.replace(".story", "") in story_name_list:
                file = os.path.join(directory_stories, filename)
                with open(file, "r", encoding="utf-8") as f:
                    story = f.read()
                stories_tokenized.append(story)

        # Save all of the summaries and generate model summaries based on iterator on list of stories
        i = 0
        for filename in os.listdir(directory_summaries):
            file = os.path.join(directory_summaries, filename)
            with open(file, "r", encoding="utf-8") as f:
                highlights = f.read()
                
                # Build and score LEAD-2 summaries
                summary_lead2 = get_lead_2(stories_tokenized[i])          
                score_lead2 = scorer.score(highlights, summary_lead2)

                # Build and score TF-IDF summaries
                summary_tfidf = get_tf_idf_summary(stories_tokenized[i])
                score_tfidf = scorer.score(highlights, summary_tfidf)
            i += 1

            # Update the columns
            highlights_tokenized.append(highlights)
            summaries_lead2.append(summary_lead2)
            scores_lead2.append(score_lead2)
            summaries_tfidf.append(summary_tfidf)
            scores_tfidf.append(score_tfidf)

            ### LLM Inference not working :( ###
            # summaries_chatsonic.append(summary_chatsonic)
            # scores_chatsonic.append(score_chatsonic)

        # Return columns
        return stories_tokenized, highlights_tokenized, summaries_lead2, scores_lead2, summaries_tfidf, scores_tfidf

# Output a csv with the info about the stories, the reference summary (highlights), the lead-2 summaries and scores, the tf-idf summaries and scores
def output_data(tokenized_stories, tokenized_highlights, summaries_lead2, scores_lead2, summaries_tfidf, scores_tfidf, directory1):
    df = pandas.DataFrame()
    df["Stories"]=tokenized_stories
    df["Highlights"]=tokenized_highlights
    df["Summaries_LEAD-2"]=summaries_lead2
    df["Scores_LEAD-2"]=scores_lead2
    df["Summaries_TF-IDF"]=summaries_tfidf
    df["Scores_TF-IDF"]=scores_tfidf
    output="./results/" + directory1.split("_")[0] + "_stories" + "_results.tsv"
    df.to_csv(output, sep="\t", encoding="utf-8")

# Main
def main():
    # Build dataset and output summary and rouge scores results for dm_stories
    directory1 = "dm_stories"
    tokenized_stories, tokenized_highlights, summaries_lead2, scores_lead2, summaries_tfidf, scores_tfidf  = build_dataset(directory1)
    output_data(tokenized_stories, tokenized_highlights, summaries_lead2, scores_lead2, summaries_tfidf, scores_tfidf, directory1)

    # Build dataset and output summary and rouge scores results for aligned_stories
    directory2 = "aligned_stories"
    tokenized_stories, tokenized_highlights, summaries_lead2, scores_lead2, summaries_tfidf, scores_tfidf = build_dataset(directory2)
    output_data(tokenized_stories, tokenized_highlights, summaries_lead2, scores_lead2, summaries_tfidf, scores_tfidf, directory2)
main()