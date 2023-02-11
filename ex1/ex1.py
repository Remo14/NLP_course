# Imports
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import en_core_web_sm
import ru_core_news_sm
import numpy as np

# Plot the perfect Zipf's curve to contrast it against our results
def zipfs_curve(depth):

        # Setting the values for a perfect zipf's curve - The second value being half of the length, the third value a third of the length and so on - for comparison and plotting it in grey dots
        ziffian_values = [depth/i for i in range(1, depth + 1)]
        x_axis = np.array(range(0, depth))
        y_axis = np.array(ziffian_values)
        plt.plot(x_axis, y_axis, label="Zipf's curve", ls=":", color="grey")

# Plot the function with rank and frequency values and the given title, and the perfect Zipf's curve to compare to our results. Limit the axes to view it properly. Export the results in TSV
def zipf_plot(dataframe, title):

        # Establishing the depth to which we limit the view and plotting our data
        width = dataframe["rank"].iloc[-1]
        height = dataframe["frequency"].iloc[0]
        sns.relplot(x="rank", y="frequency", data=dataframe, color="orange")

        # Saving the results in a tsv file
        filename = "./results/" + title.split(", by ")[1].lower().split(" ")[1].replace(" ", "_") + "_results.tsv"
        dataframe.to_csv(filename, sep = "\t", encoding="utf-8", index=False)

        # Limiting the view of the Figure
        plt.xlim(0, width)
        plt.ylim(0, height)

        # Adding the title and the zipfs curve for comparison
        plt.title(title)
        zipfs_curve(width)
        plt.show()
        plt.close()

# Create a new dataframe with the content words, their frequency and their rank
def build_dataframe(words):

        # Count the frequency of the content words and create column for words and frequency, then sort by frequency
        dataframe = pd.DataFrame.from_records(list(dict(Counter(words)).items()), columns=['word','frequency'])
        dataframe = dataframe.sort_values(by=['frequency'], ascending=False)

        # Create new column for the rank
        dataframe['rank'] = list(range(1, len(dataframe) + 1))
        return dataframe

# Get a list of all the content words from text in the specified language (English or Russian)
def get_words(text, language):

        # Either use the English model or the Russian one
        if language == "English":
                nlp = en_core_web_sm.load()
        elif language == "Russian":
                nlp = ru_core_news_sm.load()
        else:
                print("\nUnsupported language.\n")
                return
        doc = nlp(text)

        # Get all words that are not stop words or punctuation (Content words)
        words = [token.text for token in doc if not token.is_stop and not token.is_punct and token.text.isalpha()]
        return words

# Read the file and return the text as a string
def read_file(filename):
        text = open(filename, "r", encoding="utf-8")
        text = text.read()
        return text

# Main
def main():

        # Leaves of Grass from Walt Whitman as our English text / Pasternak's poems as our Russian text
        filename_en = "./data/whitman.txt"
        filename_ru = "./data/pasternak.txt"

        # Read both files as text
        text_en = read_file(filename_en)
        text_ru = read_file(filename_ru)

        # Create list of words for both languages
        words_en = get_words(text_en, "English")
        words_ru = get_words(text_ru, "Russian")
        
        # Build the two dataframes
        dataframe_en = build_dataframe(words_en)
        dataframe_ru = build_dataframe(words_ru)

        # Draw the function with rank and frequency as axes for both languages and add the figure title. Export results into TSV
        zipf_plot(dataframe_en, "Leaves of Grass, by Walt Whitman")
        zipf_plot(dataframe_ru, "Poems 1912-1914, by Boris Pasternak")
main()