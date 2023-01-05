import string
from bs4 import BeautifulSoup

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
             "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
             "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his",
             "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell",
             "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them",
             "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this",
             "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were",
             "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos",
             "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours",
             "yourself",
             "yourselves"]

# This table is used to remove punctuation from a string. From the documentation:
# If a third argument is given, it must be a string, whose characters will be
# deleted from the input string.
punctuation_translation_table = str.maketrans('', '', string.punctuation)

if __name__ == '__main__':
    test_sentence = """I'm a sentence with punctuation!!! and html tags <br />
     and even stopwords like 'and'. To make things worse, I contain numbers 
     like 1, 2, 3 and 4. and html tags containing punctuation like <br /> and <i>.</i>.
     Contractions like I'm, I don't and he'll are also a problem. He'll will become hell
     for example which has a totally different meaning."""

    print(f"Original sentence: {test_sentence}")

    # First convert to lower case
    test_sentence = test_sentence.lower()

    # Remove html tags
    soup = BeautifulSoup(test_sentence, features="html.parser")
    sentence = soup.get_text()

    print(f"Remove html tags: {sentence}")

    # Next remove punctuation
    sentence = sentence.translate(punctuation_translation_table)
    print(f"Remove punctuation: {sentence}")

    # Now remove stopwords
    sentence = ' '.join([word for word in sentence.split() if word not in stopwords])
    print(f"Remove stopwords: {sentence}")
