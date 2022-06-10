"""
Access the books saved as txt files and count the word frequencies
"""
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from copy import deepcopy
from typing import List

# Remove stop words only if `remove_stop_words` is set to True.
remove_stop_words = True
if remove_stop_words:
    fr_stops = stopwords.words('french') + ["a", "swann", "d'un", "si",
                                            # , "cette", "ces"
                                            "qu'elle", "elles", "s'il", "la",
                                            "là",
                                            "ah", "à", "ai", "ont", "aux", "au",
                                            "le", "les", "de", "des", "du",
                                            "un",
                                            "une",
                                            "et", "elle", "il", "ne", "en", "y",
                                            "sa", "son", "ses", "nos", "vos",
                                            "mon",
                                            "ma", "ton", "ta", "leur", "leurs",
                                            "ça", "avait", "avaient", "ont",
                                            # "ce"
                                            "auront", "aurais", "aurait", "as",
                                            "quand", "ai", "qu'est-ce"]
else:

    fr_stops = []


def loading(papers: dict,
            folder: str,
            algorithm: str,
            **kwargs):
    print(kwargs, "loading")
    if algorithm == "chi_squared" or algorithm == "burrows_delta":
        # For chi squared and Burrows delta extract the word list
        # of the corpora and tokenize them.
        book_by_author = get_book_by_author(papers,
                                            folder,
                                            single_books=False)
        book_by_author_tokens = get_book_by_author_tokens(book_by_author)
        return book_by_author_tokens
    elif algorithm in ["Ridge", "NeuralNet", "K-NN", "PCA", "SVM"]:
        # In this elif (for Ridge, Neural Net, KNN, PCA and SVM),
        # we compute the frequency of the most common words in the joint corpus.

        if "true_disputed_class" not in kwargs.keys():
            raise ValueError("You need to pass the `true_disputed_class` "
                             "to use Ridge !")
        # Extract from the corpus the list of words and tokenize it.
        book_by_author = get_book_by_author(papers,
                                            folder,
                                            single_books=False)
        book_by_author_tokens = get_book_by_author_tokens(book_by_author)
        if "segment_length" in kwargs.keys():
            # Split the token list in segments of the length given as input.
            author_segments = get_author_segments(book_by_author_tokens,
                                                  segment_length=kwargs[
                                                      "segment_length"])
            del kwargs["segment_length"]
        else:
            author_segments = get_author_segments(book_by_author_tokens)
        # Use the function `get_features` to compute the frequency of the
        # tokens in the token list. The function `get_features` also
        # splits the data into a train set and a test set and outputs
        # `targets_train`and `targets_test`. These are vectors containing 0 for
        # the texts belonging to the first subcorpus and 1 to the second
        # subcorpus.

        features, targets, features_test, target_test, keys = \
            get_features(book_by_author_tokens,
                         author_segments,
                         **kwargs
                         )
        disputed_text = papers["Disputed"][0]
        start = kwargs["start_range"]
        end = kwargs["end_range"]

        # Save the computed features.
        with open(f"featuresModernFrench_{disputed_text}_{start}_{end}.pkl",
                  "wb") as f:
            pickle.dump((features,
                         features_test,
                         targets,
                         target_test,
                         keys), f)
        # Return features and targets for both train and test sets.
        return features, features_test, targets, target_test, keys


# Hereafter we have the functions that are called by the function `loading`


def read_files_into_string(filenames: List[str],
                           folder: str):
    strings = []
    for filename in filenames:
        with open(f'{folder}/book_{filename}.txt', encoding="utf8") as f:
            strings.append(f.read())
    return '\n'.join(strings)


def get_book_by_author(papers: dict,
                       folder: str,
                       single_books: bool = False,
                       ):
    # Extract a list of words from the corpus in string format. If
    # `single_books` is True, the function returns different list of words
    # for every book.
    # If `single_books` is False, the function returns a common word list for
    # the whole corpus, without dividing it by books.
    book_by_author = {}
    if not single_books:

        for author, files in papers.items():
            book_by_author[author] = read_files_into_string(files, folder)
            ls = list(book_by_author[author].split(" "))
            for word in deepcopy(ls):
                if word in fr_stops:
                    ls.remove(word)
            book_by_author[author] = " ".join(word for word in ls)
    else:
        for author, files in papers.items():
            for file in files:
                if author not in book_by_author.keys():
                    book_by_author[author] = {}
                file_content = read_files_into_string([file])
                ls = list(file_content.split(" "))
                for word in deepcopy(ls):
                    if word in fr_stops:
                        ls.remove(word)
                book_by_author[author][file] = " ".join(word for word in ls)
    return book_by_author


def get_book_by_author_tokens(book_by_author):
    # It converts a list of words passed as input into a list tokens with
    # the method `nltk.word_tokenize`.
    book_by_author_tokens = {}
    for author in book_by_author.keys():
        if book_by_author[author] is dict:
            raise ValueError("Don't use stylometry with single books!!")
        tokens = nltk.word_tokenize(book_by_author[author], language='french')
        # Filter out punctuation
        book_by_author_tokens[author] = ([token for token in tokens
                                          if any(c.isalpha() for c in token)])

    for author in book_by_author.keys():
        book_by_author_tokens[author] = (
            [token.lower() for token in book_by_author_tokens[author]])
    book_by_author_tokens["Disputed"] = (
        [token.lower() for token in book_by_author_tokens["Disputed"]])
    return book_by_author_tokens


def get_author_segments(book_by_author_tokens,
                        segment_length: int = 1700):
    # Function to split a token lists in segments of length `segment_length`.
    # The length of the segments can be changed with the value
    # of `segment_length`.
    author_segments = {}

    for author in book_by_author_tokens.keys():
        ls = book_by_author_tokens[author]
        segment_id = 0
        author_segments[author] = [[]]
        for i, word in enumerate(ls):
            author_segments[author][segment_id].append(word)
            if i % segment_length == 0:
                segment_id += 1
                author_segments[author].append([])
    return author_segments


def get_features(book_by_author_tokens,
                 author_segments: dict,
                 true_disputed_class: int,
                 start_range: int = 9900,
                 end_range: int = 10000,
                 **kwargs):
    # Function that computes the features for Ridge, SVM, KNN,
    # Neural Networks etc.
    # In particular, we compute the frequency of every token in each of the
    # token segments obtained with the function `get_author_segments`.
    print(kwargs, "features")
    joint_corpus = None
    features = []
    features_test = []
    target = []
    target_test = []
    # Build a joint corpus and identify the `n_most_common` most
    # frequent words in it.
    for key in book_by_author_tokens.keys():
        if joint_corpus is None:
            joint_corpus = book_by_author_tokens[key]

        joint_corpus = joint_corpus + book_by_author_tokens[key]
    joint_freq_dist = nltk.FreqDist(joint_corpus)
    most_common = list(joint_freq_dist.most_common(end_range))
    # Possibly add extra words like determinative pronouns,
    # articles or color adjectives. Useful to add candidate
    # words to detect interference.
    if "extra_words" in kwargs.keys():
        for word, count in joint_freq_dist.most_common(20000):
            if word in kwargs["extra_words"]:
                most_common.append((word, count))
                print("Appended " + word)

    # For every author, iterate over the text segments and compute the frequency
    # of every common word in that segment. The vector obtained with the
    # frequencies of all the common words in a segment is used as feature for
    # that segment.
    # The features for Disputed are kept separate and used as features for the
    # test set.
    for k, author in enumerate(book_by_author_tokens.keys()):
        print(k, "k")
        for i in range(len(author_segments[author])):
            counts = []
            keys = []
            for word, joint_count in most_common[start_range:]:
                segment_count = author_segments[author][i].count(word)
                counts.append(segment_count)
                keys.append(word)
            if author == "Disputed":
                target_test.append(true_disputed_class)
                features_test.append(counts)
            else:
                features.append(counts)
                target.append(k)
    # Output features for train and test sets as well as for the corresponding
    # labels. `target` is used by the algorithms as labels while
    # `target_test` is unknown to the algorithm.
    # `features_test` and `target_test` are used to give the final
    # prediction and accuracy.
    return (np.array(features).astype(float),
            np.array(features_test).astype(float),
            np.array(target).astype(float),
            np.array(target_test).astype(float),
            keys)
