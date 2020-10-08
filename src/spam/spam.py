import collections
from collections import Counter
from collections import defaultdict

import numpy as np
from math import log

from src.spam import util, svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For simplicity, you should split on whitespace, not
    punctuation or any other character. For normalization, you should convert
    everything to lowercase.  Please do not consider the empty string (" ") to be a word.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    return message.lower().strip().split()


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    basicDict = dict()
    for message in messages:
        words = get_words(message)
        for word in words:
            if word in basicDict:
             basicDict[word] = basicDict[word] + 1
            else:
                basicDict[word] = 1

    filteredDict = dict()
    for (key, value) in basicDict.items():
        if(value > 5):
            filteredDict[key] = value
    return filteredDict

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    rows = len(messages)
    cols = len(word_dictionary)
    matrix = np.zeros((rows, cols))
    vocabulary=  list(word_dictionary.keys())

    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            index = vocabulary.index(word) if word in vocabulary else -1
            if(index is not -1):
                matrix[i][index] = matrix[i][index] + 1

    return matrix


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    frequencies = np.zeros((2,len(matrix))) # Frecuencia de cada pablabra en cada label
    labelCounter = np.zeros(2) # Cantidad de cada label
    labelProbability = np.zeros(2) # probabilidad apriori de cada label

    for i, row in enumerate(matrix):
        label = labels[i]
        labelCounter[label] += 1
        for j, count in enumerate(row):
            frequencies[label][j] += count
    size = len(labels)

    labelProbability[0] = log(labelCounter[0] / size) # Probabilidad a priori de no ser spam
    labelProbability[1] = log(labelCounter[1] /size) # Probabilidad a priori de ser spam
    print(frequencies)
    return labelProbability, frequencies

def predict_from_naive_bayes_model(labelProbability, frequencies, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    predictions = np.zeros(len(matrix))
    for k, row in enumerate(matrix):
        likelyHood = labelProbability

        # Probability is not spam
        for i, feature in enumerate(row):
            probability = frequencies[0][i] / 2 # frecuencia / cantidad de labels
            likelyHood[0] += log(probability)
        #Probability is spam
        for i, feature in enumerate(row):
            probability = frequencies[1][i] / 2 # frecuencia / cantidad de labels
            likelyHood[1] += log(probability)

        if(likelyHood[1] > likelyHood[0]):
            predictions[k] = 1
        else:
            predictions[k] = 0

    return predictions


def get_top_five_naive_bayes_words(labelProbability, frequencies, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    return
   


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    return 


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    labelProbability, frequencies = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(labelProbability, frequencies, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(labelProbability, frequencies, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

def test_main():
    print(get_words("This is a test"))
    messages = ["test test test test test is test fail test", "fucking test this is is is is is is is is is a test", "asd asd"]
    vocabulary = create_dictionary(messages)
    print(vocabulary)

    matrix = transform_text(messages, vocabulary)

    fit_naive_bayes_model(matrix, [0, 1, 0])

if __name__ == "__main__":
    main()
    # test_main()