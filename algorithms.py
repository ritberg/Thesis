import nltk
import torch
import numpy
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_burrows_delta(authors: Tuple[str],
                          book_by_author_tokens: dict,
                          n_most_common: int):
    # Compute Burrow's delta for the two candidate texts
    authors = [*authors]

    # Step 1:   Create a joint corpus,
    #           Compute how often the words are used with nltk.FreqDist()
    #           Find the `n_most_common` words.
    joint_corpus = (book_by_author_tokens[authors[0]]
                    + book_by_author_tokens[authors[1]]
                    + book_by_author_tokens["Disputed"])

    joint_freq_dist = nltk.FreqDist(joint_corpus)
    most_common = list(joint_freq_dist.most_common(n_most_common))

    # Step 2:   Compute the percentage of use of each selected word
    #               in each subcorpus.
    word_shares = {}
    for author in authors:
        word_shares[author] = {}
        total_words = 0
        for word, joint_count in most_common:
            word_shares[author][word] \
                = book_by_author_tokens[author].count(word)
            total_words += word_shares[author][word]
        for word, joint_count in most_common:
            word_shares[author][word] /= total_words
    word_shares["Disputed"] = {}
    total_words = 0
    for word, joint_count in most_common:
        word_shares["Disputed"][word] \
            = book_by_author_tokens["Disputed"].count(word)
        total_words += word_shares["Disputed"][word]
    for word, joint_count in most_common:
        word_shares["Disputed"][word] /= total_words

    # Step 3:   Compute the mean and the standard deviation
    #                of the values computed at Step 2 across the authors

    mean_of_mean = {}
    std_of_means = {}
    for word, joint_count in most_common:
        word_shares_list = [word_shares[author][word]
                            for author in authors] \
                           + [word_shares["Disputed"][word]]
        mean_of_mean[word] = numpy.mean(word_shares_list)
        std_of_means[word] = numpy.std(word_shares_list)

    # Step 4:   Compute the Z scores for every subcorpus as follows:
    #           - Subtract  the mean of mean computed at Step 3 from the mean
    #               computed at Step 2
    #           - Divide the previous result by the standard deviation
    #               computed at Step 3
    z_scores = {}
    for author in authors + ["Disputed"]:
        z_scores[author] = {}
        for word, _ in most_common:
            z_scores[author][word] = (word_shares[author][word] -
                                      mean_of_mean[word]) / std_of_means[word]

    # Step 5:   Compute the Delta score (distance) between the Disputed text and
    #           each subcorpus according to the Burrows' Delta formula
    #           computing the mean across words of absolute values
    #           of the Z scores differences.

    delta = {}
    for author in authors:
        delta[author] = 0
        for word, _ in most_common:
            delta[author] += numpy.abs(
                z_scores[author][word] - z_scores["Disputed"][word])
        delta[author] /= n_most_common
    print(delta)
    # The function returns the delta scores among Disputed
    # and each subcorpus.
    return delta


def compute_chi_squared(authors: Tuple[str],
                        book_by_author_tokens: dict,
                        n_most_common: int):

    scores = {}
    for author in authors:

        # Step 1: Build a joint corpus and identify the `n_most_common` most
        #           frequent words in it.
        joint_corpus = (book_by_author_tokens[author] +
                        book_by_author_tokens["Disputed"])
        joint_freq_dist = nltk.FreqDist(joint_corpus)
        most_common = list(joint_freq_dist.most_common(n_most_common))[100:]

        # Step 2: Compute the percentage size of each subcorpus diving the
        #           number of words of each subcorpus by the number of words
        #           in the joint corpus.

        author_share = (len(book_by_author_tokens[author])
                        / len(joint_corpus))

        # Step 3:   Compute the chi squared score according to the
        #           Kilgariff's chi squared formula as follows:
        #               - Compute the frequency of every word in the joint
        #                   corpus and in every individual subcorpus.
        #               - Compute the expected frequency multiplying the
        #                   percentage size computed at Step 2 by the joint
        #                   corpus frequency.
        #               - For every word, compute the squared of the difference
        #                   between the expected frequency and every subcorpus
        #                   frequency and divide it by the expected frequency.
        #              - Compute the chi-squared score summing the values
        #                   computed above, at this Step.

        chisquared = 0
        for word, joint_count in most_common:

            author_count = book_by_author_tokens[author].count(word)
            disputed_count = book_by_author_tokens["Disputed"].count(word)

            expected_author_count = joint_count * author_share
            expected_disputed_count = joint_count * (1 - author_share)

            chisquared += ((author_count - expected_author_count) *
                           (author_count - expected_author_count) /
                           expected_author_count)

            chisquared += ((disputed_count - expected_disputed_count) *
                           (disputed_count - expected_disputed_count)
                           / expected_disputed_count)
        scores[author] = chisquared
    print(scores)
    # The function returns the chi squared scores computed at Step 3.
    return {"chisquared": scores}


def compute_ridge(features,
                  features_test,
                  target,
                  target_test,
                  alpha):
    classifier = RidgeClassifier(alpha=alpha, class_weight="balanced")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features_test = scaler.transform(features_test)
    classifier.fit(features, target)
    train_score = classifier.score(features, target)
    test_score = classifier.score(features_test, target_test)

    prediction = classifier.predict(features_test)
    weights = classifier.coef_[0]

    return {"prediction": prediction,
            "weights": weights,
            "train_score": train_score,
            "test_score": test_score}


def compute_svm(features,
                features_test,
                target,
                target_test,
                C):
    # Step 1: Use scikit-learn function `SVC` to create an SVM.
    # We specify that we look at linearly separable data
    # and fix a value of C. The value of C is proportional to the
    # number of points that are allowed to be classified on the wrong side.
    # In particular, a higher C means that the margin becomes narrower.
    classifier = SVC(kernel='linear', random_state=241, C=C)
    # Step 2: Apply standardization to the the data on both train and test.
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features_test = scaler.transform(features_test)
    classifier.fit(features, target)
    train_score = classifier.score(features, target)
    test_score = classifier.score(features_test, target_test)
    # Step 3: Apply SVM to make a prediction
    prediction = classifier.predict(features_test)
    # Step 4: Read the SVM coefficients.
    weights = classifier.coef_[0]
    # Finally, return the predictions, the model coefficients,
    # and train and test accuracies.
    return {"prediction": prediction,
            "weights": weights,
            "train_score": train_score,
            "test_score": test_score}


def compute_knn(features,
                features_test,
                target,
                target_test,
                n_neighbors):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features_test = scaler.transform(features_test)
    classifier.fit(features, target)
    train_score = classifier.score(features, target)
    test_score = classifier.score(features_test, target_test)

    prediction = classifier.predict(features_test)

    return {"prediction": prediction,
            "target": target,
            "target_test": target_test,
            "train_score": train_score,
            "test_score": test_score}


def compute_pca(features,
                features_test,
                n_components):
    pca = PCA(n_components=n_components,
              random_state=241)
    proj_features = pca.fit_transform(features)
    proj_features_test = pca.transform(features_test)
    return {"proj_features": proj_features,
            "proj_features_test": proj_features_test}


def compute_neural_networks(features,
                            features_test,
                            target,
                            target_test,
                            lr,
                            single_layer=False):
    # Step 1: Define the input size, i.e. the number of considered
    #           `n_most_common` words by reading the shape of the input
    #           features.
    input_size = features.shape[1]
    # Step 2: Standardize the input data, i.e. subtract to every word frequency
    # the mean and divide by the standard deviation.
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Step 3: Define the MLP layers using the package PyTorch.
    #           We use Linear layers and Tanh as non linear activation function.
    if single_layer:
        classifier = nn.Sequential(nn.Linear(input_size, 300),
                                   nn.Tanh(),
                                   nn.Linear(300, 2))
    else:
        classifier = nn.Sequential(nn.Linear(input_size, 300),
                                   nn.Tanh(),
                                   nn.Linear(300, 300),
                                   nn.Tanh(),
                                   nn.Linear(300, 2))
    # Step 4: Define the loss function
    w = numpy.mean(target)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([1 - w, w]))
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # Step 5: Convert the input features from numpy arrays to PyTorch tensors.
    cat_target = numpy.vstack(
        [numpy.array([1, 0]) if t == 0 else numpy.array([0, 1]) for t in
         target])
    features = torch.FloatTensor(
        features)
    cat_target = torch.FloatTensor(
        cat_target)

    # Step 6: Tune the networks weights using the Backpropagation algorithm
    #           (provided by PyTorch via the method `backward`.) iterating over
    #           the training set.  We provide more detailed comments in the
    #           loop.
    batch_size = 10
    for epoch in range(500):
        running_loss = 0.0

        for i in numpy.arange(0, features.shape[0], batch_size):

            data_ids = torch.randperm(features.shape[0])[:10]

            inputs = features[data_ids]
            labels = cat_target[data_ids]

            optimizer.zero_grad()
            # Apply the MLP to the input that is the computed frequencies.
            outputs = classifier(inputs)
            # Compute the loss using the MLP output and the labels.
            loss = criterion(outputs, labels)
            # Use backpropagation implemented in PyTorch in the
            #   method `backward`. It changes the networks weights if the loss
            #   is high, otherwise it leaves them almost unchanged.
            loss.backward()
            optimizer.step()

            # print loss every 2000 iterations. It should decrease. If not,
            # the network is not learning.
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    # Step 7: Compute the final predictions and accuracies.
    predictions = torch.sigmoid(
        classifier(
            features
        )
    )
    features_test = scaler.transform(features_test)
    predictions_test = torch.sigmoid(
        classifier(
            torch.FloatTensor(features_test)
        )
    )
    train_score = compute_accuracy(predictions, target)
    test_score = compute_accuracy(predictions_test, target_test)
    print(train_score, test_score)

    # Step 8: Save the network

    if single_layer:
        torch.save(classifier, open("single_layer_model_torch.pt", "wb"))
        torch.save(classifier.state_dict(),
                   open("single_layer_model_torch_w.pt",
                        "wb"))
    else:
        import os
        for i in [1, 2, 3, 18, 19, 20, 21, 22]:
            if not os.path.exists(
                    "ModernFrenchMakine_300_1300_model_torch" + str(i) + ".pt"):
                torch.save(classifier, open(
                    "ModernFrenchMakine_300_1300_model_torch" + str(i) + ".pt",
                    "wb"))
                torch.save(classifier.state_dict(),
                           open("ModernFrenchMakine_300_1300_model_torch" + str(
                               i) + ".pt",
                                "wb"))
                break
    return {"prediction": predictions_test,
            "train_score": train_score,
            "test_score": test_score}


def compute_accuracy(predictions, target):
    predictions = torch.nn.functional.softmax(predictions, dim=1)
    predictions = numpy.array(
        [0. if p[0] > p[1] else 1. for p in predictions.detach().numpy()])
    correct = (predictions == target).sum()
    return correct / target.shape[0]
