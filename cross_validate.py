import copy
import pickle
from loading import loading
from algorithms import compute_knn, \
    compute_pca, \
    compute_svm, \
    compute_ridge, \
    compute_chi_squared, \
    compute_neural_networks, \
    compute_burrows_delta
from plot import ridge_bar_plot

algo_functions_mapping = {
    "Ridge": compute_ridge,
    "K-NN": compute_knn,
    "PCA": compute_pca,
    "SVM": compute_svm,
    "chi_squared": compute_chi_squared,
    "NeuralNet": compute_neural_networks,
    "burrows_delta": compute_burrows_delta
}


def cross_validate(books,
                   folder,
                   algorithm,
                   start_range,
                   end_range,
                   extra_words,
                   save_file_name,
                   **kwargs):
    """
    :param books: dictionary containing authors as keys and list of books as
                  values. It does not contain Disputed. For example:
                  books = {"Makine":[0, 1, 2, 3, 4],
                           "Proust":[5, 6, 7, 8]}
                  Each of these texts will be considered as Disputed at one
                  round of cross_validation.
    :return:
    """
    all_books = []
    operae = {}
    for book_list in books.values():
        all_books = all_books + book_list
    # The cross validation considers each document in the corpus
    # as a Disputed by turn. This is done in the following for loop.
    for disputed in all_books:
        for i, (author, book_list) in enumerate(books.items()):
            book_list_copy = copy.deepcopy(book_list)
            if disputed in book_list:
                book_list_copy.remove(disputed)
                true_disputed_class = i
            operae[author] = book_list_copy
        operae["Disputed"] = [disputed]
        print(operae)
        # After separating the Disputed text from the main corpus,
        # we call the loading function.
        # The function returns tokens lists for Burrows' Delta/
        # Chi-squared and features vectors for the
        # machine learning methods.
        if algorithm not in ["chi_squared", "burrows_delta"]:
            # The loading functions parameters can be changed. For example,
            # we can vary the range of words, the segment length and adding
            # additional words in the features.
            features, features_test, targets, target_test, keys = \
                loading(operae,
                        folder,
                        algorithm,
                        start_range=start_range,
                        end_range=end_range,
                        true_disputed_class=true_disputed_class,
                        extra_words=extra_words,
                        segment_length=400,
                        )
            # After loading the features, we call a function (e.g. compute_svm)
            # that train a model.
            # The available functions are in the dictionary
            # `algo_functions_mapping`. The functions return a dictionary
            # containing the predictions and the accuracy on test and train
            # sets.
            returns = algo_functions_mapping[algorithm](features,
                                                        targets,
                                                        features_test,
                                                        target_test,
                                                        **kwargs)

            test_accuracy = returns["test_score"]
            print(returns["prediction"], "predictions")
            print(f"For disputed book {disputed} : "
                  f"test score : {test_accuracy})")
            returns["keys"] = keys
        else:
            book_by_authors_token = loading(operae,
                                            folder,
                                            algorithm,
                                            start_range=start_range,
                                            end_range=end_range)
            returns = algo_functions_mapping[algorithm](books.keys(),
                                                        book_by_authors_token,
                                                        **kwargs)

        # Save the features to allow reusing them for training
        # another algorithm.
        pickle.dump(returns, open(save_file_name + str(disputed) + ".p", "wb"))


"""
if __name__ == '__main__':
    books = {"Makine": [1, 2, 3, 4, 5],
             "Proust": [6, 7, 8, 9, 10]}
    folder = "dataMP_1"
    algorithm = "NeuralNet"
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=300,
                   end_range=1300,
                   save_file_name="ProustMakine_300_1300_NeuralNet",
                   extra_words=["intérieurement", "stupidement"]
                   )
cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=500,
                   end_range=1000,
                   save_file_name="ProustMakine_500_1000_NeuralNet",
                   extra_words=["intérieurement", "stupidement"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=500,
                   end_range=600,
                   save_file_name="ProustMakine_500_600_NeuralNet",
                   extra_words=["intérieurement", "stupidement"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=600,
                   end_range=700,
                   save_file_name="ProustMakine_600_700_NeuralNet",
                   extra_words=["intérieurement", "stupidement"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=700,
                   end_range=800,
                   save_file_name="ProustMakine_700_800_NeuralNet",
                   extra_words=["intérieurement", "stupidement"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=800,
                   end_range=900,
                   save_file_name="ProustMakine_800_900_NeuralNet",
                   extra_words=["intérieurement", "stupidement"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=900,
                   end_range=1000,
                   save_file_name="ProustMakine_900_1000_NeuralNet",
                   extra_words=["intérieurement", "stupidement"]
                   )"""

"""
books = {"Russian-French": [1, 2, 3, 4, 9],
         "French-French": [5, 6, 7, 8]}
folder = "dataMixed_1"
for (start_range, end_range) in zip([500, 600, 700, 800, 900, 300, 300, 500],
                                    [600, 700, 800, 900, 1000, 800, 1300, 1000]):
    cross_validate(books,
                   folder,
                   "Ridge",
                   alpha=10,
                   start_range=start_range,
                   end_range=end_range,
                   save_file_name="alpha10RussianFrenchMakine_" + str(start_range) + "_" +
                                  str(end_range) + "_Ridge",
                   extra_words=["intérieurement", "stupidement"])

    cross_validate(books,
                   folder,
                   "SVM",
                   C=1e-5,
                   start_range=start_range,
                   end_range=end_range,
                   save_file_name="C1e-5RussianFrenchMakine_" + str(
                       start_range) + "_"
                                  + str(end_range) + "_SVM",
                   extra_words=["intérieurement", "stupidement"]
                   )
    cross_validate(books, folder, "K-NN", n_neighbors=2, #7
                   start_range=start_range,
                   end_range=end_range,
                   save_file_name="neigh2RussianFrenchMakine_" + str(
                       start_range) + "_"
                                  + str(end_range) + "_KNN",
                   extra_words=["intérieurement", "stupidement"])
# cross_validate(books, folder, algorithm)
"""
"""books = {"Russian-French": [1, 2, 3, 4, 9],
         "French-French": [5, 6, 7, 8]}
folder = "dataMixed_1"
algorithm = "K-NN"
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=300,
               end_range=1300,
               save_file_name="RussianFrenchMakine_300_1300_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])"""
"""
books = {"Russian-French": [1, 2, 3, 4, 9],
         "French-French": [5, 6, 7, 8]}
folder = "dataMixed_1"
algorithm = "K-NN"
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=600,
               end_range=700,
               save_file_name="RussianFrenchMakine_600_700_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])
books = {"Russian-French": [1, 2, 3, 4, 9],
         "French-French": [5, 6, 7, 8]}
folder = "dataMixed_1"
algorithm = "K-NN"
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=700,
               end_range=800,
               save_file_name="RussianFrenchMakine_700_800_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])
books = {"Russian-French": [1, 2, 3, 4, 9],
         "French-French": [5, 6, 7, 8]}
folder = "dataMixed_1"
algorithm = "K-NN"
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=800,
               end_range=900,
               save_file_name="RussianFrenchMakine_800_900_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])
books = {"Russian-French": [1, 2, 3, 4, 9],
         "French-French": [5, 6, 7, 8]}
folder = "dataMixed_1"
algorithm = "K-NN"
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=900,
               end_range=1000,
               save_file_name="RussianFrenchMakine_900_1000_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])
books = {"Russian-French": [1, 2, 3, 4, 9],
         "French-French": [5, 6, 7, 8]}
folder = "dataMixed_1"
algorithm = "K-NN"
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=300,
               end_range=800,
               save_file_name="RussianFrenchMakine_300_800_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=500,
               end_range=1000,
               save_file_name="RussianFrenchMakine_500_1000_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])
cross_validate(books, folder, algorithm, n_neighbors=7, start_range=300,
               end_range=1300,
               save_file_name="RussianFrenchMakine_300_1300_KNN",
               extra_words=["verdâtre",
                            "brunâtre",
                            "grisâtre",
                            "blanchâtre",
                            "gouttelettes",
                            "dansotter"])"""
"""
if __name__ == '__main__':
    books = {"Russian-French": [1, 2, 3, 4, 9],
             "French-French": [5, 6, 7, 8]}
    folder = "dataMixed_1"
    algorithm = "NeuralNet"
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=300,
                   end_range=800,
                   save_file_name="RussianFrenchMakine_300_800_NeuralNet",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )
    books = {"Russian-French": [1, 2, 3, 4, 9],
             "French-French": [5, 6, 7, 8]}
    folder = "dataMixed_1"
    algorithm = "NeuralNet"
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=500,
                   end_range=1000,
                   save_file_name="RussianFrenchMakine_500_1000_NeuralNet",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=500,
                   end_range=600,
                   save_file_name="RussianFrenchMakine_500_600_NeuralNet",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=600,
                   end_range=700,
                   save_file_name="RussianFrenchMakine_600_700_NeuralNet",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=700,
                   end_range=800,
                   save_file_name="RussianFrenchMakine_700_800_NeuralNet",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=800,
                   end_range=900,
                   save_file_name="RussianFrenchMakine_800_900_NeuralNet",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )
    cross_validate(books,
                   folder,
                   algorithm,
                   lr=1e-3,
                   single_layer=False,
                   start_range=900,
                   end_range=1000,
                   save_file_name="RussianFrenchMakine_900_1000_NeuralNet",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )"""
# books = {"Makine": [1, 2],
#         "Proust": [5, 6]}
# folder = "dataMP_1"
# algorithm = "SVM"
# C = 10
"""books = {"Makine": [1, 2, 3, 4, 5],
         "Proust": [6, 7, 8, 9, 10]}
folder = "dataMP_1"
algorithm = "Ridge"
cross_validate(books,
               folder,
               algorithm,
               start_range=500,
               end_range=600,
               save_file_name = "MakineProust_500_600_Ridge",
               extra_words=["intérieurement", "stupidement"])"""
"""
books = {"russian-french": [1, 2, 3, 4, 9],
         "french-french": [5, 6, 7, 8]}
folder = "dataMixed_1"

for (start_range, end_range) in zip([800, 900, 300, 300, 500],
                                    [900, 1000, 800, 1300,
                                     1000]):
    cross_validate(books,
                   folder,
                   "Ridge",
                   start_range=start_range,
                   end_range=end_range,
                   save_file_name="RussianFrenchMakine_" + str(start_range) + "_" +
                                  str(end_range) + "_Ridge",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"])

    cross_validate(books,
                   folder,
                   "SVM",
                   C=10,
                   start_range=start_range,
                   end_range=end_range,
                   save_file_name="RussianFrenchMakine_" + str(
                       start_range) + "_"
                                  + str(end_range) + "_SVM",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"]
                   )
    cross_validate(books, folder, "K-NN", n_neighbors=7,
                   start_range=start_range,
                   end_range=end_range,
                   save_file_name="RussianFrenchMakine_" + str(
                       start_range) + "_"
                                  + str(end_range) + "_KNN",
                   extra_words=["verdâtre",
                                "brunâtre",
                                "grisâtre",
                                "blanchâtre",
                                "gouttelettes",
                                "dansotter"])
    #Riprendere da 800-->900\

if __name__ == '__main__':
    #books = {"Russian-French": [1, 2, 3], #, 4, 9],#, 2, 3, 4, 9],
             "French-French": [18, 19, 20, 21, 22]} #, 23]}
    folder = "dataMixed_1"

    #for (start_range, end_range) in zip([500, 600, 700, 800, 900], #, 700, 800, 900],
    #                                    [600, 700, 800, 900, 1000]): #, 800, 900, 1000]):
        algorithm = "NeuralNet"
        cross_validate(books,
                       folder,
                       algorithm,
                       lr=1e-3,
                       single_layer=False,
                       start_range=start_range,
                       end_range=end_range,
                       save_file_name="DeterminativesModernFrench" + str(start_range) + "_" +
                                          str(end_range) + "_NeuralNet",
                       extra_words=["ce", "cette", "ces"]
                       )
        cross_validate(books,
                       folder,
                       "Ridge",
                       alpha=1,
                       start_range=start_range,
                       end_range=end_range,
                       save_file_name="DeterminativesModernFrench" + str(start_range) + "_" +
                                      str(end_range) + "_Ridge",
                       extra_words=["ce", "cette", "ces"])

        cross_validate(books,
                       folder,
                       "SVM",
                       C=1,
                       start_range=start_range,
                       end_range=end_range,
                       save_file_name="DeterminativesModernFrench" + str(
                           start_range) + "_"
                                      + str(end_range) + "_SVM",
                       extra_words=["ce", "cette", "ces"]
                       )
        cross_validate(books, folder, "K-NN", n_neighbors=2, #7
                       start_range=start_range,
                       end_range=end_range,
                       save_file_name="DeterminativesModernFrench" + str(
                           start_range) + "_"
                                      + str(end_range) + "_KNN",
                       extra_words=["ce", "cette", "ces"])"""
# cross_validate(books, folder, algorithm)

if __name__ == '__main__':
    books = {"Russian-French": [1, 2, 3, 4, 9],  # , 4, 9],#, 2, 3, 4, 9],
             "French-French": [18, 19, 20, 21, 22]}  # , 23]}
    folder = "dataMixed_1"
    #Contemporary French
    cross_validate(books,
                   folder,
                   "chi_squared",
                   start_range=0,
                   end_range=3000,
                   n_most_common=200, #500
                   save_file_name="chi_squared_classic",
                   extra_words=["ce", "cette", "ces"])
    """

    # Classic French
    books = {"Russian-French": [1, 2, 3, 4, 9],  # , 4, 9],#, 2, 3, 4, 9],
             "French-French": [5, 6, 7, 8]}  # , 23]}
    folder = "dataMixed_1"
    
    cross_validate(books,
                   folder,
                   "chi_squared",
                   start_range=0,
                   end_range=1000,
                   n_most_common=500, #500
                   save_file_name="chi_squared_classic_true",
                   extra_words=["ce", "cette", "ces"])"""