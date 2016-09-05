from keras.engine import Merge
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Embedding, GRU
from keras.preprocessing import sequence
import os
import cPickle as pickle
import numpy as np
from itertools import izip
from sklearn.cross_validation import KFold
import random
from data_preprocessing import represent_as_kmers
from itertools import product
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def prepare_data(f_in_k_mers, f_in_sequences, taxa_index_dict):
    f_in_k_mers.next()
    f_in_sequences.next()

    X_k_mers = []
    X_sequences = []
    Y = []

    for line_k_mers, line_sequences in izip(f_in_k_mers, f_in_sequences):
        y = [0] * len(taxa_index_dict)
        p, c, o, f, g = line_k_mers.split(",")[0:5]

        y[taxa_index_dict["p_" + p]] = 1
        y[taxa_index_dict["c_" + c]] = 1
        y[taxa_index_dict["o_" + o]] = 1
        y[taxa_index_dict["f_" + f]] = 1
        y[taxa_index_dict["g_" + g]] = 1
        x_k_mers = map(int, line_k_mers.split(",")[5:])
        x_sequences = line_sequences.strip().split(",")[-1]

        X_k_mers.append(x_k_mers)
        X_sequences.append(x_sequences)
        Y.append(y)

    return np.array(X_k_mers), np.array(X_sequences), np.array(Y)


def evaluate_model(Y_true, Y_pred):

    Y_pred= np.asarray(Y_pred)
    low_values_indices = Y_pred.transpose() < Y_pred.max(axis=1)  # Where values are low
    Y_pred.transpose()[low_values_indices] = 0
    Y_pred.transpose()[low_values_indices.__invert__()] = 1

    accuracy = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average='macro')

    return accuracy, f1


def map_sequences(sequences, keywords):
    k = len(keywords[0])
    new_sequences = []
    for sequence in sequences:
        s = []
        for i in range(0, len(sequence) - k):
            s.append(keywords.index(sequence[i:i + k]) + 1)

        new_sequences.append(s)

    return np.array(new_sequences)


def create_500_data(sequences, keywords):
    new_sequences = []
    new_kmers = []
    for sequence in sequences:
        i = random.randint(0, len(sequence)-500)
        new_sequence = sequence[i:i+500]
        new_sequences.append(new_sequence)
        new_kmers.append(represent_as_kmers(5,new_sequence,keywords))
    return np.array(new_kmers), np.array(new_sequences)


def put_to_vizualization_data(vizualization_data, results_accuracy, results_f1, method):

    vizualization_data.append(["phylum", results_accuracy[0], results_f1[0], method])
    vizualization_data.append(["class", results_accuracy[1], results_f1[1], method])
    vizualization_data.append(["order", results_accuracy[2], results_f1[2], method])
    vizualization_data.append(["family", results_accuracy[3], results_f1[3], method])
    vizualization_data.append(["genus", results_accuracy[4], results_f1[4], method])

    return vizualization_data


def make_a_plot(data, title):
    df = DataFrame(data, columns=["taxa", "accurancy", "f1", "method"])
    sns.pointplot(x="taxa", y="accurancy", hue="method", data=df,
                       palette={"cnn": "blue", "rnn": "orange", "rf":"purple", "b_rnn":"forestgreen", "cnn+rnn":"yellow"},
                       markers=["o", "o", "o", "o", "o"], linestyles=["--", "--", "--", "--", "--"])

    plt.savefig(title + "_accuracy.pdf")
    plt.close()
    sns.pointplot(x="taxa", y="f1", hue="method", data=df,
                  palette={"cnn": "blue", "rnn": "orange", "rf": "purple", "b_rnn": "forestgreen", "cnn+rnn": "yellow"},
                  markers=["o", "o", "o", "o", "o"], linestyles=["--", "--", "--", "--", "--"])


    plt.savefig(title + "_f1.pdf")
    plt.close()



f_k_mers = open(os.path.expanduser("./data/k_mers.csv"))
f_sequences = open(os.path.expanduser("./data/sequences.csv"))
taxa_index_dict = pickle.load(open(os.path.expanduser("./data/taxa_index_dict.p")))

X_k_mers, X_sequences, Y = prepare_data(f_k_mers, f_sequences, taxa_index_dict)

f_k_mers.close()
f_sequences.close()

chars = ['a', 'c', 'g', 't']
keywords = [''.join(i) for i in product(chars, repeat=5)]

X_k_mers_500, X_sequences_500 = create_500_data(X_sequences, keywords)

X_sequences = map_sequences(X_sequences, keywords)
X_sequences_500 = map_sequences(X_sequences_500, keywords)

X_sequences = sequence.pad_sequences(X_sequences)
X_sequences_500 = sequence.pad_sequences(X_sequences_500, maxlen=X_sequences.shape[1])

""" shuffle data """
rng_state = np.random.get_state()
np.random.shuffle(X_k_mers)

np.random.set_state(rng_state)
np.random.shuffle(X_sequences)

np.random.set_state(rng_state)
np.random.shuffle(X_k_mers_500)

np.random.set_state(rng_state)
np.random.shuffle(X_sequences_500)

np.random.set_state(rng_state)
np.random.shuffle(Y)

results = {"accuracy_rf": [], "accuracy_rf_500": [], "f1_rf": [], "f1_rf_500": [], "accuracy_cnn": [],
           "accuracy_cnn_500": [], "f1_cnn": [], "f1_cnn_500": [], "accuracy_rnn": [], "accuracy_rnn_500": [],
           "f1_rnn": [], "f1_rnn_500": [], "accuracy_bidirectional_rnn": [], "accuracy_bidirectional_rnn_500": [],
           "f1_bidirectional_rnn": [], "f1_bidirectional_rnn_500": [], "accuracy_rnn_cnn": [],
           "accuracy_rnn_cnn_500": [],
           "f1_rnn_cnn": [], "f1_rnn_cnn_500": []}


""" 10 - fold cross validation"""

kf = KFold(X_k_mers.shape[0], n_folds=2)

for train, test in kf:

    X_train_k_mers = X_k_mers[train]
    X_test_k_mers = X_k_mers[test]
    X_test_k_mers_500 = X_k_mers_500[test]

    X_train_sequences = X_sequences[train]
    X_test_sequences = X_sequences[test]
    X_test_sequences_500 = X_sequences_500[test]

    accuracy_rf = []
    accuracy_cnn = []
    accuracy_rnn = []
    accuracy_bidirectional_rnn = []
    accuracy_cnn_rnn = []
    accuracy_rf_500 = []
    accuracy_cnn_500 = []
    accuracy_rnn_500 = []
    accuracy_bidirectional_rnn_500 = []
    accuracy_cnn_rnn_500 = []
    f1_rf = []
    f1_cnn = []
    f1_rnn = []
    f1_bidirectional_rnn = []
    f1_cnn_rnn = []
    f1_rf_500 = []
    f1_cnn_500 = []
    f1_rnn_500 = []
    f1_bidirectional_rnn_500 = []
    f1_cnn_rnn_500 = []

    # predict each taxonomic class separately
    for taxa in ['p_', 'c_', 'o_', 'f_', 'g_']:
        taxonomic_ranks = [k for k in taxa_index_dict if k.startswith(taxa)]
        indexes = [taxa_index_dict[rank] for rank in taxonomic_ranks]

        Y_train = Y[train][:,indexes]
        Y_test = Y[test][:,indexes]


        """random forest prediction"""
        print "random forest prediction"
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train_k_mers, Y_train)

        Y_pred = rf.predict(X_test_k_mers)
        accuracy, f1 = evaluate_model(Y_test, Y_pred)
        accuracy_rf.append(accuracy)
        f1_rf.append(f1)

        Y_pred_500 = rf.predict(X_test_k_mers_500)
        accuracy, f1 = evaluate_model(Y_test, Y_pred_500)
        accuracy_rf_500.append(accuracy)
        f1_rf_500.append(f1)


        """cnn prediction"""
        print "cnn prediction"
        model_cnn = Sequential()
        model_cnn.add(Convolution1D(10, 5, border_mode='same', input_shape = (1024,1)))
        model_cnn.add(Activation('relu'))
        model_cnn.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))

        model_cnn.add(Convolution1D(20, 5, border_mode='same'))
        model_cnn.add(Activation('relu'))
        model_cnn.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))

        model_cnn.add(Flatten())

        model_cnn.add(Dense(len(taxonomic_ranks)))
        model_cnn.add(Activation('softmax'))

        model_cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model_cnn.fit(X_train_k_mers.reshape(X_train_k_mers.shape + (1,)), Y_train, nb_epoch=2, batch_size=128)

        Y_pred = model_cnn.predict(X_test_k_mers.reshape(X_test_k_mers.shape + (1,)), batch_size=128)
        accuracy, f1 = evaluate_model(Y_test, Y_pred)
        accuracy_cnn.append(accuracy)
        f1_cnn.append(f1)

        Y_pred_500 = model_cnn.predict(X_test_k_mers_500.reshape(X_test_k_mers_500.shape + (1,)), batch_size=64)
        accuracy, f1 = evaluate_model(Y_test, Y_pred_500)
        accuracy_cnn_500.append(accuracy)
        f1_cnn_500.append(f1)


        """rnn prediction"""
        print "rnn prediction"

        X_validation = X_train_sequences[0:100]
        Y_validation = Y_train[0:100]
        X_train_sequences_train = X_train_sequences[100:-1]
        Y_train_sequences = Y_train[100:-1]

        model_rnn = Sequential()
        model_rnn.add(Embedding(len(keywords)+1, 100, input_length=X_sequences.shape[1]))
        model_rnn.add(GRU(input_shape=(X_sequences.shape[1], 1), output_dim=1))
        model_rnn.add(Dense(len(taxonomic_ranks)))
        model_rnn.add(Activation('softmax'))

        model_rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        max_acc = 0
        iterations_after_max = 0
        best_weights = ""

        for i in range(1):
            print Y_train_sequences.shape
            h = model_rnn.fit(X_train_sequences_train, Y_train_sequences, nb_epoch=1, batch_size=64)
            val_pred = model_rnn.predict(X_validation, batch_size=32)
            acc, f1 = evaluate_model(Y_validation, val_pred)

            if acc >= max_acc:
                max_acc = acc
                iterations_after_max = 0
                best_weights = model_rnn.get_weights()

            iterations_after_max += 1
            if iterations_after_max > 15 or max_acc >= 0.99:
                break

        model_rnn.set_weights(best_weights)

        Y_pred = model_rnn.predict(X_test_sequences, batch_size=32)
        accuracy, f1 = evaluate_model(Y_test, Y_pred)
        accuracy_rnn.append(accuracy)
        f1_rnn.append(f1)

        Y_pred_500 = model_rnn.predict(X_test_sequences_500, batch_size=32)
        accuracy, f1 = evaluate_model(Y_test, Y_pred_500)
        accuracy_rnn_500.append(accuracy)
        f1_rnn_500.append(f1)


        """bidirectional rnn prediction"""
        print "bidirectional rnn prediction"

        model_bidirectional_rnn = Sequential()

        forwards = Sequential()
        forwards.add(Embedding(len(keywords)+1, 50, input_length=X_sequences.shape[1]))
        forwards.add(GRU(input_shape=(X_sequences.shape[1], 1), output_dim=1))

        backwards = Sequential()
        backwards.add(Embedding(len(keywords)+1, 50, input_length=X_sequences.shape[1]))
        backwards.add(GRU(input_shape=(X_sequences.shape[1], 1), output_dim=1, go_backwards=True))

        model_bidirectional_rnn.add((Merge([forwards, backwards], mode='concat')))
        model_bidirectional_rnn.add(Dense(len(taxonomic_ranks)))
        model_bidirectional_rnn.add(Activation('softmax'))

        model_bidirectional_rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        max_acc = 0
        iterations_after_max = 0
        best_weights = ""

        for i in range(1):
            print Y_train_sequences.shape
            h = model_bidirectional_rnn.fit([X_train_sequences_train, X_train_sequences_train],
                              Y_train_sequences, nb_epoch=1, batch_size=64)
            val_pred = model_bidirectional_rnn.predict([X_validation, X_validation], batch_size=32)
            acc, f1 = evaluate_model(Y_validation, val_pred)

            if acc >= max_acc:
                max_acc = acc
                iterations_after_max = 0
                best_weights = model_bidirectional_rnn.get_weights()

            iterations_after_max += 1
            if iterations_after_max > 15 or max_acc >= 0.99:
                break

        model_bidirectional_rnn.set_weights(best_weights)

        Y_pred = model_bidirectional_rnn.predict([X_test_sequences, X_test_sequences], batch_size=32)
        accuracy, f1 = evaluate_model(Y_test, Y_pred)
        accuracy_bidirectional_rnn.append(accuracy)
        f1_bidirectional_rnn.append(f1)

        Y_pred_500 = model_bidirectional_rnn.predict([X_test_sequences_500, X_test_sequences_500], batch_size=32)
        accuracy, f1 = evaluate_model(Y_test, Y_pred_500)
        accuracy_bidirectional_rnn_500.append(accuracy)
        f1_bidirectional_rnn_500.append(f1)


        """cnn + rnn prediction"""
        print "cnn + rnn prediction"

        model_rnn_cnn = Sequential()
        model_rnn_cnn.add(Embedding(len(keywords) + 1, 100, input_length=X_sequences.shape[1]))

        model_rnn_cnn.add(Convolution1D(10, 5, border_mode='same'))
        model_rnn_cnn.add(Activation('relu'))
        model_rnn_cnn.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))

        model_rnn_cnn.add(Convolution1D(20, 5, border_mode='same'))
        model_rnn_cnn.add(Activation('relu'))
        model_rnn_cnn.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))

        model_rnn_cnn.add(GRU(output_dim=3))

        model_rnn_cnn.add(Dense(len(taxonomic_ranks)))
        model_rnn_cnn.add(Activation('softmax'))

        model_rnn_cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        max_acc = 0
        iterations_after_max = 0
        best_weights = ""

        for i in range(1):
            print Y_train_sequences.shape
            h = model_rnn_cnn.fit(X_train_sequences_train, Y_train_sequences, nb_epoch=1, batch_size=64)
            val_pred = model_rnn_cnn.predict(X_validation, batch_size=32)
            acc, f1 = evaluate_model(Y_validation, val_pred)

            if acc >= max_acc:
                max_acc = acc
                iterations_after_max = 0
                best_weights = model_rnn_cnn.get_weights()

            iterations_after_max += 1
            if iterations_after_max > 15 or max_acc >= 0.99:
                break

        model_rnn_cnn.set_weights(best_weights)

        Y_pred = model_rnn_cnn.predict(X_test_sequences, batch_size=32)
        accuracy, f1 = evaluate_model(Y_test, Y_pred)
        accuracy_cnn_rnn.append(accuracy)
        f1_cnn_rnn.append(f1)

        Y_pred_500 = model_rnn_cnn.predict(X_test_sequences_500, batch_size=32)
        accuracy, f1 = evaluate_model(Y_test, Y_pred_500)
        accuracy_cnn_rnn_500.append(accuracy)
        f1_cnn_rnn_500.append(f1)


    results["accuracy_cnn"].append(accuracy_cnn)
    results["accuracy_rf"].append(accuracy_rf)
    results["accuracy_rnn"].append(accuracy_rnn)
    results["accuracy_bidirectional_rnn"].append(accuracy_bidirectional_rnn)
    results["accuracy_rnn_cnn"].append(accuracy_cnn_rnn)
    results["f1_cnn"].append(f1_cnn)
    results["f1_rf"].append(f1_rf)
    results["f1_rnn"].append(f1_rnn)
    results["f1_bidirectional_rnn"].append(f1_bidirectional_rnn)
    results["f1_rnn_cnn"].append(f1_cnn_rnn)
    results["accuracy_cnn_500"].append(accuracy_cnn_500)
    results["accuracy_rf_500"].append(accuracy_rf_500)
    results["accuracy_rnn_500"].append(accuracy_rnn_500)
    results["accuracy_bidirectional_rnn_500"].append(accuracy_bidirectional_rnn_500)
    results["accuracy_rnn_cnn_500"].append(accuracy_cnn_rnn_500)
    results["f1_cnn_500"].append(f1_cnn_500)
    results["f1_rf_500"].append(f1_rf_500)
    results["f1_rnn_500"].append(f1_rnn_500)
    results["f1_bidirectional_rnn_500"].append(f1_bidirectional_rnn_500)
    results["f1_rnn_cnn_500"].append(f1_cnn_rnn_500)


for k in results:
    results[k] = np.array(results[k])
    print k, results[k].mean(axis=0)

vizualization_data = []
vizualization_data = put_to_vizualization_data(vizualization_data, results["accuracy_cnn"].mean(axis=0), results["f1_cnn"].mean(axis=0), "cnn")
vizualization_data = put_to_vizualization_data(vizualization_data, results["accuracy_rnn"].mean(axis=0), results["f1_rnn"].mean(axis=0), "rnn")
vizualization_data = put_to_vizualization_data(vizualization_data, results["accuracy_rf"].mean(axis=0), results["f1_rf"].mean(axis=0), "rf")
vizualization_data = put_to_vizualization_data(vizualization_data, results["accuracy_bidirectional_rnn"].mean(axis=0), results["f1_bidirectional_rnn"].mean(axis=0), "b_rnn")
vizualization_data = put_to_vizualization_data(vizualization_data, results["accuracy_rnn_cnn"].mean(axis=0), results["f1_rnn_cnn"].mean(axis=0), "cnn+rnn")

vizualization_data_500 = []
vizualization_data_500 = put_to_vizualization_data(vizualization_data_500, results["accuracy_cnn_500"].mean(axis=0), results["f1_cnn_500"].mean(axis=0), "cnn")
vizualization_data_500 = put_to_vizualization_data(vizualization_data_500, results["accuracy_rnn_500"].mean(axis=0), results["f1_rnn_500"].mean(axis=0), "rnn")
vizualization_data_500 = put_to_vizualization_data(vizualization_data_500, results["accuracy_rf_500"].mean(axis=0), results["f1_rf_500"].mean(axis=0), "rf")
vizualization_data_500 = put_to_vizualization_data(vizualization_data_500, results["accuracy_bidirectional_rnn_500"].mean(axis=0), results["f1_bidirectional_rnn_500"].mean(axis=0), "b_rnn")
vizualization_data_500 = put_to_vizualization_data(vizualization_data_500, results["accuracy_rnn_cnn_500"].mean(axis=0), results["f1_rnn_cnn_500"].mean(axis=0), "cnn+rnn")

make_a_plot(vizualization_data, "results")
make_a_plot(vizualization_data_500, "results_500")
