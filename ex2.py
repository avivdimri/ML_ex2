import numpy as np
import sys

label = 3
features = 5


def takeFirst(elem):
    return elem[0]


def normal(data):
    avg = np.mean(data, axis=0)
    dev = np.std(data, axis=0)
    for i in range(len(data)):
        data[i] = (data[i] - avg) / dev


def knn_model(k, x_arr, y_arr, test):
    knn = []
    for i, x in enumerate(x_arr):
        dis = np.linalg.norm(test - x)
        knn.append((dis, y_arr[i]))
    knn.sort(key=takeFirst)
    knn = knn[0:k]
    counter = [0 for _ in range(label)]
    for lab in knn:
        counter[lab[1]] += 1
    max_index = np.argmax(counter)
    return max_index


def per_train(n, T, x, y):
    w = np.zeros((label, features))
    for t in range(T):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        for i in range(len(x)):
            y_hat = np.argmax(np.dot(w, x[i]))
            if y_hat != y[i]:
                w[y[i]] += n * x[i]
                w[y_hat] -= n * x[i]
    return w


def svm_train(x, y):
    lamda = 0.0001
    n = 0.001
    lam_n = lamda * n
    w = np.zeros((label, features))
    for t in range(82):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        for i in range(len(x)):
            y_r = np.argmax(np.dot(w, x[i]))
            if y_r == y[i]:
                tmp = np.dot(w, x[i])
                tmp[y[i]]= -np.inf
                y_r = np.argmax(tmp)
            loss_func = max(0, 1 - np.dot(w[y[i]], x[i]) + np.dot(w[y_r], x[i]))
            if loss_func > 0:
                w[y[i]] = np.dot(w[y[i]], (1 - lam_n)) + np.dot(n, x[i])
                w[y_r] = np.dot(w[y_r], (1 - lam_n)) - np.dot(n, x[i])
                for j in range(label):
                    if j != y_r and j != y[i]:
                        w[j] *= (1 - lam_n)
            else:
                for j in range(label):
                        w[j] *= (1 - lam_n)
    return w


def pa_train(x, y):
    w = np.zeros((label, features))
    for t in range(119):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        for i in range(len(x)):
            y_r = np.argmax(np.dot(w, x[i]))
            if y_r == y[i]:
                tmp = np.dot(w, x[i])
                tmp[y[i]]= -np.inf
                y_r = np.argmax(tmp)
            if y_r != y[i]:
                loss_func = max(0, 1 - np.dot(w[y[i]], x[i]) + np.dot(w[y_r], x[i]))
                tau = loss_func / (2 * (np.linalg.norm(x[i]) ** 2))
                w[y[i]] += np.dot(tau, x[i])
                w[y_r] -= np.dot(tau, x[i])
    return w


train_x, train_y, test_x, out_fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

x_arr = np.loadtxt(train_x, delimiter=',')
normal(x_arr)

new_column = [1 for _ in range(len(x_arr))]
x_arr = np.insert(x_arr, 0, new_column, axis=1)
x_arr = np.delete(x_arr, 5, 1)

y_arr = np.loadtxt(train_y).astype(int)

test_arr = np.loadtxt(test_x, delimiter=',')
normal(test_arr)

new_column = [1 for _ in range(len(test_arr))]
test_arr = np.insert(test_arr, 0, new_column, axis=1)
test_arr = np.delete(test_arr, 5, 1)

f = open(out_fname, "w+")
w_per = per_train(0.83, 102, x_arr.copy(), y_arr.copy())
w_svm = svm_train(x_arr.copy(), y_arr.copy())
w_pa = pa_train(x_arr.copy(), y_arr.copy())
for i in range(len(test_arr)):
    f.write(f"knn: {knn_model(3, x_arr, y_arr, test_arr[i])}, perceptron: {np.argmax(np.dot(w_per, test_arr[i]))}, svm: {np.argmax(np.dot(w_svm, test_arr[i]))}, pa: {np.argmax(np.dot(w_pa, test_arr[i]))}\n")
