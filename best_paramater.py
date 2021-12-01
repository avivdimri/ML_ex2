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




def svm_train(lamda, n, T, x, y):
    lam_n = lamda * n
    w = np.zeros((label, features))
    for t in range(T):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        for i in range(len(x)):
            y_r = np.argmax(np.dot(w, x[i]))
            if y_r == y[i]:
                tmp = np.dot(w, x[i])
                tmp[y[i]] = -np.inf
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


def parms_svm(x_train, y_train):
    per_valid = 10
    len_valid = int(len(x_train) / per_valid)
    max_ep=1
    max3_success=0
    max_3n=1
    max_lamda3=1
    for epoch in range(40,200):
        max_lamda2 = 1
        max2_success = 0
        max_n = 1
        n = 1
        while n > 0.001:
            max_lamda = 1
            lamda = 0.01
            max_success = 0
            while lamda >= 0.0000001:
                success = 0
                i = 0
                while i < per_valid:
                    begin = i * len_valid
                    end = begin + len_valid - 1
                    tmp_x = x_train.copy()
                    tmp_y = y_train.copy()
                    validation_x = tmp_x[begin:end]
                    validation_y = tmp_y[begin:end]
                    for j in range(len_valid):
                        tmp_x = np.delete(tmp_x, begin, axis=0)
                        tmp_y = np.delete(tmp_y, begin, axis=0)
                    w = svm_train(lamda, n, epoch, tmp_x, tmp_y)
                    for x, y in zip(validation_x, validation_y):
                        y_hat = np.argmax(np.dot(w, x))
                        if y == y_hat:
                            success += 1
                    i += 1
                success /= per_valid
                if success > max_success:
                    max_success = success
                    max_lamda = lamda
                lamda /= 10
            if max_success > max2_success:
                max2_success = max_success
                max_lamda2 = max_lamda
                max_n = n
            n /= 10
        if max2_success > max3_success:
            print(f"11111")
            max3_success = max2_success
            max_3n = max_n
            max_lamda3 = max_lamda2
            max_ep = epoch
        print(f" the percentage success is : {(max2_success / len(validation_x)) * 100}% with n = {max_n} for lamda = {max_lamda2} for epoch = {epoch} ")

    print(f" the best best percentage success is : {(max3_success / len(validation_x)) * 100}% with n = {max_3n} for lamda = {max_lamda3}for epoch = {max_ep} ")
    return max_lamda3, max_3n,max_ep



def parms_percepteron(x_train, y_train):
    per_valid = 10
    len_valid = int(len(x_train) / per_valid)
    max_n2 = 1
    max_epoch = 1
    max2_success = 0
    for epoch in range(1, 200):
        max_n = 1
        n = 0.01
        max_success = 0
        while n < 1:
            success = 0
            i = 0
            while i < per_valid:
                begin = i * len_valid
                end = begin + len_valid - 1
                tmp_x = x_train.copy()
                tmp_y = y_train.copy()
                validation_x = tmp_x[begin:end]
                validation_y = tmp_y[begin:end]
                for j in range(len_valid):
                    tmp_x = np.delete(tmp_x, begin, axis=0)
                    tmp_y = np.delete(tmp_y, begin, axis=0)
                w = per_train(n, epoch, tmp_x, tmp_y)
                for x, y in zip(validation_x, validation_y):
                    y_hat = np.argmax(np.dot(w, x))
                    if y == y_hat:
                        success += 1
                i += 1
            success /= per_valid
            if success > max_success:
                max_success = success
                max_n = n
            n += 0.1
        if max_success > max2_success:
            max2_success = max_success
            max_n2 = max_n
            max_epoch = epoch
        print(
            f" the percentage success is : {(max_success / len(validation_x)) * 100}% with n = {max_n} for epoch = {epoch} ")
    print(
        f" the best percentage success is : {(max2_success / len(validation_x)) * 100}% with n = {max_n2} for epoch = {max_epoch} ")
    return max_epoch, max_n2


def k_parm_knn(x_train, y_train):
    per_valid = 10
    len_valid = int(len(x_train) / per_valid)
    max_success = 0
    max_k = 1
    for k in range(1, len(x_train)):
        i = 0
        success = 0
        while i < per_valid:
            begin = i * len_valid
            end = begin + len_valid - 1
            tmp_x = x_train.copy()
            tmp_y = y_train.copy()
            validation_x = tmp_x[begin:end]
            validation_y = tmp_y[begin:end]
            for j in range(len_valid):
                tmp_x = np.delete(tmp_x, begin, axis=0)
                tmp_y = np.delete(tmp_y, begin, axis=0)
            for x, y in zip(validation_x, validation_y):
                y_hat = knn_model(k, tmp_x, tmp_y, x)
                if y == y_hat:
                    success += 1
            i += 1
        success /= per_valid
        if success > max_success:
            max_success = success
            max_k = k
        print(f" the percentage success is : {(success / len(validation_x)) * 100}% for k = {k} ")

    print(f" the best percentage success is : {(max_success / len(validation_x)) * 100}% for k = {max_k} ")
    return max_k

def pa_train(T,x, y):
    w = np.zeros((label, features))
    for t in range(T):
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

def epoch_parm_pa(x_train, y_train):
    per_valid = 10
    len_valid = int(len(x_train) / per_valid)
    max_epoch = 1
    max_success = 0
    for epoch in range(1, 200):
        success = 0
        i = 0
        while i < per_valid:
            begin = i * len_valid
            end = begin + len_valid - 1
            tmp_x = x_train.copy()
            tmp_y = y_train.copy()
            validation_x = tmp_x[begin:end]
            validation_y = tmp_y[begin:end]
            for j in range(len_valid):
                tmp_x = np.delete(tmp_x, begin, axis=0)
                tmp_y = np.delete(tmp_y, begin, axis=0)
            w = pa_train(epoch, tmp_x, tmp_y)
            for x, y in zip(validation_x, validation_y):
                y_hat = np.argmax(np.dot(w, x))
                if y == y_hat:
                    success += 1
            i += 1
        success /= per_valid
        if success > max_success:
            max_success = success
            max_epoch = epoch
        print(f" the percentage success is : {(success / len(validation_x)) * 100}% with for epoch = {epoch} ")
    print(f" the best percentage success is : {(max_success / len(validation_x)) * 100}% for epoch = {max_epoch} ")
    return max_epoch

train_x, train_y, test_x, out_fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

x_arr = np.loadtxt(train_x, delimiter=',')
normal(x_arr)

new_column = [1 for _ in range(len(x_arr))]
x_arr = np.insert(x_arr, 0, new_column, axis=1)
x_arr = np.delete(x_arr, 5, 1)

y_arr = np.loadtxt(train_y).astype(int)
#epoch_parm_pa(x_arr,y_arr)
#parms_percepteron(x_arr,y_arr)
parms_svm(x_arr, y_arr)
# k_parm_knn(x_arr,y_arr)
test_arr = np.loadtxt(test_x, delimiter=',')
# test_arr = test_arr.reshape(-1, 5)
normal(test_arr)
