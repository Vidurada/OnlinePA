import numpy as np
import csv

# Classic Method
def classic(lt, xt):
    return lt / xt


# First Relaxation Method
def first(C, lt, xt):
    return min(C, (lt / xt))


# Second Relaxation Method
def second(C, lt, xt):
    return lt / (xt + (1 / (2 * C)))


# input
train_data = np.loadtxt("train_data.csv", dtype='i', delimiter=',')
train_labels = np.loadtxt("train_labels.csv", dtype='i', delimiter=',')
test_data = np.loadtxt("test_data.csv", dtype='i', delimiter=',')
test_labels = np.loadtxt("test_labels.csv", dtype='i', delimiter=',')

iterations = 10
C = 1
update_option = 'classic'

# PA algoritm
# initiate
w = np.zeros(9)

for i in range(iterations):
    print("")
    print("==============================================================================")
    print("Iterations = ",iterations," ,C = ",C, " ,Update Option= ",update_option)
    print("")
    for index in range(len(train_data)):
        # receive the x instance at time t
        xt = train_data[index, :]
        #print(xt)
        # predict
        y = train_labels[index] * np.dot(w, xt)
        #print(ys)
        if y < 1:
            # loss calculation
            lt = max(0, (1 - y))
            xt_square = np.sum(xt ** 2)

            # select the updating option
            if update_option == 'classic':
                tt = classic(lt, xt_square)
            elif update_option == 'first':
                tt = first(C, lt, xt_square)
            else:
                tt = second(C, lt, xt_square)

            # update weights
            w = w + tt * train_labels[index] * xt
    print("w = ", w)
    print("")
    # testing
    for r in range(len(test_data)):
        # receive the x instance at time t
        xtt = test_data[r, :]
        # the predictions
        prediction = np.sign(np.dot(w.T, test_data.T))

    print ("Prediction Matrix= ",prediction)
    c = np.count_nonzero(prediction - test_labels)
    print("Total rows of data: ", len(test_data))

    correct_predictions = len(test_data) - c

    print("Correct Predictions: ", correct_predictions)

    percentage = (correct_predictions/len(test_data))*100

    print("Correct Percentage: ", percentage, "%" )




