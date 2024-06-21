import pandas as pd
import numpy as np


class NaiveB():
    
    global X_prob
    global lprobs
    
    def __init__(self, train_df, label):
        self.X_train, self.y_train = self.Starget(df=train_df, label=label)
        self.label = label

        self.X_prob, self.y_prob = self.probability()
        
    
    def Starget(self, df, label):
        X = df.drop(columns=[label], axis = 1)
        y = df[label]
        return X, y
        
        
    def probability(self):
        X_prob = {}

        T_col = self.y_train.value_counts().to_frame()
        T_col.reset_index(inplace=True)
        T_col.columns = ['class', 'count']

        T_v = {i : j for (i, j) in zip(T_col['class'], T_col['count'])}
        T_key = list(T_v.keys())

        for i in self.X_train:
            column_dictionary = {}
            a = self.X_train[i].value_counts().to_frame()
            X_values = a.index.to_list()
            ydata = pd.DataFrame(data={i : self.X_train[i], 'y' : self.y_train})

            for j in X_values:
                X_dictionary = {}
                exdata = ydata[ydata[i] == j]

                for k in T_key:
                    ydataf = exdata[exdata['y'] == k]
                    X_dictionary[k] = len(ydataf) / T_v[k]

                column_dictionary[j] = X_dictionary
            X_prob[i] = column_dictionary
        y_prob = {a : b / sum(list(T_v.values())) for (a, b) in T_v.items()}

        return X_prob, y_prob
    
    
    
    
    def predictor(self, X_new):
        cols = list(self.X_prob.keys())
        col_new = {i : j for (i, j) in zip(cols, X_new)}

        lprobs = {}
        for l, v in self.y_prob.items():
            cate_v = [self.X_prob[cn][cl][l] for (cn, cl) in col_new.items()]
            lprobs[l] = round((np.prod(cate_v) * v), 4)

        prob_ks = list(lprobs.keys())
        prob_vs = list(lprobs.values())

        return prob_ks[np.argmax(prob_vs)]
    
    
    
    
    def predict(self, test_df):
        X_test, y_test = self.Starget(df=test_df, label=self.label)
        X_test_vals = X_test.values
        y_test_vals = y_test.values
        if len(X_test_vals) == 1:
            return self.predictor(X_new=X_test_vals[0])
        preds = [self.predictor(X_new=i) for i in X_test_vals]
        actual_vals = np.array(y_test_vals)
        preds = np.array(preds)
        corrects = np.count_nonzero(np.where((actual_vals == preds), 1, 0))
        return corrects / len(actual_vals)

    def print_confusion_matrix(self, test_df):
        X_test, y_test = self.Starget(df=test_df, label=self.label)
        X_test_vals = X_test.values
        y_test_vals = y_test.values
        if len(X_test_vals) == 1:
            return self.predictor(X_new=X_test_vals[0])
        preds = [self.predictor(X_new=i) for i in X_test_vals]
        actual_vals = np.array(y_test_vals)
        dists = np.unique(actual_vals).tolist()
        confusion_matrix = [[0] * len(dists) for i in range(len(dists))]
        conf_map = {}
        for i in range(len(dists)):
            conf_map[dists[i]] = i
        for i, j in zip(actual_vals, preds):
            confusion_matrix[conf_map[i]][conf_map[j]] +=1
        conf_matrix = pd.DataFrame(confusion_matrix, dists, dists)
        print("Confusion Matrix")
        print(conf_matrix)

        preds = np.array(preds)
        corrects = np.count_nonzero(np.where((actual_vals == preds), 1, 0))
        return corrects / len(actual_vals)


def compute_train_data(folds, i) :
    if (i == 0):
        train = folds[1]
        for j in range(2, len(folds)):
            train = pd.concat([train, folds[j]])
    else :
        train = folds[0]
        for j in range(1,len(folds)):
            if not i ==j :
                train = pd.concat([train, folds[j]])
    return train

def kfold():
    columns = []
    print("Enter meta file")
    meta = input()
    with open(meta , 'r') as file:
        data = file.read()
        for column in data.split("\n"):
            if len(column) > 1:
                columns.append(column.split(":")[0])
        print(columns)
    print("Enter data filename")
    data_file = input()
    with open(data_file , 'r') as file:
        df = pd.read_csv(file,names = columns)
    print(df.head())

    print("Enter value of k")
    k = int(input())

    df.sample(frac=1)

    folds = np.array_split(df, k)

    avg = 0
    for i in range(k):
        train_data = compute_train_data(folds, i)
        nb = NaiveB(train_df=train_data, label='class')
        accuracy = nb.predict(test_df= folds[i])
        print("accuracy for fold ",i+1," is ",accuracy*100)
        avg += accuracy*100

    print("Avg accuracy is ", avg/k)

def confusion_matrix():
    columns = []
    print("Enter meta file")
    meta = input()
    with open(meta , 'r') as file:
        data = file.read()
        for column in data.split("\n"):
            if len(column) > 1:
                columns.append(column.split(":")[0])
        print(columns)
    print("Enter training filename")
    train_file = input()
    with open(train_file , 'r') as file:
        df = pd.read_csv(file,names = columns)
    print(df.head())
    nb = NaiveB(train_df=df, label='class')

    print("Enter test filename")
    test_file = input()
    with open(test_file , 'r') as file:
        test_df = pd.read_csv(file,names = columns)
    accuracy = nb.print_confusion_matrix(test_df= test_df)
    

def naive():
    columns = []
    while True:
        print("Enter 1 to train")
        print("Enter 2 to k-fold cross validation")
        print("Enter 3 to test accuracy")
        print("Enter 4 to confusion matrix")
        print("Enter 5 to exit")

        choice = int(input())
        print("You entered "+ str(choice))
        
        if choice == 1 :
            print("Enter meta filename")
            meta = input()
            with open(meta , 'r') as file:
                data = file.read()
                for column in data.split("\n"):
                    if len(column) > 1:
                        columns.append(column.split(":")[0])
                print(columns)
            print("Enter training filename")
            train_file = input()
            with open(train_file , 'r') as file:
                df = pd.read_csv(file,names = columns)
            print(df.head())
            nb = NaiveB(train_df=df, label='class')
            # train logic
        elif choice == 2:
            kfold()
        elif choice == 3:
            # accuracy
            print("Enter test filename")
            test_file = input()
            with open(test_file , 'r') as file:
                test_df = pd.read_csv(file,names = columns)
            print(test_df.head())
            accuracy = nb.predict(test_df= test_df)
            acc = accuracy*100
            print("accuracy " + str(acc))
        elif choice == 4:
            print("Enter test filename")
            test_file = input()
            with open(test_file , 'r') as file:
                test_df = pd.read_csv(file,names = columns)
            print("Accuracy is : ", nb.print_confusion_matrix(test_df=test_df)*100)
        else :
            break

if __name__ == "__main__":
    naive()

