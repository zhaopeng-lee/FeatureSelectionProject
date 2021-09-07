from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import pandas as pd
import numpy as np
import random
class PerformanceHandler :
    """this class is aiming to handle the performance problem"""
    #******
    #this is a static dimension sampling function, take original data as input
    #output a sequence of n dimensional feature set and its associated CV mean value
    #******
    def Sampling_nd_featuresets(
        dimention_of_subset:'int',
        trainning_set,
        class_set,
        mod,
        number_of_iter,
        N_classes,
        return_X_y:'bool'=True,
        return_classes:'bool'=True):
        """this function can take a X and y as input, train it with mod, test the model with
        cross validation, and return a dataframe with all feature num and its associated CV values.
        the feature dimention is static to 'dimension_of_subset'"""
        feature_n_combination = []#list to store all combination
        list_feature_num =[]#list to store all features' num
        list_features_CVmeans = []#list to return with all feature num and its associated CV mean value
        dimention_of_feature = len(trainning_set.columns)
        #******
        # data preprocessing
        #******
        list_feature_selected = input("please input the features' number \
            that are relavant to the classes")
        print(list_feature_selected)
        if list_feature_selected != '':
            list_feature_selected = list_feature_selected.split(',')
            list_feature_selected = [int(i) for i in list_feature_selected]
        else:
            list_feature_selected = []
        list_feature_dropped = input("please input the features' number\
            that are irrelavant")
        if list_feature_dropped != '':
            list_feature_dropped = list_feature_dropped.split(',')
            list_feature_dropped = [int(i) for i in list_feature_dropped]
        else:
            list_feature_dropped = []
        #******
        for i in range(dimention_of_feature):
            list_feature_num.append(i)
        list_feature_num = [i for i in list_feature_num if i not in list_feature_selected and i not in list_feature_dropped]
        feature_n_combination_tuple = list(combinations(list_feature_num,dimention_of_subset-len(list_feature_selected)))
        for list_features in feature_n_combination_tuple:
            feature_n_combination.append(list(list_features))
        for lists in feature_n_combination:
            for i in list_feature_selected:
                lists.append(i)
        #******
        # train every subsets and calculate CV mean value
        #******
        y_Iter = class_set
        clf = mod
        for i in range(number_of_iter):
            randomnum = random.randint(0,len(feature_n_combination)-1)
            try:
                X_Iterate = trainning_set.iloc[:,feature_n_combination[randomnum]]
                scores = cross_val_score(clf, X_Iterate, y_Iter, cv=5,n_jobs=-1)
                list_tem = feature_n_combination[randomnum]
                list_tem.append(scores.mean())
                list_features_CVmeans.append(list_tem)
            except:
                print("Error erupted")
        list_frame = pd.DataFrame(list_features_CVmeans)
        #******
        # if needed generate accuracy class
        #******
        list_accuracy_class = []
        for i in range(len(list_features_CVmeans)):
            accuracy = list_features_CVmeans[i][-1]
            list_accuracy_class.append(PerformanceHandler.Change_float_into_classes(accuracy,1,N_classes))
        list_frame[dimention_of_subset+1] = list_accuracy_class
        if return_X_y and return_classes:
            X_iter = list_frame.iloc[:,0:dimention_of_subset]
            y_iter = list_frame.iloc[:,dimention_of_subset+1]
            return X_iter,y_iter
        elif return_X_y == True and return_classes == False:
            X_iter = list_frame.iloc[:,0:dimention_of_subset]
            y_iter = list_frame.iloc[:,dimention_of_subset]
            return X_iter,y_iter
        else:
            return list_frame
    #******
    #this function can change a dataframe type feature num set to onehot representation
    #******
    def Featurenum_to_Onehot (X_features,original_feature_num:'int'):
        """this function can encode the numTypefeature dataframe to a onehotType dataframe"""
        list_final = []
        for i in range(len(X_features)):
            list_tem = [0 for i in range(original_feature_num)]
            for num in X_features.iloc[i,:]:
                list_tem[num]=1
            list_final.append(list_tem)
        return pd.DataFrame(list_final)
    #******
    #this is a random dimension sampling function, take original data as input
    #output a sequence of random dimensional feature set and its associated CV mean value
    #******
    def Sampling_randomd_featuresets(
            trainning_set,
            class_set,
            mod,
            number_of_iter,
            N_classes,
            return_X_y:'bool'=True,
            return_classes:'bool'=True):
            """this function can take a X and y as input, train it with mod, test the model with
            cross validation, and return a dataframe with all feature num and its associated CV values.
            the feature dimention is randomly selected,but always contain the selected features"""
            feature_n_combination = []#list to store all combination
            feature_n_combination_onehot = []#list to store all combination in onehot
            list_feature_num =[]#list to store all features' num
            list_features_CVmeans = []#list to return with all feature num and its associated CV mean value
            dimension_of_feature = len(trainning_set.columns)
            #******
            # data preprocessing
            #******
            list_feature_selected = input("please input the features' number \
                that are relavant to the classes")
            if list_feature_selected != '':
                list_feature_selected = list_feature_selected.split(',')
                list_feature_selected = [int(i) for i in list_feature_selected]
            else:
                list_feature_selected = []
            list_feature_dropped = input("please input the features' number\
                that are irrelavant")
            if list_feature_dropped != '':
                list_feature_dropped = list_feature_dropped.split(',')
                list_feature_dropped = [int(i) for i in list_feature_dropped]
            else:
                list_feature_dropped = []
            #******
            # generate the possible feature set - list_feature_num
            # generate possible combination which contain the feature selectet
            #******
            for i in range(dimension_of_feature):
                list_feature_num.append(i)
            list_feature_num = [i for i in list_feature_num if i not in list_feature_selected and i not in list_feature_dropped]
            for i in range(number_of_iter):
                if list_feature_selected != []:
                    randomdimension = random.randint(0,len(list_feature_num))
                else:
                    randomdimension = random.randint(1,len(list_feature_num))
                list_randomdim = random.sample(list_feature_num,randomdimension)
                if list_randomdim not in feature_n_combination:
                    feature_n_combination.append(list_randomdim)
                else:
                    continue
            if list_feature_selected != []:
                for lists in feature_n_combination:
                    for i in list_feature_selected:
                        lists.append(i)
            feature_array = np.array(feature_n_combination,dtype = object)
            del feature_n_combination
            #******
            #change the dataset into onehot representation
            #******
            for i in range(len(feature_array)):
                list_tem = [0 for i in range(dimension_of_feature)]
                for num in range(len(feature_array[i])):
                    list_tem[feature_array[i][num]]=1
                feature_n_combination_onehot.append(list_tem)
            #******
            # train every subsets and calculate CV mean value
            #******
            y_Iter = class_set
            clf = mod
            for i in range(len(feature_array)):
                try:
                    X_Iterate = trainning_set.iloc[:,feature_array[i]]
                    scores = cross_val_score(clf, X_Iterate, y_Iter, cv=5,n_jobs=-1)
                    list_tem = feature_n_combination_onehot[i]
                    list_tem.append(scores.mean())
                    list_features_CVmeans.append(list_tem)
                except:
                    print("Error erupted")
            list_frame = pd.DataFrame(list_features_CVmeans)
            #******
            # if needed generate accuracy class
            #******
            list_accuracy_class = []
            for i in range(len(list_features_CVmeans)):
                accuracy = list_features_CVmeans[i][-1]
                list_accuracy_class.append(PerformanceHandler.Change_float_into_classes(accuracy,1,N_classes))
    
            list_frame[dimension_of_feature+1] = list_accuracy_class
            if return_X_y and return_classes:
                X_iter = list_frame.iloc[:,0:dimension_of_feature]
                y_iter = list_frame.iloc[:,-1]
                return X_iter,y_iter
            elif return_X_y == True and return_classes == False:
                X_iter = list_frame.iloc[:,0:dimension_of_feature]
                y_iter = list_frame.iloc[:,-2]
                return X_iter,y_iter
            else:
                return list_frame

    #******
    #this function takes feature set with onehot representation, and output a pool of different onehot set
    #******
    def pool_generator (X_initial,size_of_pool:"int"):
        """this function can provide different feature subset compare to the input set,
        you should pass in onehot version of X, and the output is also in a onehot version"""
        list_X_initial = []
        dimention_of_feature = len(X_initial.columns)
        generate_feature_set = []
        list_feature_num = [i for i in range(dimention_of_feature)]
        for i in range(len(X_initial)):
            list_tem = X_initial.iloc[i,:]
            list_X_initial_each = []
            for num in range(len(list_tem)):
                if list_tem[num] == 1:
                    list_X_initial_each.append(num)
                
            list_X_initial.append(sorted(list_X_initial_each))
            #print(list_X_initial)
        
        for i in range(size_of_pool):
            randomnum = random.randint(1,dimention_of_feature)
            feature_randomdim = random.sample(list_feature_num,randomnum)
            feature_randomdim.sort()
            if feature_randomdim not in list_X_initial and feature_randomdim not in generate_feature_set:
                generate_feature_set.append(feature_randomdim)  
        list_final = []
        for i in range(len(generate_feature_set)):
            list_tem = [0 for i in range(dimention_of_feature)]
            for num in generate_feature_set[i]:
                list_tem[num]=1
            list_final.append(list_tem)
        return pd.DataFrame(list_final)
    #******
    #reverse transform as Featurenum_to_Onehot
    #******
    def Onehot_to_Featurenum(X_initial):
        list_X_initial = []
        for i in range(len(X_initial)):
            list_tem = X_initial.iloc[i,:]
            list_X_initial_each = []
            for num in range(len(list_tem)):
                if list_tem[num] == 1:
                    list_X_initial_each.append(num)
            list_X_initial.append(sorted(list_X_initial_each))
        return np.array(list_X_initial)
    #******
    #this function can change float type of metric into classes
    #******
    def Change_float_into_classes(Num, Max_num, N_classes):
        return int(Num/(Max_num/N_classes))
    #******
    #this function takes in feature set as X, and accuracy set as y, a pool of other feature set
    #and use active learning to quickly enhenced the model
    #******
    def Use_AL_to_train_featureset(X,y,N_originalset,N_queries,size_of_pool,N_classes,Original_Trainning_model,Feature_Trainning_model):
        Feature_set,Metric_set = PerformanceHandler.Sampling_randomd_featuresets(X,y,Original_Trainning_model,N_originalset,N_classes)
        X_train,X_test,y_train,y_test = train_test_split(Feature_set,Metric_set,test_size=0.5)
        X_pool = PerformanceHandler.pool_generator(Feature_set,size_of_pool)
        X_pool = np.array(X_pool)
        X_train =np.array(X_train)
        y_train = np.array(y_train)
        X_test =np.array(X_test)
        y_test = np.array(y_test)
        learner = ActiveLearner(estimator=Feature_Trainning_model,X_training=X_train,y_training=y_train)
        unqueried_score = [learner.score(X_test, y_test)]
        performance_history = [unqueried_score]
        for index in range(N_queries):
            query_index, query_instance = learner.query(X_pool)
            list_tem = query_instance.tolist()
            list_tem = list_tem[0]
            list_feature_num = []
            for num in range(len(list_tem)):
                if list_tem[num] == 1:
                    list_feature_num.append(num)
            # Teach our ActiveLearner model the record it has requested.
            X_Iterate = X.iloc[:,list_feature_num]
            y_new = PerformanceHandler.Change_float_into_classes(cross_val_score(estimator=Original_Trainning_model,X = X_Iterate,y = y,cv=5,n_jobs=-1).mean(),1,N_classes)
            X_1, y_1 = X_pool[query_index], np.array([y_new])
            learner.teach(X=X_1, y=y_1)
            #print(learner.y_training)
            # Remove the queried instance from the unlabeled pool.
            X_pool = np.delete(X_pool, query_index, axis=0)
            # Calculate and report our model's accuracy.
            model_accuracy = learner.score(X_test, y_test)
            print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

            # Save our model's performance for plotting.
            performance_history.append(model_accuracy)
        return learner