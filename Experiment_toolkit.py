from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class PerformanceHandler:
    def __init__(self, X:'pd.DataFrame', y:'pd.Series') -> None:
        """initial this class with X and y in dataframe"""
        self.X = X
        self.y = y

    def powerset(seq):
        """
        Returns all the subsets of this set. This is a generator.
        """
        if len(seq) <= 1:
            yield seq
            yield []
        else:
            for item in PerformanceHandler.powerset(seq[1:]):
                yield [seq[0]]+item
                yield item

    def Change_float_into_classes(Num, Max_num, N_classes):
        """this function can change float type accuracy to classes"""
        return int(Num/(Max_num/N_classes))

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

    @staticmethod
    def Featurenum_to_Onehot (X_features,original_feature_num:'int'):
        """this function can encode the numTypefeature dataframe to a onehotType dataframe"""
        list_final = []
        for i in range(len(X_features)):
            list_tem = [0 for i in range(original_feature_num)]
            for num in X_features.iloc[i,:]:
                list_tem[num]=1
            list_final.append(list_tem)
        return pd.DataFrame(list_final)

    def Sum_feature_amount(self,X_onehot):
        """take onehot representation as input, ouput the number of features"""
        list_sum = []
        for i in range(len(X_onehot)):
            list_sum.append(X_onehot.iloc[i,:].sum())
        return list_sum

    def Sampling_randomd_featuresets(
            self,
            mod,
            num_cv:int=4,
            number_of_iter:int=1000,
            N_classes:int=40
            ):
            """this function can take a X and y as input, train it with mod, test the model with
            cross validation, and return a dataframe with all feature num and its associated CV values.
            the feature dimention is randomly selected,but always contain the selected features"""
            feature_n_combination = []#list to store all combination
            feature_n_combination_onehot = []#list to store all combination in onehot
            list_feature_num =[]#list to store all features' num
            list_features_CVmeans = []#list to return with all feature num and its associated CV mean value
            dimension_of_feature = len(self.X.columns)
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
            while number_of_iter >= 0:
                if list_feature_selected != []:
                    randomdimension = random.randint(0,len(list_feature_num))
                else:
                    randomdimension = random.randint(1,len(list_feature_num))
                list_randomdim = random.sample(list_feature_num,randomdimension)
                if list_randomdim not in feature_n_combination:
                    feature_n_combination.append(list_randomdim)
                    number_of_iter = number_of_iter-1
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
            y_Iter = self.y
            clf = mod
            for i in range(len(feature_array)):
                try:
                    X_Iterate = self.X.iloc[:,feature_array[i]]
                    scores = cross_val_score(clf, X_Iterate, y_Iter, cv=num_cv,n_jobs=-1)
                    list_tem = feature_n_combination_onehot[i]
                    list_tem.append(scores.mean())
                    list_features_CVmeans.append(list_tem)
                    print(str(i)+'/'+str(len(feature_array)-1))
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

            self.feature_set_rd = list_frame.iloc[:,0:dimension_of_feature]
            self.feature_set = list_frame.iloc[:,0:dimension_of_feature]
            self.performance_set_classes_rd = list_frame.iloc[:,-1]
            self.performance_classes = list_frame.iloc[:,-1]
            self.performance_set_rd = list_frame.iloc[:,-2]
            self.performance_set = list_frame.iloc[:,-2]
            self.feature_performance_frame_rd = list_frame
            self.feature_performance_frame = list_frame
            self.N_classes = N_classes
            self.mod = mod
            self.num_cv = num_cv

    def Sampling_all_featuresets(
            self,
            mod,
            N_classes:int=40,
            num_cv:int=4
            ):
            """this function can take a X and y as input, train it with mod, test the model with
            cross validation, and return a dataframe with all feature num and its associated CV values.
            the feature dimention is randomly selected,but always contain the selected features"""
            feature_n_combination = []#list to store all combination
            feature_n_combination_onehot = []#list to store all combination in onehot
            list_feature_num =[]#list to store all features' num
            list_features_CVmeans = []#list to return with all feature num and its associated CV mean value
            dimension_of_feature = len(self.X.columns)
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
            c =list(PerformanceHandler.powerset(list_feature_num))
            c.remove([])
            feature_n_combination = c
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
            y_Iter = self.y
            clf = mod
            for i in range(len(feature_array)):
                try:
                    X_Iterate = self.X.iloc[:,feature_array[i]]
                    scores = cross_val_score(clf, X_Iterate, y_Iter, cv=num_cv,n_jobs=-1)
                    list_tem = feature_n_combination_onehot[i]
                    list_tem.append(scores.mean())
                    list_features_CVmeans.append(list_tem)
                    print(str(i)+'/'+str(len(feature_array)-1))
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

            self.feature_set_all = list_frame.iloc[:,0:dimension_of_feature]
            self.feature_set = list_frame.iloc[:,0:dimension_of_feature]
            self.performance_set_classes_all = list_frame.iloc[:,-1]
            self.performance_classes = list_frame.iloc[:,-1]
            self.performance_set_all = list_frame.iloc[:,-2]
            self.performance_set = list_frame.iloc[:,-2]
            self.feature_performance_frame_all = list_frame
            self.feature_performance_frame = list_frame
            self.N_classes = N_classes

    def pool_generator (X_initial,size_of_pool:"int"):
        """this function can provide different feature subset compare to the input set,
        you should pass in onehot version of X, and the output is also in a onehot version"""
        list_X_initial = []
        dimention_of_feature = len(X_initial.columns)
        generate_feature_set = []
        #change onehot into feature number
        list_feature_num = [i for i in range(dimention_of_feature)]
        for i in range(len(X_initial)):
            list_tem = X_initial.iloc[i,:]
            list_X_initial_each = []
            for num in range(len(list_tem)):
                if list_tem[num] == 1:
                    list_X_initial_each.append(num)
                
            list_X_initial.append(sorted(list_X_initial_each))

        while size_of_pool >= 0:
            randomnum = random.randint(1,dimention_of_feature)
            feature_randomdim = random.sample(list_feature_num,randomnum)
            feature_randomdim.sort()
            if feature_randomdim not in list_X_initial and feature_randomdim not in generate_feature_set:
                generate_feature_set.append(feature_randomdim)
                size_of_pool = size_of_pool - 1  

        list_final = []
        for i in range(len(generate_feature_set)):
            list_tem = [0 for i in range(dimention_of_feature)]
            for num in generate_feature_set[i]:
                list_tem[num]=1
            list_final.append(list_tem)
        return pd.DataFrame(list_final)

    def Use_AL_to_train_featureset(
    self,
    N_queries,
    size_of_pool,
    Trainning_model
    ):
        """this function takes original dataset as input, and output a fully trained learner 
        to predict accuracy based on different feature sets"""
    
        Feature_set,Metric_set = self.feature_set_nd,self.performance_set_classes_nd
        X_train,X_test,y_train,y_test = train_test_split(Feature_set,Metric_set,test_size=0.2)
        X_pool = PerformanceHandler.pool_generator(Feature_set,size_of_pool)
        X_pool = np.array(X_pool)
        X_train =np.array(X_train)
        y_train = np.array(y_train)
        X_test =np.array(X_test)
        y_test = np.array(y_test)
        learner = ActiveLearner(estimator=Trainning_model,X_training=X_train,y_training=y_train)
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
            X_Iterate = self.X.iloc[:,list_feature_num]
            y_new = PerformanceHandler.Change_float_into_classes(cross_val_score(estimator=self.mod,X = X_Iterate,y = self.y,cv=self.num_cv,n_jobs=-1).mean(),1,self.N_classes)
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

    def Sampling_nd_featuresets(
        self,
        dimention_of_subset:'int',
        mod,
        num_cv:int=4,
        number_of_iter:int=1000,
        N_classes:int=40,
        ):
        """this function can take a X and y as input, train it with mod, test the model with
        cross validation, and return a dataframe with all feature num and its associated CV values.
        the feature dimention is static to 'dimension_of_subset'"""
        feature_n_combination = []#list to store all combination
        list_feature_num =[]#list to store all features' num
        list_features_CVmeans = []#list to return with all feature num and its associated CV mean value
        dimention_of_feature = len(self.X.columns)
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
        y_Iter = self.y
        clf = mod
        for i in range(number_of_iter):
            randomnum = random.randint(0,len(feature_n_combination)-1)
            try:
                X_Iterate = self.X.iloc[:,feature_n_combination[randomnum]]
                scores = cross_val_score(clf, X_Iterate, y_Iter, cv=num_cv,n_jobs=-1)
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
        
        self.feature_set_nd = list_frame.iloc[:,0:dimention_of_subset]
        self.feature_set = list_frame.iloc[:,0:dimention_of_subset]
        self.performance_classes_nd = list_frame.iloc[:,dimention_of_subset+1]
        self.performance_classes = list_frame.iloc[:,dimention_of_subset+1]
        self.performance_set_nd = list_frame.iloc[:,dimention_of_subset]
        self.performance_set = list_frame.iloc[:,dimention_of_subset]
        self.nd_frame = list_frame
        self.feature_performance_frame = list_frame
        self.N_classes = N_classes

    def Report(
    self,
    return_frame_report:bool=False,
    return_relationship_graph:bool = True,
    return_feature_bar:bool = True,
    performance_ranking:int= 4
    ):
        if return_relationship_graph == True:
            feature_sum = self.Sum_feature_amount(self.feature_set)
            list_feature_1 =[]
            for i in range(len(feature_sum)):
                list_tem = [feature_sum[i],self.performance_classes[i]]
                list_feature_1.append(list_tem)
            list_feature_2 = []
            for i in range(1,len(self.X.columns)):
                for num in range(self.performance_classes.min(),(self.performance_classes.max()+1)):
                    counter = list_feature_1.count([i,num])
                    list_feature_2.append([i,num,counter])
            fig = plt.figure(figsize=(15,10))
            axes = fig.add_axes([0,0,1,1])
            axes.set_title('relationship between feature amount and performance')
            axes.set_xlabel('feature amount')
            axes.set_ylabel('performance class')
            axes.scatter([list_feature_2[i][0] for i in range(len(list_feature_2))],
            [list_feature_2[i][1] for i in range(len(list_feature_2))],
            s=[list_feature_2[i][2] for i in range(len(list_feature_2))],
            c = np.random.rand(len(list_feature_2)))
        if return_feature_bar == True:
            for i in range(performance_ranking):
                idx = []
                for num in range(len(self.feature_set)):
                    if self.performance_classes[num] == (self.performance_classes.max()-i):
                        idx.append(num)
                a = self.feature_set.iloc[idx,:]
                list_different_feature_sum = []
                for q in range(len(self.X.columns)):
                    list_different_feature_sum.append(a.iloc[:,q].sum())
                f = plt.figure(figsize=(15,10))
                plt.title("How different features influence the "+str((self.performance_classes.max()-i))+" class")
                plt.xlabel("Feature number")
                plt.ylabel("Feature amount")
                plt.bar(range(len(self.X.columns)),list_different_feature_sum)

        if return_frame_report == True:
            list_frame_report = []
            for i in sorted(self.performance_classes.value_counts().index.tolist()):
                idx = []
                for num in range(len(self.feature_set)):
                    if self.performance_classes[num] == i:
                        idx.append(num)
                a = self.feature_set.iloc[idx,:]
                list_different_feature_sum = []
                for num in range(len(self.X.columns)):
                    list_different_feature_sum.append(a.iloc[:,num].sum())
                list_different_feature_sum.append(i)
                list_frame_report.append(list_different_feature_sum)
                frame = pd.DataFrame(list_frame_report)
                frame.rename(columns={len(self.X.columns):'performance class'})
            return frame

    def predict(self,Feature_list:list):
        array = np.array(self.feature_set)
        a = list(PerformanceHandler.Featurenum_to_Onehot(pd.DataFrame([Feature_list]),\
            len(self.X.columns)).iloc[0,:])
        print(a)
        for i in range(len(array)):
            if list(array[i])==a:
                print(f'The accuracy class is {self.performance_classes[i]} and the accuracy is {self.performance_set[i]}')
                break
        
                    




    

    
