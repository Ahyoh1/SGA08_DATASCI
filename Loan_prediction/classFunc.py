class modelCompile:
    def __init__(self, pred_var, targ_var, scaled):
        '''
        Class initializes self with predictor features and target variable

        pred_var contains dataframe of independent/predictor features 
        targ_var is the target variable
        scaled: Takes in boolean value. Scales the parameters if True and doesn't scale if False
        '''
        X = pred_var.values
        y = targ_var
        if scaled == True:
            sca = MinMaxScaler()
            X   = sca.fit_transform(X)
            self.X = X
        elif scaled == False:
            self.X = X
        self.y = y


    def modBuilder(self, mod, size, r_s):

        '''
        Function defines the model and splits the data set into train and test.

        mod: Input is the current model being passed in which could be RandomForest, 
        DecisionTree, Log/linear or any kind. Model parameters are chosen in the function.
        size: The prefered percentage of test size out of the whole dataset.
        r_s: random state set. 
        '''
        # X and y arrays are returned from init class function
        X = self.X
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_s, test_size=size)
        y_train = np.ravel(y_train)
        self.f_mod = mod.fit(X_train, y_train)
        self.y_test = y_test
        self.y_train = y_train
    
        return X_train, X_test


    def pred_(self, test):
        '''
        Function predicts the xtest using the current model selected

        test: specifies the split train test dataset from the mod builder function
        
        '''
        #Model is returned from the modbuilder function. Reused to predict test set
        f_mod = self.f_mod
        pred_ = f_mod.predict(test)
        eval = []
        for i in pred_:
            yhat = np.array(i).tolist()
            eval.append(yhat)

        self.eval = eval

        return eval
    

    def yhatPred(self, test, numbers):
        '''
        This function prints the ytest and ypred side by side in a dataframe format.

        test: The test values that were predicted. 
        numbers: numbers of preview loaded from the table
        '''
        tab = pd.DataFrame({'Actual': test, 'Predicted' : self.eval})
        tab = tab.head(numbers)

        return tab
    
    
    def accuracy_(self):
        '''
        This function returns the accuracy score of the model fitted in the class function
        '''

        a_score = metrics.accuracy_score(self.y_test, self.eval)

        return a_score


    def rmse_(self):
        mse = metrics.mean_squared_error(self.y_test, self.eval)
        rmse = np.sqrt(mse)

        return rmse


    def hyper (self, mod, rand, iter, cv, vb, r_s, n_j):
        mod_random = RandomizedSearchCV(estimator = mod, param_distributions = rand, n_iter = iter, 
                                        cv = cv, verbose = vb, random_state = r_s, n_jobs = n_j)
        mod_random.fit(X_train, self.y_train)
        best_parameter = mod_random.best_params_
        
        return  best_parameter