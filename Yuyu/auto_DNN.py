#########################################################################################################################################
# Project     : Automation of Model Training
#
# Coding      : Xiong Yuyu
#
# Date        : Since 2020-02-05
#
# Note        : Determining the Number of Hidden Layers
#   0 - Only capable of representing linear separable functions or decisions.
#   1 - Can approximate any function that contains a continuous mapping
#       from one finite space to another.
#   2 - Can represent an arbitrary decision boundary to arbitrary accuracy
#       with rational activation functions and can approximate any smooth
#       mapping to any accuracy.
#
# Questions   : Not clear how to send metrics as input paramter to build estimator
#
# Description : Automate the training of a dense neutral network
#               1) 
#               2) 
#########################################################################################################################################


from Model.model_config import *
from Global_fun import *

glb_metrics = ['accuracy']
glb_rand_seed = 2019

class cls_auto_DNN(object):
    """Class to automate the trainng of DNN. It includes functions:
       1) Set the parameters for model tuning
       2) Model training
       3) Prediction
       4) Evaluation
    """
    def __init__(self, x_train, y_train, 
                    batch_size_flag = True, 
                    epoch_max = 100, 
                    dropout_flag = True,
                    metrics = ['accuracy'], 
                    cv = 1,
                    rand_seed = glb_rand_seed):
        """Function: Initialize the class to train DNN
           Input:    1) 
                     2)       
                     x) cv. 1: use boot to tune model (fast), otherwise, cross validation (slow)               
           Output:              
        """
        self.x_train = x_train
        self.y_train = y_train 
        self.input_dim = x_train.shape[1]
        self.output_dim = y_train.shape[1]         
        self.batch_size_flag = batch_size_flag
        self.epoch_max = epoch_max # Use early stopping to decide whether to stop tuning        
        self.dropout_flag = dropout_flag 
        self.metrics = metrics
        if cv >= 1:
            self.cv = cv # 1: bootstrap, otherwise use CV to tune model parameters
        else:
            sys.exit("cv >= 1")
        
        self.rand_seed = rand_seed

        #Intialize the tuning parameters
        self.__fun_param_set()
    #End of Function "__init__"

    def __fun_param_set(self):
        """Function: Set parameters to train DNN based on input parameters
           Input:                                        
           Output:              
        """
        #set the number of neurons in hidden layers
        #layer-2 is half of layer-1
        neuron_num_1st_layer = [self.input_dim, int(1.5*self.input_dim), 2*self.input_dim]
        #NOTE. if memory is not enough
#         parameter = 0.2
#         neuron_num_1st_layer = [self.input_dim, int(1.5*self.input_dim), 2*self.input_dim]
#         neuron_num_1st_layer = [int(parameter*x) for x in neuron_num_1st_layer]
        
        neuron_num_2nd_layer = [int(x/2) for x in neuron_num_1st_layer]
        self.neurons = list(zip(neuron_num_1st_layer, neuron_num_2nd_layer))
        self.neurons = [list(x) for x in self.neurons]

        self.optimizer = Adam() # By default we use Adam, and not tune learning rate

        #Set activation function for hidden layer
        self.activation_hidden = 'relu'        

        #set activation function /loss function for output layer based on output dimensionality
        if self.output_dim > 1:
            self.activation_output = 'sigmoid' #multi-class multi-label classification
            self.loss_fun = 'binary_crossentropy'
        else:
            self.activation_output = 'softmax' #binary classfication
            self.loss_fun = 'categorical_crossentropy'

        #Set batch size
        if self.batch_size_flag == True:
            self.batch_size = [16, 32] #Tune batch size
        else:
            self.batch_size = [32] #fix batch size
        
        if self.dropout_flag == True:
            self.dropout_rate = [0.2, 0.4]
        else:
            self.dropout_rate = [0.2]     

        #split training data into training and validation (fast version of model training)
        if self.cv == 1:
            t_size = int(self.x_train.shape[0]*0.8)
            self.train_val_split = [-1]*t_size + [0]*(self.x_train.shape[0]-t_size)
            seed(self.rand_seed)
            shuffle(self.train_val_split)
            self.ps = PredefinedSplit(self.train_val_split)
        else:
            self.ps = self.cv
    #End of Function 'fun_param_set'

    def fun_print_param(self):
        """Function: To print out parameters to tune model. Debugging only
           Input:                                        
           Output:              
        """     
        print("neurons: ", self.neurons)
        print("optimizer: adam" )        
        print("hidden activation: ", self.activation_hidden)
        print("output activation: ", self.activation_output)
        print("loss function: ", self.loss_fun)
        print("batch size: ", self.batch_size)    
        print("dropout rate: ", self.dropout_rate)
        print("metrics: ", self.metrics)
        print("cv: ", self.cv)
        if self.cv == 1:
            for train_index, validation_index in self.ps.split():
                print("training: ", len(train_index), "; validation: ", len(validation_index))
    #End of Function 'fun_print_param'
    
    @staticmethod
    def __fun_create_model(neurons, dropout_rate, 
                        input_dim, output_dim,
                        activation_hidden, activation_output,
                        # eval_metrics,
                        loss_fun):
        """Function: To create model estimator
            Note. Not clear how to set metrics as an input parameter
            Input:    1) neurons. A pair: 1st = #neurons in 1st hidden layer. 
                                            2st = #neurons in 2nd hidden layer  
                    2) dropout_rate.                  
            Output:  estimator
        """  
        print('+'*30, " parameters inside Function fun_create_model ", '-'*30)
        print("input_dim = {}, output_dim = {}, activation_hidden = {}, activation_output = {}, loss_fun = {}" \
                    .format(input_dim, output_dim, activation_hidden, activation_output, loss_fun))    

        # create model
        np.random.seed(glb_rand_seed)
        model = Sequential()    

        #Layer 1 -- hidden               
        model.add(Dense(units = neurons[0], 
                        input_shape = (input_dim,),                         
                        kernel_initializer = 'random_uniform',                         
                        kernel_regularizer = l2(0.01),
                        bias_regularizer = l2(0.01)))    
        model.add(BatchNormalization())            
        model.add(Activation(activation_hidden))    
        model.add(Dropout(rate = dropout_rate, 
                            seed = glb_rand_seed))

        #Layer 2 -- hidden
        model.add(Dense(units = neurons[1], 
                        kernel_initializer = 'random_uniform',                        
                        kernel_regularizer = l2(0.01),
                        bias_regularizer = l2(0.01)))    
        model.add(BatchNormalization())
        model.add(Activation(activation_hidden))    
        model.add(Dropout(rate = dropout_rate, 
                            seed = glb_rand_seed))

        #Layer 3 -- output
        model.add(Dense(units = output_dim,                         
                        kernel_initializer = 'random_uniform', 
                        activation = activation_output))    
        
        # Compile model
        model.compile(optimizer = Adam(),
                      loss = loss_fun,
                      metrics = glb_metrics)
        
        return model
    #End of Function 'fun_create_model'

    def fun_train_model(self, reproducible = False):
        """Function: To train model
           Input:    1) 
                     2) 
           Output:  trained model
        """  
        #Not clear yet how to make model training reproducible
        #If gridSearchCV has only 1 CPU, it's reproducible
        np.random.seed(self.rand_seed)

        # create model     
        print('+'*30, " tuning param ", '-'*30)
        self.fun_print_param() #For debugging only

        model = KerasClassifier(build_fn = cls_auto_DNN.__fun_create_model, 
                                input_dim = self.input_dim, 
                                output_dim = self.output_dim, 
                                activation_hidden = self.activation_hidden,
                                activation_output = self.activation_output,
                                # eval_metrics = ['accuracy'], #Not allowed.
                                loss_fun = self.loss_fun,                                 
                                verbose=0)
        
        #set parameter grid        
        param_grid = dict(neurons = self.neurons, 
                            batch_size = self.batch_size, 
                            dropout_rate = self.dropout_rate)
        
        if reproducible == True:
            job_num = 1
        else:
            job_num = -1 #use all processes
        
        grid = GridSearchCV(estimator = model, 
                            param_grid = param_grid, 
                            n_jobs = job_num,                             
                            cv = self.ps)
        
        earlystop_callback = EarlyStopping(monitor = 'acc', 
                                            mode = 'max', 
                                            patience = 1)

        grid_result = grid.fit(X = self.x_train, 
                                y = self.y_train, 
                                epochs = self.epoch_max, 
                                callbacks=[earlystop_callback],
                                verbose = True)
        # summarize results        
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        
        return grid_result
    #End of Function 'fun_train_model'

    def fun_pred(self, grid_result, x_test, type = 'prob'):        
        if type == 'prob':
            pred = grid_result.predict_proba(x_test)
        else:
            pred = grid_result.predict(x_test)
        return pred
    #End of Function 'fun_pred'

    #TO be revised...
    def fun_eval(self, y_pred, y_test):
        """Function: evaluate the results
           Note. Now ad-hoc
           Input:    1) 
                     2) 
           Output:  
        """  
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = pd.DataFrame(y_pred, columns = ["c1", "c2", "c3","c4"])
        y_pred = y_pred.astype(int)
        y_test = y_test.reset_index(drop=True)
        print(np.mean(y_pred["c1"] == y_test["c1"]))
    #End of Function 'fun_eval'
    
#End of class 'cls_auto_DNN
