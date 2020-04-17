import os
import sys
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from configuration import log_root_dir



class Model_Performance_Diagnosis:
    def __init__(self, job_id, product_id):
        self.__job_id = job_id
        self.__product_id = product_id
        self.__train_log_path = os.path.join(log_root_dir, job_id, product_id,"training_result_step.log")
        self.__test_log_path = os.path.join(log_root_dir, job_id, product_id,"test_result_step.log")
        
    def get_class_order(self, input_list):
        result = list(set(input_list))
        for i in range(0,len(result)):
            result[i] = int(result[i])
        result.sort()
        for i in range(0,len(result)):
            result[i] = str(result[i])
        return result

    def get_train_diagonosis(self):
        
        train_log_path = self.__train_log_path
        if not os.path.exists(train_log_path):
            print("ERROR: No train/valide log exists, please check if the job_id or product_id are correct."
                  "Also possible because the file name has been modified or file has been removed." 
                 )
        else:
            train_valid_log = pd.read_table(train_log_path)
        
            train_log = train_valid_log[train_valid_log["type"] == "train"]
        
            confusion_matrix_dict = {} #Used to store confusion matrix for all epochs, key: epochs, value: confusion_matrix
        
            # Start iteration for retrive the predict_label and actual label list to form the confusion matrix 
            for epoch in train_log["epoch"].unique():
                train_log_per_epoch = train_log[train_log["epoch"] == epoch]
            
                # Parse predict label to single number and append to predict_label_list
                predict_label_list = []
                for i in train_log_per_epoch.index:
                    predict_label = train_log_per_epoch["predict_labels"][i].split("[")[1].split("]")[0].split(" ")
                    for j in predict_label:
                        if j != "":
                            predict_label_list.append(j)
            
                # Parse actual label to single number and append to actual_label_list
                actual_label_list = []
                for i in train_log_per_epoch.index:
                    actual_label = train_log_per_epoch["actual_labels"][i].split("[")[1].split("]")[0].split(" ")
                    for j in actual_label:
                        if j != "":
                            actual_label_list.append(j)
        
                # Form confusion matrix with benchmarked label list
                class_labels = self.get_class_order(actual_label_list)
                cm_obj = confusion_matrix(actual_label_list, 
                                          predict_label_list, 
                                          labels=class_labels)
                confusion_matrix_dict[str(epoch)] = cm_obj
            
                #Get max epoch in order to auto-scaling the plot size
                max_epoch = epoch
        
            # Do initialization of the plot, 
            fig, ax = plt.subplots(nrows= max_epoch -1, 
                                   ncols=2, 
                                   figsize=(8*2, 
                                            6.5 * (max_epoch-1)))
            
            sns.set(style="darkgrid")
            
            for epoch in range(0,max_epoch-1):
                sns.heatmap(confusion_matrix_dict[str(epoch)], 
                            cmap="YlGnBu", 
                            annot=True, 
                            fmt="d", 
                            ax=ax[epoch,0])
                ax[epoch,0].set_title('epoch = ' + str(epoch),
                                      fontsize = 15)
                ax[epoch,0].set_xlabel('predict_label',
                                       fontsize = 15)
                ax[epoch,0].set_ylabel('actual_label',
                                       fontsize = 15)
            
                sns.lineplot(x="step", 
                             y="train_accuracy",
                             data=train_log[train_log["epoch"] == epoch], 
                             ax=ax[epoch,1])
                ax[epoch,1].set_title('epoch = ' + str(epoch),
                                  fontsize = 15)
                ax[epoch,1].set_xlabel('train_accuracy',
                                   fontsize = 15)
                ax[epoch,1].set_ylabel('steps',
                                   fontsize = 15)
                ax[epoch,1].set_yticks(np.arange(0,1,0.1))
            
            diagnosis_file_name = "diagnosis/train_diagnosis_" + self.__job_id + "_" + self.__product_id + ".png"
            plt.savefig(diagnosis_file_name)

    def get_valid_diagonosis(self):
        
        train_log_path = self.__train_log_path
        if not os.path.exists(train_log_path):
            print("ERROR: No train/valide log exists, please check if the job_id or product_id are correct."
                  "Also possible because the file name has been modified or file has been removed." 
                 )
        else:
            train_valid_log = pd.read_table(train_log_path)
        
            valid_log = train_valid_log[train_valid_log["type"] == "validation"]
        
            confusion_matrix_dict = {} #Used to store confusion matrix for all epochs, key: epochs, value: confusion_matrix
        
            # Start iteration for retrive the predict_label and actual label list to form the confusion matrix 
            for epoch in valid_log["epoch"].unique():
                valid_log_per_epoch = valid_log[valid_log["epoch"] == epoch]
            
                # Parse predict label to single number and append to predict_label_list
                predict_label_list = []
                for i in valid_log_per_epoch.index:
                    predict_label = valid_log_per_epoch["predict_labels"][i].split("[")[1].split("]")[0].split(" ")
                    for j in predict_label:
                        if j != "":
                            predict_label_list.append(j)
            
                # Parse actual label to single number and append to actual_label_list
                actual_label_list = []
                for i in valid_log_per_epoch.index:
                    actual_label = valid_log_per_epoch["actual_labels"][i].split("[")[1].split("]")[0].split(" ")
                    for j in actual_label:
                        if j != "":
                            actual_label_list.append(j)
        
                # Form confusion matrix with benchmarked label list
                class_labels = self.get_class_order(actual_label_list)
                cm_obj = confusion_matrix(actual_label_list, 
                                          predict_label_list, 
                                          labels=class_labels)
                confusion_matrix_dict[str(epoch)] = cm_obj
            
                #Get max epoch in order to auto-scaling the plot size
                max_epoch = epoch
        
            # Do initialization of the plot, 
            fig, ax = plt.subplots(nrows= max_epoch -1, 
                                   ncols=2, 
                                   figsize=(8*2, 
                                            6.5 * (max_epoch-1)))
            
            sns.set(style="darkgrid")
            
            for epoch in range(0,max_epoch-1):
                sns.heatmap(confusion_matrix_dict[str(epoch)], 
                            cmap="YlGnBu", 
                            annot=True, 
                            fmt="d", 
                            ax=ax[epoch,0])
                ax[epoch,0].set_title('epoch = ' + str(epoch),
                                      fontsize = 15)
                ax[epoch,0].set_xlabel('predict_label',
                                       fontsize = 15)
                ax[epoch,0].set_ylabel('actual_label',
                                       fontsize = 15)
            
                sns.lineplot(x="step", 
                             y="train_accuracy",
                             data=valid_log[valid_log["epoch"] == epoch], 
                             ax=ax[epoch,1])
                ax[epoch,1].set_title('epoch = ' + str(epoch),
                                  fontsize = 15)
                ax[epoch,1].set_xlabel('train_accuracy',
                                   fontsize = 15)
                ax[epoch,1].set_ylabel('steps',
                                   fontsize = 15)
                ax[epoch,1].set_yticks(np.arange(0,1,0.1))
        
            diagnosis_file_name = "diagnosis/valid_diagnosis_" + self.__job_id + "_" + self.__product_id + ".png"
            plt.savefig(diagnosis_file_name)
            
def main(argv):
    # Need the user to provide system argv for job_id and product_id, it is prepared for frontend calling
    if len(argv) < 2 or len(argv) > 3:
        print("ERROR: Format error, refer to the usage: python test.py job_id product_id")
    elif not argv[1].isdigit():
        print("ERROR: Format error, job_id must be in int format")
    elif not argv[1].isalnum():
        print("ERROR: Format error, product_id must be consistent by character or number, without special character")
    else:
        print("-" * 82) 
        print("INFO: Start training set diagnosis " + datetime.datetime.now().strftime("%Y%m%d%H%M%S")) 
        print("-" * 82) 
        mpd_obj = Model_Performance_Diagnosis(argv[1],
                                              argv[2])
        mpd_obj.get_train_diagonosis()
        print("INFO: Finish training set diagnosis, please check train_diagnosis.png "
              + datetime.datetime.now().strftime("%Y%m%d%H%M%S")) 
        print("-" * 82) 
        print("INFO: Start validation set diagnosis " + datetime.datetime.now().strftime("%Y%m%d%H%M%S")) 
        print("-" * 82) 
        mpd_obj = Model_Performance_Diagnosis(argv[1],
                                              argv[2])
        mpd_obj.get_valid_diagonosis()
        print("INFO: Finish valid set diagnosis, please check valid_diagnosis.png "
              + datetime.datetime.now().strftime("%Y%m%d%H%M%S")) 
        
if __name__ == "__main__":
    main(sys.argv)
    
    

            
            

       


