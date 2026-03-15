import numpy as np
import pandas as pd
from data import Vertebral_column
from data import seismic_bumps
from data import churn
from data import indian_liver_patient
from data import spect_heart
import svm
#from sklearn.metrics import classification_report
import trainning_of_adaboost as toa
from sklearn.ensemble import AdaBoostClassifier
import adaboost_svm
from report import report
#from sklearn.metrics import f1_score
from sklearn.metrics  import classification_report,roc_auc_score,precision_recall_fscore_support as score

M=10
C=10000
# theta=2
for theta in range(1,2,1):
    for mr in range(5,18,2):
        new_rate = 1/mr
        Name = "1808 ADA WSVM Indian liver patient_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "ADA WSVM Vertebral column "+str(theta)+"_rate_1_"+str(mr)+".txt"
        #Name = "ADA WSVM Churn_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        #Name = "Semic bump _theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "spect_heart_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        f = open(Name, "w")
        f.write("\n\n Adaboost with SVM (sklearn):")
        print("\n\n Adaboost with SVM (sklearn):\n")

        for i in range(3, 5): 
            S_precision_A_SVM=0
            S_recall_A_SVM=0
            S_fscore_A_SVM=0
            S_precision_A_SVM_N=0
            S_recall_A_SVM_N=0
            S_fscore_A_SVM_N=0

            S_precision_A_WSVM=0
            S_recall_A_WSVM=0
            S_fscore_A_WSVM=0
            S_precision_A_WSVM_N=0
            S_recall_A_WSVM_N=0
            S_fscore_A_WSVM_N=0

            S_precision_A1_SVM=0
            S_recall_A1_SVM=0
            S_fscore_A1_SVM=0
            S_precision_A1_SVM_N=0
            S_recall_A1_SVM_N=0
            S_fscore_A1_SVM_N=0

            S_precision_A1_WSVM=0
            S_recall_A1_WSVM=0
            S_fscore_A1_WSVM=0
            S_precision_A1_WSVM_N=0
            S_recall_A1_WSVM_N=0
            S_fscore_A1_WSVM_N=0

            S_precision_A2_SVM=0
            S_recall_A2_SVM=0
            S_fscore_A2_SVM=0
            S_precision_A2_SVM_N=0
            S_recall_A2_SVM_N=0
            S_fscore_A2_SVM_N=0

            S_precision_A2_WSVM=0
            S_recall_A2_WSVM=0
            S_fscore_A2_WSVM=0
            S_precision_A2_WSVM_N=0
            S_recall_A2_WSVM_N=0
            S_fscore_A2_WSVM_N=0

            S_precision_A12_SVM=0
            S_recall_A12_SVM=0
            S_fscore_A12_SVM=0
            S_precision_A12_SVM_N=0
            S_recall_A12_SVM_N=0
            S_fscore_A12_SVM_N=0

            S_precision_A12_WSVM=0
            S_recall_A12_WSVM=0
            S_fscore_A12_WSVM=0
            S_precision_A12_WSVM_N=0
            S_recall_A12_WSVM_N=0
            S_fscore_A12_WSVM_N=0

            f.write("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
            print("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
            print("\n Test for 1 to 20: \n")            
            for kf in range(1,10):
                X_train = None
                y_train = None
                X_test = None
                y_test = None
                # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                # X_train, y_train, X_test, y_test = spect_heart.load_data(test_size= i * 0.1, new_rate=new_rate)
                #X_train, y_train, X_test, y_test = churn.load_data(test_size= i * 0.1, new_rate=new_rate)
                # X_train, y_train, X_test, y_test = seismic_bumps.load_data(test_size= i * 0.1, new_rate=new_rate)
                X_train, y_train, X_test, y_test = indian_liver_patient.load_data(test_size= i * 0.1, new_rate=new_rate)
                #learner : Adaboost with SVM lib``
                print("\n\n Test :"+str(kf)+"\n")
#-----------------------
                # print("ADA Boost with SVM starting...\n")
                # w, b, a = adaboost_svm.fit(X_train, y_train, M, C, instance_categorization=False, proposed = False)
                # test_pred = adaboost_svm.predict(X_test, w, b, a, M)
                # precision, recall, fscore, support = score(y_test, test_pred)
                # S_precision_A_SVM=S_precision_A_SVM+precision[1]
                # S_recall_A_SVM=S_recall_A_SVM+recall[1]
                #
                # S_precision_A_SVM_N=S_precision_A_SVM_N+precision[0]
                # S_recall_A_SVM_N=S_recall_A_SVM_N+recall[0]

                # print("Done.\n")

                # #learner : Adaboost with W.SVM paper 2016 

                print("ADA Boost with W.SVM starting...\n")
                # # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=True, proposed_preprocessing = False,proposed_alpha = False, test_something = True, theta=theta)
                test_pred2 = toa.predict(X_test, w, b, a, M)
                precision, recall, fscore, support = score(y_test, test_pred2)
                S_precision_A_WSVM=S_precision_A_WSVM+precision[1]
                S_recall_A_WSVM=S_recall_A_WSVM+recall[1]
                
                ### Am
                S_precision_A_WSVM_N=S_precision_A_WSVM_N+precision[0]
                S_recall_A_WSVM_N=S_recall_A_WSVM_N+recall[0]
                
                print("Done.\n")
                w = None
                b = None
                a = None

                #learner : Adaboost with proposed

                print("ADA Boost Nova 1 with SVM starting...\n")
                w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=False, proposed_preprocessing = True, proposed_alpha = False, test_something = True, theta=theta)
                test_pred3 = toa.predict(X_test, w, b, a, M)
                precision, recall, fscore, support = score(y_test, test_pred3)
                S_precision_A1_SVM=S_precision_A1_SVM+precision[1]
                S_recall_A1_SVM=S_recall_A1_SVM+recall[1]
                
                #
                S_precision_A1_SVM_N=S_precision_A1_SVM_N+precision[0]
                S_recall_A1_SVM_N=S_recall_A1_SVM_N+recall[0]
                print("Done.\n")
                w = None
                b = None
                a = None

            
            print(kf)
            print("\n\n FINAL New rate = %s, test size = %s\n\n" %(new_rate,i*0.1))
            

            f.write("\n\n Adaboost with W.SVM:")
            print("\nAdaboost with W.SVM:")
            k_precision_A_WSVM='{0:.4f}'.format(S_precision_A_WSVM/kf)
            k_recall_A_WSVM='{0:.4f}'.format(S_recall_A_WSVM/kf)
            k_fscore_A_WSVM='{0:.4f}'.format((S_precision_A_WSVM+S_recall_A_WSVM)/(2*kf))
            #
            k_precision_A_WSVM_N='{0:.4f}'.format(S_precision_A_WSVM_N/kf)
            k_recall_A_WSVM_N='{0:.4f}'.format(S_recall_A_WSVM_N/kf)
            k_fscore_A_WSVM_N='{0:.4f}'.format((S_precision_A_WSVM_N+S_recall_A_WSVM_N)/(2*kf))
            #f.write(report(y_test,test_pred).to_string())
            f.write("\nPositive: \n precision \t recall \t fscore \n")
            f.write(k_precision_A_WSVM_N+"\t"+k_recall_A_WSVM_N+"\t"+k_fscore_A_WSVM_N+"\n")
            f.write(k_precision_A_WSVM+"\t"+k_recall_A_WSVM+"\t"+k_fscore_A_WSVM+"\n")
            print("\nPositive: \n precision \t recall \t fscore \n")
            print(k_precision_A_WSVM+"\t"+k_recall_A_WSVM+"\t"+k_fscore_A_WSVM+"\n")
            f.write("\n\n\n")

            
        


  
        #learner : Adaboost
    # f.write("\n\n Adaboost with SVM (scratch):")
    # for i in range(3, 4):
    #     X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
    #     f.write("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
    #     w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=False, proposed_preprocessing = False,test_something = True)
    #     test_pred = toa.predict(X_test, w, b, a, M)
    #     f.write(report(y_test,test_pred).to_string())
    #     f.write("\n\n")
    #     f.write(roc_auc_score(y_test, test_pred).to_string())
    #     f.write("\n\n\n")

    f.close()
