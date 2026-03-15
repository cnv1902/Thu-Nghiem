import numpy as np
import pandas as pd
from data import Vertebral_column
from data import seismic_bumps
from data import churn
from data import indian_liver_patient
from data import spect_heart
from data import Co_Author
import svm
#from sklearn.metrics import classification_report
# import trainning_of_adaboost as toa
from sklearn.ensemble import AdaBoostClassifier
import adaboost_svm
from report import report
#from sklearn.metrics import f1_score
from sklearn.metrics  import classification_report,roc_auc_score,precision_recall_fscore_support as score

M=10
C=10000
# theta=2
for theta in range(0,1,1):
    for mr in range(7,8,2):
        new_rate = 1/mr
        Name = "Co_Author 2411"+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "ADABoostTree_Vertebral column "+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "ADABoostTree Churn_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "ADABoostTree Semic bump _theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "spect_heart_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        f = open(Name, "w")
        f.write("\n\n Adaboost with SVM (sklearn):")
        print("\n\n Adaboost with SVM (sklearn):\n")


        for i in range(3, 5): 
            
            S_precision_A_Tree=0
            S_recall_A_Tree=0
            S_fscore_A_Tree=0
            S_precision_A_Tree_N=0
            S_recall_A_Tree_N=0
            S_fscore_A_Tree_N=0

            f.write("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
            print("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
            print("\n Test for 1 to 20: \n")            
            for kf in range(1,20):
                X_train = None
                y_train = None
                X_test = None
                y_test = None
                # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                # X_train, y_train, X_test, y_test = spect_heart.load_data(test_size= i * 0.1, new_rate=new_rate)
                # X_train, y_train, X_test, y_test = churn.load_data(test_size= i * 0.1, new_rate=new_rate)
                #X_train, y_train, X_test, y_test = seismic_bumps.load_data(test_size= i * 0.1, new_rate=new_rate)
                X_train, y_train, X_test, y_test = Co_Author.load_data(test_size= i * 0.1, new_rate=new_rate)
                #learner : Adaboost with SVM lib``
                print("\n\n Test :"+str(kf)+"\n")
#-----------------------
                clf = AdaBoostClassifier(n_estimators=100)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                precision, recall, fscore, support = score(y_test, y_pred)
                S_precision_A_Tree=S_precision_A_Tree+precision[1]
                S_recall_A_Tree=S_recall_A_Tree+recall[1]
                S_fscore_A_Tree=S_fscore_A_Tree+fscore[1]
                #
                S_precision_A_Tree_N=S_precision_A_Tree_N+precision[0]
                S_recall_A_Tree_N=S_recall_A_Tree_N+recall[0]
                S_fscore_A_Tree_N=S_fscore_A_Tree_N+fscore[0]
                print("Done.\n")

            print(kf)
            f.write("\n\n Adaboost Tree:")
            print("\nAdaboost tree:")
            k_precision_A_Tree='{0:.4f}'.format(S_precision_A_Tree/kf)
            k_recall_A_Tree='{0:.4f}'.format(S_recall_A_Tree/kf)
            k_fscore_A_Tree2='{0:.4f}'.format(S_fscore_A_Tree/kf)
            k_fscore_A_Tree='{0:.4f}'.format((S_precision_A_Tree+S_recall_A_Tree)/(2*kf))
            #
            k_precision_A_Tree_N='{0:.4f}'.format(S_precision_A_Tree_N/kf)
            k_recall_A_Tree_N='{0:.4f}'.format(S_recall_A_Tree_N/kf)
            k_fscore_A_Tree_N='{0:.4f}'.format((S_precision_A_Tree_N+S_recall_A_Tree_N)/(2*kf))
            #f.write(report(y_test,test_pred).to_string())
            f.write("\nPositive: \n precision \t recall \t fscore \n")
            f.write(k_precision_A_Tree_N+"\t"+k_recall_A_Tree_N+"\t"+k_fscore_A_Tree_N+"\n")
            f.write(k_precision_A_Tree+"\t"+k_recall_A_Tree+"\t"+k_fscore_A_Tree+"\n")
            f.write("\n \t"+k_fscore_A_Tree2+"\n")
            print("\nPositive: \n precision \t recall \t fscore \n")
            print(k_precision_A_Tree+"\t"+k_recall_A_Tree+"\t"+k_fscore_A_Tree+"\n")
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
