import EvaluateModel as em
import pandas as pd
import Clean as cn
import NaiveBayes as nb

import docVector as dv


def naive_bayes():
    model = nb.NaiveBayesModel()
    clean = cn.DataCLean()
    doc_vector = dv.DocumentVector()
    df_clean, uniqueWords = clean.Clean()
    # print(uniqueWords)
    docVector = doc_vector.DocVector(df_clean, uniqueWords)
    df_WordGivenPI,df_WordGivenNoPi,Prob_PI,Prob_NoPI,numWordsInPI,numWordsInNoPI = model.TrainModel(docVector, uniqueWords)
    predict_df, test_data = model.Predict(Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,clean)
    test_data["PredictedClass"] = predict_df["PredictedClass"]
    print("-------------------Test Data Prediction--------------------------")
    print(test_data)
    print("--------------Naive Bayes Accuracy Stats---------------------------")
    stats = em.Evaluate()
    TP, FN, TN, FP = stats.confusion_matrix(test_data, predict_df)
    print("Accuracy = ",stats.Accuracy(TP, TN, FP, FN))
    print("Precision = ",stats.Precision(TP, FP))
    print("Recall = ",stats.Recall(TP, FN))
    print("fScore = ",stats.fScore(TP, FN, FP))
    print("True Negative = ", stats.TrueNegative(TN, FP))
    print("---------------------------------------------------------------------")


naive_bayes()
