import pandas as pd
import numpy as np
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import math


class NaiveBayesModel:

    # def undefined_extract(self, df):
    #     undefined = df['class'] == "undefined"
    #     # output dataframe without undeined
    #     df_undefine = df[undefined]
    #     print(df_undefine)
    #     return df_undefine

    def WordGivenNoPI(self, tempNegDocVector, uniqueWords):
        data = np.zeros([1, len(uniqueWords)])
        wordGivenNoPI = pd.DataFrame(data, columns=uniqueWords)
        columnSum = tempNegDocVector.sum(axis=1, skipna=True)
        numWordsInNoPI = columnSum.sum()

        for word in uniqueWords:
            nk_wordinNoPI = tempNegDocVector[word].sum()
            wordGivenNoPI.at[0, word] = (nk_wordinNoPI + 1) / (numWordsInNoPI + len(uniqueWords))
        return (wordGivenNoPI, numWordsInNoPI)

    def TrainModel(self, Train_Vector, uniqueWords):
        yesCount = Train_Vector["PurchaseIntention"] == "yes"
        tempPosDocVector = Train_Vector[yesCount]
        totalPI = tempPosDocVector["PurchaseIntention"].count()
        print("total PI ", totalPI)

        noCount = Train_Vector["PurchaseIntention"] == "no"
        tempNegDocVector = Train_Vector[noCount]
        # print(tempNegDocVector["PurchaseIntention"])
        totalNonPI = tempNegDocVector["PurchaseIntention"].count()
        print("total non PI ", totalNonPI)
        # print(totalPI+totalNonPI)
        # totalNonPI = docVector["PurchaseIntention"].count() - totalPI
        total = totalPI + totalNonPI
        Prob_PI = totalPI / total
        Prob_NoPI = totalNonPI / total

        data = np.zeros([1, len(uniqueWords)])
        wordGivenPI = pd.DataFrame(data, columns=uniqueWords)
        columnSum = tempPosDocVector.sum(axis=1, skipna=True)
        numWordsInPI = columnSum.sum()

        for word in uniqueWords:
            nk_wordinPI = tempPosDocVector[word].sum()
            wordGivenPI.at[0, word] = (nk_wordinPI + 1) / (numWordsInPI + len(uniqueWords))

        df_wordGivenNoPI, numWordsInNoPI = self.WordGivenNoPI(tempNegDocVector, uniqueWords)
        return wordGivenPI, df_wordGivenNoPI, Prob_PI, Prob_NoPI, numWordsInPI, numWordsInNoPI


    def Predict_kfold(self, Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,df_test, clean):
        # eng_df = clean.check_english(df_test)

        eng_df = df_test
        # print(eng_df)
        new_df = clean.space(eng_df)
        # print("Hello clean data")
        # print(new_df["text"])
        df_remove_stopWords = clean.remove_stopwords(new_df)
        new_corpus_df = clean.handle_negation(df_remove_stopWords)
        remove_punc_df = clean.remove_punc(new_corpus_df)

        predict_df = pd.DataFrame()
        weighPI = Prob_PI
        weighNoPI = Prob_NoPI
        count_test = 0
        for sentence in remove_punc_df['text']:
            # print(count_test)
            for word in sentence.lower().split():
                if word in uniqueWords:
                    weighPI = weighPI * df_WordGivenPI.at[0, word]
                    weighNoPI = weighNoPI * df_WordGivenNoPi.at[0, word]
                else:
                    weighPI = weighPI * (1 / (numWordsInPI + len(uniqueWords)))
                    weighNoPI = weighNoPI * (1 / (numWordsInNoPI + len(uniqueWords)))
            predict_df.at[count_test, 'WeightPI'] = weighPI
            if weighPI > weighNoPI:
                predict_df.at[count_test, 'PredictedClass'] = 'yes'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])
            else:
                predict_df.at[count_test, 'PredictedClass'] = 'no'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])

            count_test += 1
            weighPI = Prob_PI
            weighNoPI = Prob_NoPI
        return predict_df, remove_punc_df

    def Predict(self, Prob_PI, Prob_NoPI, uniqueWords, df_WordGivenPI, df_WordGivenNoPi, numWordsInPI, numWordsInNoPI,clean):
        test_path = "dataFiles\\Testing.csv"
        test_data, test_df = clean.extract(test_path)

        # eng_df = clean.check_english(test_data)
        eng_df = test_data
        # print(eng_df)
        # df_stem = clean.Stemming(eng_df)
        new_df = clean.space(eng_df)
        # print("Hello clean data")
        # print(new_df["text"])

        df_remove_stopWords = clean.remove_stopwords(new_df)

        new_corpus_df = clean.handle_negation(df_remove_stopWords)
        remove_punc_df = clean.remove_punc(new_corpus_df)

        predict_df = pd.DataFrame()
        weighPI = Prob_PI
        weighNoPI = Prob_NoPI
        count_test = 0
        for sentence in remove_punc_df['text']:
            # print(count_test)
            for word in sentence.lower().split():
                if word in uniqueWords:
                    weighPI = weighPI * df_WordGivenPI.at[0, word]
                    weighNoPI = weighNoPI * df_WordGivenNoPi.at[0, word]
                else:
                    weighPI = weighPI * (1 / (numWordsInPI + len(uniqueWords)))
                    weighNoPI = weighNoPI * (1 / (numWordsInNoPI + len(uniqueWords)))
            predict_df.at[count_test, 'WeightPI'] = weighPI
            if weighPI > weighNoPI:
                predict_df.at[count_test, 'PredictedClass'] = 'yes'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])
            else:
                predict_df.at[count_test, 'PredictedClass'] = 'no'
                # print(test_data.at[count_test,'text'],test_data.at[count_test,'PredictedClass'])

            count_test += 1
            weighPI = Prob_PI
            weighNoPI = Prob_NoPI
        return predict_df, remove_punc_df

