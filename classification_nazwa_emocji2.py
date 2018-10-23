import numpy as np
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from face_analysis import FaceAnalysis


class Classificator_v2:
    def __init__(self, src_features, src_classes, align=True, point_type=FaceAnalysis.POINTS_TYPE.Independent,
                 cheek_color=False, wrinkles=False):
        method = "Neigh"
        self.filename = 'classificator.pkl'
        self.align = align
        self.point_type = point_type
        self.cheek_color = cheek_color
        self.wrinkles = wrinkles

        features = np.loadtxt(src_features, delimiter=",")
        classes = json.loads(open(src_classes).read())
        classes1 = []

        for key, value in classes.items():
            classes1.append(value)

        clf = ""
        if (method == "Neigh"):
            clf = KNeighborsClassifier(n_neighbors=15)  # Metoda k-najblizszych sasiadow
        elif (method == "Naive"):
            clf = GaussianNB()  # Metoda naiwnego Bayesa
        elif (method == "Tree"):
            clf = DecisionTreeClassifier()  # Metoda Decision Tree

        clf.fit(features, classes1)
        joblib.dump(clf, self.filename)  # Zapisane go do pliku

    def classify_image(self, image):
        feature = FaceAnalysis.get_features(image, align=self.align, point_type=self.point_type,
                                            cheek_color=self.cheek_color, wrinkles=self.wrinkles)
        feature = feature.reshape(1, -1)
        neigh = joblib.load(self.filename)
        classes = neigh.predict(feature)
        prob = neigh.predict_proba(feature)

        return classes, prob


class Classificator:
    @staticmethod
    def CreateClassificator(srcFeat, srcClasses):  # Tworzenie klasyfikatora
        method = "Neigh"
        features = np.loadtxt(srcFeat, delimiter=",")
        classes = json.loads(open(srcClasses).read())
        classes1 = []

        for key, value in classes.items():
            classes1.append(value)

        clf = ""
        if (method == "Neigh"):
            clf = KNeighborsClassifier(n_neighbors=3)  # Metoda k-najblizszych sasiadow
        elif (method == "Naive"):
            clf = GaussianNB()  # Metoda naiwnego Bayesa
        elif (method == "Tree"):
            clf = DecisionTreeClassifier()  # Metoda Decision Tree

        clf.fit(features, classes1)
        joblib.dump(clf, 'classificator.pkl')  # Zapisane go do pliku

    @staticmethod
    def ClassifyImages(file, image, emotions_4, emotions_5, emotions_6):  # Dolaczenie nowego zdjecia i zaklasyfikaowanie go

        if (emotions_4):
            feature = FaceAnalysis.get_features(image, align=True, point_type=FaceAnalysis.POINTS_TYPE.Manual,
                                                cheek_color=False, wrinkles=False)
            path = os.path.join("resources","4_feature_4_emotions_description.json")
            # print('emotions4')
        if (emotions_5):
            feature = FaceAnalysis.get_features(image, align=True, point_type=FaceAnalysis.POINTS_TYPE.Independent,
                                                cheek_color=False, wrinkles=False)
            path = os.path.join("resources","3_feature_description.json")
            # print('emotoins5')
        if (emotions_6):
            feature = FaceAnalysis.get_features(image, align=True, point_type=FaceAnalysis.POINTS_TYPE.Independent,
                                                cheek_color=False, wrinkles=False)
            path = os.path.join("resources","10_feature_description.json")
        # print('emotions6')

        feature = feature.reshape(1, -1)
        neigh = joblib.load(file)
        classes = neigh.predict(feature)
        prob = neigh.predict_proba(feature)
        allClasses = []
        allProbs = []
        napis = ""
        probSum = 0

        with open(path) as data_file:
            data = json.load(data_file)

        for key, values in data.items():
            if (not (values in allClasses)):
                allClasses.append(values)
                allProbs.append(0)
            if (float(key) == classes[0]):
                napis = values

        for key, values in data.items():
            index = allClasses.index(values)
            allProbs[index] += round(prob[0][int(key)], 2)
            if (values == napis):
                probSum += prob[0][int(key)]

       # print(allClasses)
        #print(allProbs)

        return napis, prob, probSum, allClasses, allProbs
