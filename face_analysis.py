import cv2
import numpy as np
import dlib

from enum import Enum
from os import path


# Klasa sluzaca do obrobki twarzy
class FaceAnalysis:
    JAWLINE_POINTS = list(range(0, 17))
    RIGHT_EYEBROW_POINTS = list(range(17, 22))
    LEFT_EYEBROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))
    POINTS_TYPE = Enum('POINTS_TYPE', 'Full NoJaw Independent Manual Position AllDistances')
    predictor = dlib.shape_predictor(path.join('resources', 'shape_predictor_68_face_landmarks.dat'))
    face_cascade = cv2.CascadeClassifier(path.join('resources', 'haarcascade_frontalface_default.xml'))

    # Metoda zwraca wektor cech dla danego obrazka
    @staticmethod
    def get_features(image, align=True, point_type=POINTS_TYPE.Full, cheek_color=False, wrinkles=False):
        try:
            points = FaceAnalysis.get_landmarks(image)
            if align:
                image = FaceAnalysis.align(points, image)
                points = FaceAnalysis.get_landmarks(image)
        except:
            raise Exception('Points not found')

        point_features = []
        color_features = []
        wrinkle_features = []

        # Punkty twarzy
        if point_type == FaceAnalysis.POINTS_TYPE.Full:
            point_features = FaceAnalysis.get_features1(points)
        elif point_type == FaceAnalysis.POINTS_TYPE.NoJaw:
            point_features = FaceAnalysis.get_features2(points)
        elif point_type == FaceAnalysis.POINTS_TYPE.Independent:
            point_features = FaceAnalysis.get_features3(points)
        elif point_type == FaceAnalysis.POINTS_TYPE.Manual:
            point_features = FaceAnalysis.get_features4(points)
        elif point_type == FaceAnalysis.POINTS_TYPE.Position:
            point_features = FaceAnalysis.get_features5(points)
        elif point_type == FaceAnalysis.POINTS_TYPE.AllDistances:
            point_features = FaceAnalysis.get_features6(points)

        # Kolory twarzy
        if cheek_color:
            color_features = FaceAnalysis.ColorFeatures(image, points)

        # Zmarszczki
        if wrinkles:
            wrinkle_features = FaceAnalysis.get_wrinkles(image, points)

        features = np.hstack((point_features, color_features, wrinkle_features))

        return features

    # Metoda wykrywa twarz z czarno-bialego i zwraca polozenie lewego gornego rogu
    # z szerokoscia i wysokoscia
    @staticmethod
    def get_face(image):
        face = FaceAnalysis.face_cascade.detectMultiScale(image, 1.3, 5)

        if len(face) == 0:
            raise Exception('No face detected')

        x, y, w, h = face[0]

        return int(x), int(y), int(w), int(h)

    # Metoda pobiera czarno-biale zdjecie i zwraca zestaw 68 punktow charakterystycznych twarzy
    @staticmethod
    def get_landmarks(image):
        try:
            x, y, w, h = FaceAnalysis.get_face(image)  # Pobranie parametrow prostokatu otaczajacego twarz
        except:
            raise Exception('No face detected')
        face = dlib.rectangle(x, y, x + w, y + h)  # Utworzenie prostokatu

        landmarks = FaceAnalysis.predictor(image, face)  # Wyznaczenie punktow

        # Konwersja do numpy.ndarray
        points = np.zeros((68, 2), dtype='int')
        for i in range(0, 68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)

        return points

    # Metoda obraca i wysrodkowuje na podstawie punktow twarz na obrazie
    @staticmethod
    def align(points, image):
        dWidth = 256
        dHeight = 256
        dLeft = (0.37, 0.37)

        # Ustalenie polozenia punktow nalezaoych do oczu
        (leftStart, leftEnd) = (42, 48)
        (rightStart, rightEnd) = (36, 42)

        leftPts = points[leftStart:leftEnd]
        rightPts = points[rightStart:rightEnd]

        # ustalenie srodka kazdego oka
        leftCenter = leftPts.mean(axis=0).astype("int")
        rightCenter = rightPts.mean(axis=0).astype("int")

        dRight = 1.0 - dLeft[0]

        # Ustalenie katu pochylenia prostej przechodzacej przez oczy
        dY = rightCenter[1] - leftCenter[1]
        dX = rightCenter[0] - leftCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Ustalenienie skali przeksztalcenia nowego obrazu
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        dDist = (dRight - dLeft[0])
        dDist *= dWidth
        scale = dDist / dist

        # ustalenie srodka miedzy oczami
        center = ((leftCenter[0] + rightCenter[0]) // 2, (leftCenter[1] + rightCenter[1]) // 2)

        # przeksztalcenie obrazu twarzy
        M = cv2.getRotationMatrix2D(center, angle, scale)

        tX = dWidth * 0.5
        tY = dHeight * dLeft[0]

        M[0, 2] += (tX - center[0])
        M[1, 2] += (tY - center[1])

        (w, h) = (dWidth, dHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return output

    # Metoda pobiera zestaw 68 punktow charakterystycznych twarzy i zwraca
    # wektor cech, skladajacy sie z odleglosci punktow od centrum oraz katu
    # wzgledem centrum
    @staticmethod
    def get_features1(points):
        n_points = len(points)  # Liczba punktow
        features = np.zeros(2 * n_points)  # Wektor cech

        # Znajdywanie centrum punktow
        x_center, y_center = 0, 0
        for x, y in points:
            x_center += x
            y_center += y

        x_center /= n_points
        y_center /= n_points

        # Wyznaczenie odleglosci punktow od centrum
        for i, point in enumerate(points):
            x = point[0]
            y = point[1]

            features[i] = np.sqrt((x_center - x) ** 2 + (y_center - y) ** 2)

        # Wyznaczenie katow wzgledem centrum
        for i, point in enumerate(points):
            x = point[0]
            y = point[1]

            features[i + n_points] = np.arctan2(y - y_center, x - x_center)

        return features

    # Metoda pobiera zestaw 68 punktow charakterystycznych twarzy i zwraca
    # wektor cech, skladajacy sie z odleglosci punktow od centrum oraz katu
    # wzgledem centrum z pominieciem szczeki
    @staticmethod
    def get_features2(points):
        features = FaceAnalysis.get_features1(points)
        n_points = len(points)

        # Usuniecie punktow szczeki
        jaw_points = FaceAnalysis.JAWLINE_POINTS
        features = np.delete(features, jaw_points)
        jaw_points = [x + n_points for x in jaw_points]  # Offset do katow
        features = np.delete(features, jaw_points)

        return features

    # Wyciaganie cech z oczu, brwi i ust jako wektory, niezaleznie od siebie
    @staticmethod
    def get_features3(points):
        # Pobranie punktow oczu, brwi i ust
        right_eye, left_eye = points[FaceAnalysis.RIGHT_EYE_POINTS], points[FaceAnalysis.LEFT_EYE_POINTS]
        right_brow, left_brow = points[FaceAnalysis.RIGHT_EYEBROW_POINTS], points[FaceAnalysis.LEFT_EYEBROW_POINTS]
        mouth_outline, mouth_inner = points[FaceAnalysis.MOUTH_OUTLINE_POINTS], points[FaceAnalysis.MOUTH_INNER_POINTS]
        n_eye_points = len(right_eye)
        n_brow_points = len(right_brow)
        n_mouth_outline_points = len(mouth_outline)
        n_mouth_inner_points = len(mouth_inner)

        features = np.zeros(4 * n_eye_points + 4 * n_brow_points + 2 * n_mouth_outline_points + 2 * n_mouth_inner_points)

        # Znajdywanie centrum oczu
        eye_features = np.zeros(4 * n_eye_points)
        x_righteye_center, y_righteye_center = 0, 0
        x_lefteye_center, y_lefteye_center = 0, 0

        for i in range(n_eye_points):
            x_righteye_center += right_eye[i, 0]
            y_righteye_center += right_eye[i, 1]
            x_lefteye_center += left_eye[i, 0]
            y_lefteye_center += left_eye[i, 1]

        x_righteye_center /= n_eye_points
        y_righteye_center /= n_eye_points
        x_lefteye_center /= n_eye_points
        y_lefteye_center /= n_eye_points

        # Odleglosc i kat punktow oczu od centrum oczu
        for i in range(n_eye_points):
            x_righteye = right_eye[i, 0]
            y_righteye = right_eye[i, 1]
            x_lefteye = left_eye[i, 0]
            y_lefteye = left_eye[i, 1]

            eye_features[i] = np.sqrt((x_righteye_center - x_righteye) ** 2 + (y_righteye_center - y_righteye) ** 2)
            eye_features[i + n_eye_points] = np.sqrt((x_lefteye_center - x_lefteye) ** 2 + (y_lefteye_center - y_lefteye) ** 2)

            eye_features[i + 2 * n_eye_points] = np.arctan2(y_righteye - y_righteye_center, x_righteye - x_righteye_center)
            eye_features[i + 3 * n_eye_points] = np.arctan2(y_lefteye - y_lefteye_center, x_lefteye - x_lefteye_center)

        # Znajdywanie centrum brwi
        brow_features = np.zeros(4 * n_brow_points)
        x_rightbrow_center, y_rightbrow_center = 0, 0
        x_leftbrow_center, y_leftbrow_center = 0, 0

        for i in range(n_brow_points):
            x_rightbrow_center += right_brow[i, 0]
            y_rightbrow_center += right_brow[i, 1]
            x_leftbrow_center += left_brow[i, 0]
            y_leftbrow_center += left_brow[i, 1]

        x_rightbrow_center /= n_brow_points
        y_rightbrow_center /= n_brow_points
        x_leftbrow_center /= n_brow_points
        y_leftbrow_center /= n_brow_points

        # Odleglosc i kat punktow brwi od centrum brwi
        for i in range(n_brow_points):
            x_rightbrow = right_brow[i, 0]
            y_rightbrow = right_brow[i, 1]
            x_leftbrow = left_brow[i, 0]
            y_leftbrow = left_brow[i, 1]

            brow_features[i] = np.sqrt((x_rightbrow_center - x_rightbrow) ** 2 +
                                                     (y_rightbrow_center - y_rightbrow) ** 2)
            brow_features[i + n_brow_points] = np.sqrt((x_leftbrow_center - x_leftbrow) ** 2 +
                                                                     (y_leftbrow_center - y_leftbrow) ** 2)

            brow_features[i + 2 * n_brow_points] = np.arctan2(y_rightbrow - y_rightbrow_center,
                                                                            x_rightbrow - x_rightbrow_center)
            brow_features[i + 3 * n_brow_points] = np.arctan2(y_leftbrow - y_leftbrow_center,
                                                              x_leftbrow - x_leftbrow_center)

        # Znajdywanie centrum ust
        mouth_inner_features, mouth_outline_features = np.zeros(2 * n_mouth_inner_points), \
                                                       np.zeros(2 * n_mouth_outline_points)
        x_mouth_center, y_mouth_center = 0, 0

        for i in range(n_mouth_inner_points):
            x_mouth_center += mouth_inner[i, 0]
            y_mouth_center += mouth_inner[i, 1]

        for i in range(n_mouth_outline_points):
            x_mouth_center += mouth_outline[i, 0]
            y_mouth_center += mouth_outline[i, 1]

        x_mouth_center /= n_mouth_outline_points + n_mouth_inner_points
        y_mouth_center /= n_mouth_outline_points + n_mouth_inner_points

        # Wyznaczanie odleglosci punktow zarysu ust od centrum ust
        for i in range(n_mouth_inner_points):
            x_mouth = mouth_inner[i, 0]
            y_mouth = mouth_inner[i, 1]

            mouth_inner_features[i] = np.sqrt((x_mouth_center - x_mouth) ** 2 + (y_mouth_center - y_mouth) ** 2)
            mouth_inner_features[i + n_mouth_inner_points] = np.arctan2(y_mouth - y_mouth_center,
                                                                        x_mouth - x_mouth_center)

        # Wyznaczanie odleglosci punktow wewnetrznych ust od centrum ust
        for i in range(n_mouth_outline_points):
            x_mouth = mouth_outline[i, 0]
            y_mouth = mouth_outline[i, 1]

            mouth_outline_features[i] = np.sqrt((x_mouth_center - x_mouth) ** 2 + (y_mouth_center - y_mouth) ** 2)
            mouth_outline_features[i + n_mouth_outline_points] = np.arctan2(y_mouth - y_mouth_center,
                                                                            x_mouth - x_mouth_center)

        return np.concatenate((eye_features, brow_features, mouth_outline_features, mouth_inner_features))

    # Recznie wyciagane cechy twarzy
    @staticmethod
    def get_features4(points):
        features = np.zeros(8)

        # Pobranie uzywanych punktow
        mouth_points = points[FaceAnalysis.MOUTH_OUTLINE_POINTS]
        left_eye_points, right_eye_points = points[FaceAnalysis.LEFT_EYE_POINTS], points[FaceAnalysis.RIGHT_EYE_POINTS]
        right_brow, left_brow = points[FaceAnalysis.RIGHT_EYEBROW_POINTS], points[FaceAnalysis.LEFT_EYEBROW_POINTS]
        n_mouth_points = len(mouth_points)
        n_eye_points = len(right_eye_points)
        n_brow_points = len(right_brow)

        # Wysokosc kacikow ust wzgledem centrum ust
        y_leftcorner = mouth_points[0, 1]
        y_rightcorner = mouth_points[6, 1]

        y_mouth_center = 0
        for i in range(n_mouth_points):
            y_mouth_center += mouth_points[i, 1]
        y_mouth_center /= n_mouth_points

        features[0] = y_mouth_center - y_leftcorner
        features[1] = y_mouth_center - y_rightcorner

        # Odleglosc gornej i dolnej wargi od siebie
        y_topmouth = mouth_points[3, 1]
        y_bottommouth = mouth_points[9, 1]

        features[2] = np.abs(y_topmouth - y_bottommouth)

        # Usredniona szerokosc oczu
        x_righteye_center, y_righteye_center = 0, 0
        x_lefteye_center, y_lefteye_center = 0, 0

        for i in range(n_eye_points):
            x_righteye_center += right_eye_points[i, 0]
            y_righteye_center += right_eye_points[i, 1]
            x_lefteye_center += left_eye_points[i, 0]
            y_lefteye_center += left_eye_points[i, 1]

        x_righteye_center /= n_eye_points
        y_righteye_center /= n_eye_points
        x_lefteye_center /= n_eye_points
        y_lefteye_center /= n_eye_points

        distance_right = 0
        distance_left = 0
        for i in range(n_eye_points):
            x_righteye = right_eye_points[i, 0]
            y_righteye = right_eye_points[i, 1]
            x_lefteye = left_eye_points[i, 0]
            y_lefteye = left_eye_points[i, 1]

            distance_right += np.sqrt((x_righteye - x_righteye_center) ** 2 + (y_righteye - y_righteye_center) ** 2)
            distance_left += np.sqrt((x_lefteye - x_lefteye_center) ** 2 + (y_lefteye - y_lefteye_center) ** 2)

        features[3] = distance_right / n_eye_points
        features[4] = distance_left / n_eye_points

        # Usredniona odleglosc brwi od oczu
        distance_right = 0
        distance_left = 0
        for i in range(n_brow_points):
            x_rightbrow = right_brow[i, 0]
            y_rightbrow = right_brow[i, 1]
            x_leftbrow = left_brow[i, 0]
            y_leftbrow = left_brow[i, 1]

            distance_right += np.sqrt((x_rightbrow - x_righteye_center) ** 2 + (y_rightbrow - y_righteye_center) ** 2)
            distance_left += np.sqrt((x_leftbrow - x_lefteye_center) ** 2 + (y_leftbrow - y_lefteye_center) ** 2)

        features[5] = distance_right / n_brow_points
        features[6] = distance_left / n_brow_points

        # Kat pod ktorym znajduja sie brwi
        a1, _ = np.polyfit(left_brow[:, 0], left_brow[:, 1], 1)
        a2, _ = np.polyfit(right_brow[:, 0], right_brow[:, 1], 1)

        features[6] = np.abs(np.arctan(a1) * 180 / np.pi)
        features[7] = np.abs(np.arctan(a2) * 180 / np.pi)

        return features

    # Zestaw cech jako zwykle wspolrzedne punktow
    @staticmethod
    def get_features5(points):
        features = np.zeros(2*len(points))

        for i in range(len(points)):
            x, y = points[i]
            features[2*i] = x
            features[2*i + 1] = y

        return features

    # Zestaw cech jako wszystkie odleglosci miedzy punktami brwi, oczu i ust
    @staticmethod
    def get_features6(points):
        mouth_points = np.concatenate((points[FaceAnalysis.MOUTH_OUTLINE_POINTS], points[FaceAnalysis.MOUTH_INNER_POINTS]))
        left_eye_points = points[FaceAnalysis.LEFT_EYE_POINTS]
        right_eye_points = points[FaceAnalysis.RIGHT_EYE_POINTS]
        left_brow_points = points[FaceAnalysis.LEFT_EYEBROW_POINTS]
        right_brow_points = points[FaceAnalysis.RIGHT_EYEBROW_POINTS]

        n_mouth_points = len(mouth_points)
        n_eye_points = len(left_eye_points)
        n_brow_points = len(left_brow_points)

        mouth_features = np.zeros(int((n_mouth_points*(n_mouth_points-1))/2))

        for i in range(n_mouth_points):
            k = 0
            for j in range(i+1, n_mouth_points - i):
                x1, y1 = mouth_points[i]
                x2, y2 = mouth_points[j]

                mouth_features[i+k] = np.sqrt((x1-x2)**2 + (y1-y2)**2)

                k += 1

        left_eye_features = np.zeros(int((n_eye_points * (n_eye_points - 1)) / 2))
        right_eye_features = np.zeros(int((n_eye_points * (n_eye_points - 1)) / 2))

        for i in range(n_eye_points):
            k = 0
            for j in range(i+1, n_eye_points - i):
                x1, y1 = left_eye_points[i]
                x2, y2 = left_eye_points[j]

                left_eye_features[i + k] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                x1, y1 = right_eye_points[i]
                x2, y2 = right_eye_points[j]

                right_eye_features[i + k] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                k += 1

        left_brow_features = np.zeros(int((n_brow_points * (n_brow_points - 1)) / 2))
        right_brow_features = np.zeros(int((n_brow_points * (n_brow_points - 1)) / 2))

        for i in range(n_brow_points):
            k = 0
            for j in range(i + 1, n_brow_points - i):
                x1, y1 = left_brow_points[i]
                x2, y2 = left_brow_points[j]

                left_brow_features[i + k] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                x1, y1 = right_brow_points[i]
                x2, y2 = right_brow_points[j]

                right_brow_features[i + k] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                k += 1

        return np.concatenate((mouth_features, left_eye_features, right_eye_features, left_brow_features, left_eye_features))

    # Metoda wyciagajaca odcien koloru czola
    def ForeheadColor(image,points):
        nose = points[27]  # wspolrzedne punktu znajdujacego sie na nosie
        hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # zamiana modelu barw z rgb na hsv

        forehead = (nose[0],int(nose[1]*0.75))
        color = hsvIm[forehead[0],forehead[1],0]#
        return color #zwroc odcien koloru czola

    # Metoda wyciagajaca odcien koloru policzka i porownywaniu z kolorem z czola. Poszukiwanie rumiencow
    def ColorFeatures(image,points):
        nose = points[26]  # wspolrzedne punktu znajdujacego sie na czubka nosa
        jaw = points[2]  # wspolrzedne punktu znajdujacego sie na krawedzi twarzy

        forehead_color = FaceAnalysis.ForeheadColor(image,points)
        hsvIm = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # zamiana modelu barw z rgb na hsv
        cheek = (int((nose[0]+jaw[0])/2),nose[1])
        cheek_color = hsvIm[cheek[0],cheek[1],0]

        return int(forehead_color-cheek_color)  # zwroc odcien koloru czola, policzka i ich roznice

    # Metoda wyciagajaca zmarszczki z czola
    # Pobiera obraz z obrocona i wysrodkowana twarza, oraz punkty char. tej  twarzy
    @staticmethod
    def get_wrinkles(aligned, points):
        # Punkty brwi
        left_brow = points[FaceAnalysis.LEFT_EYEBROW_POINTS]
        right_brow = points[FaceAnalysis.RIGHT_EYEBROW_POINTS]

        # Wyznaczenie punktu na brwiach od ktorego bedzie wyznaczane czolo
        x_left_bottom, y_left_bottom = right_brow[2][0], right_brow[2][1]
        x_right_bottom, y_right_bottom = left_brow[2][0], left_brow[2][1]

        # Wyciecie prostokatu zawierajacego czolo
        distance_from_brow = 3
        forehead_height = 30
        forehead = aligned[y_left_bottom - distance_from_brow - forehead_height:y_right_bottom - distance_from_brow,
                           x_left_bottom:x_right_bottom, :]

        # Wyznaczenie krawedzi
        gray = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 300, 800, apertureSize=5)

        # Wyznaczenie linii prostych z krawedzi
        min_line_length = 10
        max_line_gap = 5
        threshold = 20
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, min_line_length, max_line_gap)

        # Licznik zmarszczek poziomych
        horitontal_wrinkles_counter = 0

        # Jezeli wykryto linie
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if -20 < angle < 20:
                    horitontal_wrinkles_counter += 1

        # Wyciecie prostokatu miedzy brwiami
        glabella = aligned[y_left_bottom - 25:y_right_bottom + 10, x_left_bottom - 10:x_right_bottom + 10, :]

        # Wyznaczanie krawedzi
        gray = cv2.cvtColor(glabella, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100, apertureSize=3)

        # Wyznaczanie linii prostych z krawedzi
        minLineLength = 10
        maxLineGap = 1
        threshold = 10
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)

        # Licznik zmarszczek pionowych
        vertical_wrinkles_counter = 0

        # Jezli wykryto linie
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if 80 < angle < 100 or -100 < angle < -80:
                    vertical_wrinkles_counter += 1

        return np.hstack((horitontal_wrinkles_counter, vertical_wrinkles_counter))
