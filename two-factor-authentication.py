# OpenCV (Open Source Computer Vision Library) is an open-source library that includes several hundreds of computer vision algorithms.
# Its package includes several shared or static libraries like image processing, video analysis, object detection.
import cv2
# NumPy is a library  adding support for large, multi-dimensional arrays and matrices,
# along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np
# Used to recognize faces from python. It is built using dlib (open source)
# which is modern C++ toolkit which contains ML algos. We’ll need to enable CUDA lib when compiling dlib.
# Accuracy: 99.38%
import face_recognition
# The functions OS module allows us to operate on underlying Operating System tasks, irrespective of platform.
import os
# The sys module in Python provides various functions and variables that are used to manipulate different parts of the Python runtime environment.
# It allows operating on the interpreter.
import sys
# It lets you start new applications right from the Python program you are currently writing.
import subprocess

# This function is used to check if the name entered by the user is a verifed user or not by checking in the CSV file.


def searchName(name):
    # r+ =read only mode
    with open('encode.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
        # Append all the names in the array and check if entered name is in the registered list and return accordingly.
        nameList.append(entry[0])
        if name in nameList:
            return True
        else:
            return False

# To convert the image into an array of integers scaled between -1 to 1 of size 128.


def fetchEncoding(images):
    encodeList = []
    for img in images:
        # opencv read: BGR , convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # List of face encodings to compare
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


if __name__ == '__main__':
    try:
        # Path where all the registerd users images are stored.
        path = 'auth_face'

        images = []
        classNames = []
        myList = os.listdir(path)  # fetch images

        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)  # Add image in images list.
            # Name in classNames list
            classNames.append(os.path.splitext(cl)[0])

        # Convert all the images into their encoding of 128 dimensions.
        encodingComp = fetchEncoding(images)

        nameAuth = input('Enter your name > ')
        #password = input('Enter Password > ')

        cap = cv2.VideoCapture(0)  # webcam:on

 # If the name entered by user is present in the verified list of users then check the face with the help of camera
        if searchName(nameAuth):
            try:
                while True:
                    # Open the webcam and check if the face in front of webcam matches with the face stored in the database.
                    success, img = cap.read()
                    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                # Face Locations gives a 2D array with a box around the human face. It uses CNN.
                # Each face location is a tuple of pixel positions for (top, right, bottom, left)
                    facesCurFrame = face_recognition.face_locations(imgS)

                    encodesCurFrame = face_recognition.face_encodings(
                        imgS, facesCurFrame)

                    flag = 1
                #
                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        try:
                            matches = face_recognition.compare_faces(
                                encodingComp, encodeFace)
    # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    # for each comparison face. The distance tells you how similar the faces are.
                            faceDis = face_recognition.face_distance(
                                encodingComp, encodeFace)
                            # returns indices of minimum value of array in part. axis
                            matchIndex = np.argmin(faceDis)
                            if matches[matchIndex]:
                                print("Identified as " + nameAuth)
                                flag = 1
                                password = int(input("Enter password > "))
                                if(password == 29112002):
                                    print("User is authenticated")
                            # sys.exit(0)
                                    subprocess.call(
                                        ['python', 'run_dataset.py'])
                                else:
                                    print("Wrong password!")
                                    sys.exit(0)

                            else:
                                print(
                                    "User is not authenticated to access simulation!")

                                cv2.putText(img, 'Identity could not be confirmed! ', (50, 30),
                                            cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (255, 0, 0), 2)
                                flag = 0

                        except Exception as e:
                            print(f"ERROR: {e}")
                            sys.exit(0)

                    if flag == 0:
                        break

                    cv2.imshow('Webcam', img)
                    cv2.waitKey(5)  # 5 milliseconds window will pop up

            except Exception as e:
                print(f"ERROR: {e}")
                sys.exit(0)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(0)
