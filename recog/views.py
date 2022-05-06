from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import cv2

from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from . import cascade as casc

from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import json
from faceRecog.settings import BASE_DIR

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
import os
from PIL import Image
from django.contrib import messages
from aadhar_detection import Detector

# Create your views here.
def index(request):
    return render(request, 'index.html')
def errorImg(request):
    return render(request, 'error.html')


def home(request):

	return render(request, 'recog/home2.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recog/index.html')
	else:
		print("not admin")

		return render(request,'recog/index.html')

@login_required
def not_authorised(request):
	return render(request,'recog/not_authorised.html')



def create_dataset(request):
    #print request.POST
    userId = request.POST['userId']
    print (cv2.__version__)
   
    
    
    cam = cv2.VideoCapture(0)

    
    id = userId
    # Our dataset naming counter
    
    while(True):
        ret, img = cam.read()
        card = roi_extract(img)
        cv2.imshow("Card",card)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            cv2.imwrite(BASE_DIR+'/ml/aadhar/user.'+str(id)+'.jpg', card)
            print("{} written!".format('user.'+str(id)+'.jpg'))
            
    #releasing the cam
    
    # destroying all the windows
    cv2.destroyAllWindows()
    
    status,cardimage = card_verify(BASE_DIR+'/ml/aadhar/user.'+str(id)+'.jpg')
    print('xxxxxx',status)
    if status==True:
        cv2.imshow('card', cardimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        messages.success(request, 'Card authenticated successfully')
        
    else:
        cv2.imshow('card', cardimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        messages.error(request, 'Card Authentication Failed')
        return redirect('/dashboard')

    cam = cv2.VideoCapture(0)
    
    
    # Detect face
    #Creating a cascade image classifier
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')
    #camture images from the webcam and process and detect the face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture(0)

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    id = userId
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        #cam.read will return the status variable and the captured colored image
        ret, img = cam.read()
        #the returned img is a colored image but for the classifier to work we need a greyscale image
        #to convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #To store the faces
        #This will detect all the images in the current frame, and it will return the coordinates of the faces
        #Takes in image and some other parameter for accurate result
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        #In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        for(x,y,w,h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1
            # Saving the image dataset, but only the face part, cropping the rest
            cv2.imwrite(BASE_DIR+'/ml/dataset/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Face",img)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        #To get out of the loop
        if(sampleNum>35):
            break
    #releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()

    return redirect('/dashboard')
    
    
def roi_extract(img):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold =  cv2.threshold(imgGrey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(threshold,kernel,iterations = 3)

    contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = max(contours, key=lambda x: cv2.contourArea(x)) #max contours finding using area
        x, y, w, h = cv2.boundingRect(contours)
        cv2.rectangle(img,(x,y),(x+w,y+h),[0,0,255],1)
        crop_img=img[y+1:y+h,x+1:x+w]  #crop the area
        crop_img=cv2.resize(crop_img,(325,204))
    return crop_img
    

def card_verify(card_path):
    image = cv2.imread(card_path)
    detector = Detector( path_config='models/research/object_detection/pre_trained_models/pipeline.config', path_ckpt='models/research/object_detection/checkpoint/saved_model/ckpt-5', path_to_labels="models/research/object_detection/label_map.pbtxt" )
    image, original_image, coordinate_dict,check = detector.predict(image)    
    return check,image


def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''


    #Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Path of the samples
    path = BASE_DIR+'/ml/dataset'

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        # create a list for the path for all the images that is available in the folder
        # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
        #print imagePaths

        # Now, we loop all the images and store that userid and the face with different image list
        faces = []
        Ids = []
        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            faceImg = Image.open(imagePath).convert('L') #convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            faceNp = np.array(faceImg, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to split the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hence have to convert into int format
            ID = os.path.split(imagePath)[-1].split('.')[1] # -1 so that it will count from backwards and split the second index of the '.' Hence id
            # Images
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            #print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
            label_encoder = LabelEncoder()
            Ids_enc = label_encoder.fit_transform(Ids)
            Ids_enc_dict = {i: str(j) for i, j in zip(Ids, Ids_enc)}
            with open('encoded_names.json', 'w') as fp:
                json.dump(Ids_enc_dict, fp)
        return np.array(Ids_enc), np.array(faces)

    # Fetching ids and faces
    ids, faces = getImagesWithID(path)

    #Training the recognizer
    # For that we need face samples and corresponding labels
    recognizer.train(faces, ids)

    # Save the recogzier state so that we can access it later
    recognizer.save(BASE_DIR+'/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('/dashboard')

def get_liveness_model():

    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3),
                    activation='relu',
                    input_shape=(24,100,100,1)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')

    model = get_liveness_model()

    model.load_weights("model.h5")
    print("Loaded model from disk")

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(BASE_DIR+'/ml/recognizer/trainingData.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0


    # Initialize some variables

    input_vid = []


    while(True):
        if len(input_vid) < 24:
            ret, img = cam.read()
            liveimg = cv2.resize(img, (100, 100))
            liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
            input_vid.append(liveimg)

        else:
            ret, img = cam.read()
            liveimg = cv2.resize(img, (100, 100))
            liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)
            input_vid.append(liveimg)
            inp = np.array([input_vid[-24:]])
            inp = inp / 255
            inp = inp.reshape(1, 24, 100, 100, 1)
            pred = model.predict(inp)
            input_vid = input_vid[-25:]

            if pred[0][0] > .99:
                with open('encoded_names.json') as fp:
                    Ids_enc_dict = json.load(fp)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceDetect.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    getId, conf = rec.predict(gray[y:y + h, x:x + w])  # This will predict the id of the face
                    for name, id in Ids_enc_dict.items():
                        if id == str(getId):
                            getId = name
                            break
                    # print conf;
                    if conf < 60:
                        userId = getId
                        cv2.putText(img , 'ID : '+str(userId), (x, y + h), font, 2, (0, 255, 0), 2)
                        
                    else:
                        cv2.putText(img, "Unknown", (x, y + h), font, 2, (0, 0, 255), 2)

                    # Printing that number below the face
                    # @Prams cam image, id, location,font style, color, stroke

            else:
                cv2.putText(img, 'WARNING!', (img.shape[1] // 2, img.shape[0] // 2), font, 1.0, (255, 255, 255), 1)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif (userId != 'Unknown' and userId!=0):
                print('Success!')
                cv2.destroyAllWindows()
                cam.release()
                #cv2.putText(img , str(userId), (x, y + h), font, 2, (0, 255, 0), 2)
                cv2.imshow('Video', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                return redirect('/records/details/' + str(userId))
            elif cv2.waitKey(1) & 0xFF == ord('d'):
                cam.release()
                cv2.destroyAllWindows()
                return render(request, 'Unidentified.html')

            # Display the liveness score in top left corner
            cv2.putText(img, str(pred[0][0]), (20, 20), font, 1.0, (255, 255, 0), 1)
            # Display the resulting image
            cv2.imshow('Video', img)

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/dashboard')

def eigenTrain(request):
    path = BASE_DIR+'/ml/dataset'

    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w= df.getImagesWithID(path)
    print ('features'+str(faces.shape[1]))
    # Spliting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    #print ">>>>>>>>>>>>>>> "+str(y_test.size)
    n_classes = y_test.size
    target_names = ['Sudarshan Paul','Ben Afflek']
    n_components = 15
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()

    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ",y_pred)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the gallery of the most significative eigenfaces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    # plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    svm_pkl_filename = BASE_DIR+'/ml/serializer/svm_classifier.pkl'
    # Open the file to save as pkl file
    svm_model_pkl = open(svm_pkl_filename, 'wb')
    pickle.dump(clf, svm_model_pkl)
    # Close the pickle instances
    svm_model_pkl.close()



    pca_pkl_filename = BASE_DIR+'/ml/serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    plt.show()

    return redirect('/dashboard')


def detectImage(request):
    userImage = request.FILES['userImage']

    svm_pkl_filename =  BASE_DIR+'/ml/serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    #print "Loaded SVM model :: ", svm_model

    pca_pkl_filename =  BASE_DIR+'/ml/serializer/pca_state.pkl'

    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    #print pca

    '''
    First Save image as cv2.imread only accepts path
    '''
    im = Image.open(userImage)
    #im.show()
    imgPath = BASE_DIR+'/ml/uploadedImages/'+str(userImage)
    im.save(imgPath, 'JPEG')

    '''
    Input Image
    '''
    try:
        inputImg = casc.facecrop(imgPath)
        inputImg.show()
    except :
        print("No face detected, or image not recognized")
        return redirect('/error_image')

    imgNp = np.array(inputImg, 'uint8')
    #Converting 2D array into 1D
    imgFlatten = imgNp.flatten()
    #print imgFlatten
    #print imgNp
    imgArrTwoD = []
    imgArrTwoD.append(imgFlatten)
    # Applyting pca
    img_pca = pca.transform(imgArrTwoD)
    #print img_pca

    pred = svm_model.predict(img_pca)
    print(svm_model.best_estimator_)
    print(pred[0])

    return redirect('/records/details/'+str(pred[0]))
