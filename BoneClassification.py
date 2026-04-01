from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

main = tkinter.Tk()
main.title("CLASSIFICATION OF FRACTURED BONES USING MACHINE LEARNING")
main.geometry("1300x1200")

filename = None
X = None
Y = None
X_train = None
X_test = None
y_train = None
y_test = None
classifier = None

labels = ['Chest', 'Elbow', 'Finger', 'Hand', 'Head', 'Shoulder', 'Wrist']

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    if not filename:
        return
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
        
def featuresExtraction():
    text.delete('1.0', END)
    global X, Y, filename
    
    dataset_dir = filename if filename else 'Dataset'
    if not os.path.exists(dataset_dir):
        messagebox.showerror("Error", f"Dataset directory '{dataset_dir}' not found. Please upload dataset.")
        return
        
    text.insert(END, f"Extracting features from '{dataset_dir}'...\n")
    main.update_idletasks()
    
    X_list = []
    Y_list = []
    
    for i, label in enumerate(labels):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.exists(label_dir):
            continue
            
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_resized = cv2.resize(img, (32, 32))
            im2arr = np.array(img_resized).astype('float32') / 255.0
            
            X_list.append(im2arr.ravel())
            Y_list.append(i)
            
    if len(X_list) == 0:
        messagebox.showerror("Error", "No valid images found in the dataset directory.")
        return
        
    X = np.asarray(X_list)
    Y = np.asarray(Y_list)
    
    if not os.path.exists('model'):
        os.makedirs('model')
    np.save('model/X.txt.npy', X)
    np.save('model/Y.txt.npy', Y)

    text.insert(END,"Available Bones images in dataset is : "+str(labels)+"\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total available features in each image : "+str(X.shape[1])+"\n")
    text.insert(END,"Features Extraction Complete! Showing sample image for 3 seconds...\n")
    main.update_idletasks()
    
    test_img = X[0].reshape(32,32,3)
    test_img = cv2.resize(test_img,(250,250))
    test_img_disp = (test_img * 255).astype(np.uint8)
    cv2.imshow("Sample Image from dataset after features extraction", test_img_disp)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

def trainTestGenerator():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    if X is None or Y is None:
        messagebox.showerror("Error", "Please run Features Extraction first.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total images used to train Random Forest : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total images used to test Random Forest : "+str(X_test.shape[0])+"\n")

def randomForest():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    global classifier
    if X_test is None or y_test is None:
        messagebox.showerror("Error", "Please run Train & Test Data Generator first.")
        return
    if os.path.exists('model/model.txt'):
        with open('model/model.txt', 'rb') as file:
            classifier = pickle.load(file)
    else:
        classifier = RandomForestClassifier(n_estimators=200, random_state=0)
        classifier.fit(X_train, y_train)
        with open('model/model.txt', 'wb') as file:
            pickle.dump(classifier, file)
            
    pred = classifier.predict(X_test)
    random_acc = accuracy_score(y_test,pred)*100
    text.insert(END,"Random Forest Bone Classification Accuracy on Test Images : "+str(random_acc)+"\n\n")
    text.insert(END,"Random Forest Bone Classification Report on Test Images\n\n")
    text.insert(END,str(classification_report(y_test, pred))+"\n")


def predict():
    global classifier
    if classifier is None:
        messagebox.showerror("Error", "Please build or load the Random Forest model first.")
        return
    filename = filedialog.askopenfilename(initialdir="testImages")
    if not filename:
        return
    img = cv2.imread(filename)
    if img is None:
        messagebox.showerror("Error", "Failed to load image.")
        return
    img_resized = cv2.resize(img, (32,32))
    im2arr = np.array(img_resized)
    im2arr = im2arr.reshape(32,32,3)
    test_img = np.asarray(im2arr)
    test_img = test_img.astype('float32')
    test_img = test_img/255
    test_img = test_img.ravel()
    imgData = []
    imgData.append(test_img)
    test_data = np.asarray(imgData)
    print(test_data.shape)
    predicted_class = classifier.predict(test_data)[0]
    print(predicted_class)

    display_img = cv2.imread(filename)
    display_img = cv2.resize(display_img, (400,400))
    window_name = 'Bone Classified as : ' + labels[int(predicted_class)]
    cv2.putText(display_img, window_name, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    


font = ('times', 16, 'bold')
title = Label(main, text='CLASSIFICATION OF FRACTURED BONES USING MACHINE LEARNING')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Dataset Collection or Upload", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

featuresButton = Button(main, text="Features Extraction", command=featuresExtraction)
featuresButton.place(x=700,y=200)
featuresButton.config(font=font1)

traintestButton = Button(main, text="Train & Test Data Generator", command=trainTestGenerator)
traintestButton.place(x=700,y=250)
traintestButton.config(font=font1) 

rfButton = Button(main, text="Build Random Forest Model", command=randomForest)
rfButton.place(x=700,y=300)
rfButton.config(font=font1)

predictButton = Button(main, text="Upload Test Image & Classify Bone", command=predict)
predictButton.place(x=700,y=350)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='RoyalBlue2')
main.mainloop()
