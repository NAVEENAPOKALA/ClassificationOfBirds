# Import required packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Read the images (By default images are being read in BGR format)
image0 = cv2.imread('Data/image0.jpg')
image1 = cv2.imread('Data/image1.jpg')
image2 = cv2.imread('Data/image2.jpg')

################### Plotting Blue, Green, Red and Gray scale images ##############

# Split the image into its BGR channels
b0, g0, r0 = cv2.split(image0)
b1, g1, r1 = cv2.split(image1)
b2, g2, r2 = cv2.split(image2)


# Create a figure with 9 subplots
fig, axs = plt.subplots(3, 4, figsize=(10, 5))

# Display the red channel for all images
axs[0,0].set_title('Red Channel')
axs[0,0].imshow(r0)
axs[1,0].imshow(r1)
axs[2,0].imshow(r2)

# Display the green channel for all images
axs[0,1].set_title('Green Channel')
axs[0,1].imshow(g0)
axs[1,1].imshow(g1)
axs[2,1].imshow(g2)

# Display the blue channel for all images
axs[0,2].set_title('Blue Channel')
axs[0,2].imshow(b0)
axs[1,2].imshow(b1)
axs[2,2].imshow(b2)

# Convert the image to grayscale
gray_image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Display the grayscale images
axs[0,3].set_title('Grayscale Image')
axs[0,3].imshow(gray_image0, cmap=plt.get_cmap('gray'))
axs[1,3].imshow(gray_image1, cmap=plt.get_cmap('gray'))
axs[2,3].imshow(gray_image2, cmap=plt.get_cmap('gray'))
plt.show()


# Calculating the height and width of the grayscale images

height_image0, width_image0 = gray_image0.shape
height_image1, width_image1 = gray_image1.shape
height_image2, width_image2 = gray_image2.shape

# Print the dimensions
print("Height and Width of image0: ", height_image0, width_image0);
print("Height and Width of image1: ", height_image1, width_image1);
print("Height and Width of image2: ", height_image2, width_image2);

# Function to resize the image 
def resizeImage(image):
    return cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

# Making a function call to reduce the dimensions
gray_image0 = resizeImage(gray_image0)
gray_image1 = resizeImage(gray_image1)
gray_image2 = resizeImage(gray_image2)

# Calculating the height and width of the new reduced size grayscale images
height_image0, width_image0 = gray_image0.shape
height_image1, width_image1 = gray_image1.shape
height_image2, width_image2 = gray_image2.shape

#Print the dimensions
print("Resized height and width of image0",height_image0, width_image0)
print("Resized height and width of image1",height_image1, width_image1)
print("Resized height and width of image2",height_image2, width_image2)


################# Creating the feature Vectors for all the images ####################


#Creating feature vectors of size 256 for gray_image0
cc = round(((height_image0)*(width_image0))/256)
flat_gray_image0 = np.zeros((cc, 257), np.uint8)
k=0
for i in range(0,height_image0,16):
    for j in range(0,width_image0,16):
        gray_image0_tmp = gray_image0[i:i+16,j:j+16]
        flat_gray_image0[k,0:256] = gray_image0_tmp.flatten()
        k = k + 1
        
fspaceimg0 = pd.DataFrame(flat_gray_image0)

noOfObservations=len(fspaceimg0)

print("no of observations for image0 : ",noOfObservations)

print("dimensions for image0 : ",fspaceimg0.shape)

#Creating feature vectors of size 256 for gray_image1

cc = round(((height_image1)*(width_image1))/256)
flat_gray_image1 = np.ones((cc, 257), np.uint8)
k=0
for i in range(0,height_image1,16):
    for j in range(0,width_image1,16):
        gray_image1_tmp = gray_image1[i:i+16,j:j+16]
        flat_gray_image1[k,0:256] = gray_image1_tmp.flatten()
        k = k + 1
        
fspaceimg1 = pd.DataFrame(flat_gray_image1) 

noOfObservations=len(fspaceimg1)

print("no of observations for image1 : ",noOfObservations)

print("dimensions for image1 : ",fspaceimg1.shape)

#Creating feature vectors of size 256 for gray_image2

cc = round(((height_image2)*(width_image2))/256)
flat_gray_image2 = np.full((cc, 257), 2)
k=0
for i in range(0,height_image2,16):
    for j in range(0,width_image2,16):
        gray_image2_tmp = gray_image2[i:i+16,j:j+16]
        flat_gray_image2[k,0:256] = gray_image2_tmp.flatten()
        k = k + 1
        
fspaceimg2 = pd.DataFrame(flat_gray_image2)

noOfObservations=len(fspaceimg2)

print("no of observations for image2 : ",noOfObservations)

print("dimensions for image2 : ",fspaceimg2.shape)

fspace=[fspaceimg0,fspaceimg1,fspaceimg2]
featurespace = pd.concat(fspace)

#Creating sliding feature vectors of size 256 for gray_image0

cc = round(((height_image0-15)*(width_image0-15)))
slide_gray_image0 = np.zeros((cc, 257), np.uint8)
k=0
for i in range(0,height_image0-15,1):
    for j in range(0,width_image0-15,1):
        gray_image0_tmp = gray_image0[i:i+16,j:j+16]
        slide_gray_image0[k,0:256] = gray_image0_tmp.flatten()
        k = k + 1
        
fspaceSlideGrayImage0 = pd.DataFrame(slide_gray_image0) #panda object data frame
fspaceSlideGrayImage0.to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/Birds/Data/Spreadsheets/slide_gray_image0.csv',
index=False)

#Creating sliding feature vectors of size 256 for gray_image1
cc = round(((height_image1-15)*(width_image1-15)))
slide_gray_image1 = np.ones((cc, 257), np.uint8)
k=0
for i in range(0,height_image1-15,1):
    for j in range(0,width_image1-15,1):
        gray_image1_tmp = gray_image1[i:i+16,j:j+16]
        slide_gray_image1[k,0:256] = gray_image1_tmp.flatten()
        k = k + 1
        
fspaceSlideGrayImage1 = pd.DataFrame(slide_gray_image1) #panda object data frame
fspaceSlideGrayImage1.to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/Birds/Data/Spreadsheets/slide_gray_image1.csv',
index=False)

#Creating sliding feature vectors of size 256 for gray_image2

cc = round(((height_image2-15)*(width_image2-15)))
slide_gray_image2 = np.full((cc, 257), 2)
k=0
for i in range(0,height_image2-15,1):
    for j in range(0,width_image2-15,1):
        gray_image2_tmp = gray_image2[i:i+16,j:j+16]
        slide_gray_image2[k,0:256] = gray_image2_tmp.flatten()
        k = k + 1
        
fspaceSlideGrayImage2 = pd.DataFrame(slide_gray_image2)


###################### Data Preprocessing/ Analyzing the data #######################

# Calculating no of observations 

noOfObservations=len(featurespace)

print("Total no of observations : ",noOfObservations)

print("dimensions of the total feature space : ",featurespace.shape)

# # Calculating mean
flat_gray_image0_mean = flat_gray_image0[:,0:256].mean(axis=0)
flat_gray_image1_mean = flat_gray_image1[:,0:256].mean(axis=0)
flat_gray_image2_mean = flat_gray_image2[:,0:256].mean(axis=0)
plt.title("Mean")
plt.plot(flat_gray_image0_mean,'.')
plt.plot(flat_gray_image0_mean)
plt.plot(flat_gray_image1_mean,'x')
plt.plot(flat_gray_image1_mean)
plt.plot(flat_gray_image2_mean,'+')
plt.plot(flat_gray_image2_mean)
plt.show()
print("Mean of flat_gray_image0 is: ", flat_gray_image0_mean)
print("Mean of flat_gray_image1 is: ", flat_gray_image1_mean)
print("Mean of flat_gray_image2 is: ", flat_gray_image2_mean)


# Calculating standard deviation
flat_gray_image0_std = flat_gray_image0[:,0:256].std(axis=0)
flat_gray_image1_std = flat_gray_image1[:,0:256].std(axis=0)
flat_gray_image2_std = flat_gray_image2[:,0:256].std(axis=0)
plt.title("Standard Deviation")
plt.plot(flat_gray_image0_std,'.')
plt.plot(flat_gray_image0_std)
plt.plot(flat_gray_image1_std,'x')
plt.plot(flat_gray_image1_std)
plt.plot(flat_gray_image2_std,'+')
plt.plot(flat_gray_image2_std)
plt.show()
print("Standard Deviation of flat_gray_image0 is: ", flat_gray_image0_std)
print("Standard Deviation of flat_gray_image1 is: ", flat_gray_image1_std)
print("Standard Deviation of flat_gray_image2 is: ", flat_gray_image2_std)


#Merge feature vectors img0 and img1
frames01 = [fspaceimg0, fspaceimg1]
mged01 = pd.concat(frames01)

#Merge feature vectors img0, img1 and img2
frames012 = [fspaceimg0, fspaceimg1, fspaceimg2]
mged012 = pd.concat(frames012)

#Randomizing the vectors
df01 = mged01.sample(frac = 1)

#Randomizing the vectors
df012 = mged012.sample(frac = 1)

#Merge slide feature vectors img0 and img1 and randomizing them
framesSlide01 = [fspaceSlideGrayImage0, fspaceSlideGrayImage1]
mgedframesSlide01 = pd.concat(framesSlide01)
dfS01 = mgedframesSlide01.sample(frac = 1)

#Merge slide feature vectors img0, img1, img2 and randomizing them
framesSlide012 = [fspaceSlideGrayImage0, fspaceSlideGrayImage1, fspaceSlideGrayImage2]
mgedframesSlide012 = pd.concat(framesSlide012)
dfS012 = mgedframesSlide012.sample(frac = 1)

############################ Training the model ###########################

# Split the block vectors of image0,1 into 80:20
row, col = df01.shape
TR = round(row*0.8)
TT = row-TR

X01_train = df01.iloc[0:TR,0:256]
Y01_train = df01.iloc[0:TR,256]
X01_test = df01.iloc[TR:row,0:256]
Y01_test = df01.iloc[TR:row,256]

# Standardizing the data
scaler = StandardScaler()
X01_train_Standard = scaler.fit_transform(X01_train)
X01_test_Standard = scaler.fit_transform(X01_test)

X01_train_Standard = pd.DataFrame(X01_train_Standard)
X01_test_Standard = pd.DataFrame(X01_test_Standard)

# Applying PCA
cmp = 30
pca1 = PCA(n_components=cmp)
X01_train_PCA = pca1.fit_transform(X01_train_Standard)
X01_test_PCA = pca1.fit_transform(X01_test_Standard)

# Resulted Train and Test data after applying PCA
X01_train_PCA = pd.DataFrame(X01_train_PCA)
X01_test_PCA = pd.DataFrame(X01_test_PCA)


# Splie block vectors of image0,1 ,2 data into 80:20
row, col = df012.shape
TR = round(row*0.8)
TT = row-TR

X012_train = df012.iloc[0:TR,0:256]
Y012_train = df012.iloc[0:TR,256]
X012_test = df012.iloc[TR:row,0:256]
Y012_test = df012.iloc[TR:row,256]

# Standardizing the data
scaler = StandardScaler()
X012_train_Standard = scaler.fit_transform(X012_train)
X012_test_Standard = scaler.fit_transform(X012_test)

X012_train_Standard = pd.DataFrame(X012_train_Standard)
X012_test_Standard = pd.DataFrame(X012_test_Standard)

# Applying PCA
pca2 = PCA(n_components=cmp)
X012_train_PCA = pca2.fit_transform(X012_train_Standard)
X012_test_PCA = pca2.fit_transform(X012_test_Standard)

# Resulted Train and Test data after applying PCA
X012_train_PCA = pd.DataFrame(X012_train_PCA)
X012_test_PCA = pd.DataFrame(X012_test_PCA)


# Split the Sliding vectors of image0,1 data into 80:20
row, col = dfS01.shape
TR = round(row*0.8)
TT = row-TR

XS01_train = dfS01.iloc[0:TR,0:256]
YS01_train = dfS01.iloc[0:TR,256]
XS01_test = dfS01.iloc[TR:row,0:256]
YS01_test = dfS01.iloc[TR:row,256]

# Standardizing the data

scaler = StandardScaler()
XS01_train_Standard = scaler.fit_transform(XS01_train)
XS01_test_Standard = scaler.fit_transform(XS01_test)

XS01_train_Standard = pd.DataFrame(XS01_train_Standard)
XS01_test_Standard = pd.DataFrame(XS01_test_Standard)

# Applying PCA
pca3 = PCA(n_components=cmp)
XS01_train_PCA = pca3.fit_transform(XS01_train_Standard)
XS01_test_PCA = pca3.fit_transform(XS01_test_Standard)

# Resulted Train and Test data after applying PCA
XS01_train_PCA = pd.DataFrame(XS01_train_PCA)
XS01_test_PCA = pd.DataFrame(XS01_test_PCA)


# Split the Sliding vectors of image0,1,2 data into 80:20
row, col = dfS012.shape
TR = round(row*0.8)
TT = row-TR

XS012_train = dfS012.iloc[0:TR,0:256]
YS012_train = dfS012.iloc[0:TR,256]
XS012_test = dfS012.iloc[TR:row,0:256]
YS012_test = dfS012.iloc[TR:row,256]

# Standardizing the data

scaler = StandardScaler()
XS012_train_Standard = scaler.fit_transform(XS012_train)
XS012_test_Standard = scaler.fit_transform(XS012_test)

XS012_train_Standard = pd.DataFrame(XS012_train_Standard)
XS012_test_Standard = pd.DataFrame(XS012_test_Standard)

# Applying PCA
pca4 = PCA(n_components=cmp)
XS012_train_PCA = pca4.fit_transform(XS012_train_Standard)
XS012_test_PCA = pca4.fit_transform(XS012_test_Standard)

# Resulted Train and Test data after applying PCA
XS012_train_PCA = pd.DataFrame(XS012_train_PCA)
XS012_test_PCA = pd.DataFrame(XS012_test_PCA)


########## plotting the features in all categories to see if they follow same distribution

fig, axs = plt.subplots(2, 4, figsize=(25, 10))

means = X01_train_PCA.mean()
axs[0,0].set_title('training set(block) of images 0,1 after using PCA')
axs[0,0].set_xlabel('Features')
axs[0,0].set_ylabel('Mean Value')
axs[0,0].plot(means.index, means.values)

means = X01_test_PCA.mean()
axs[0,1].set_title('test set(block) of images 0,1 after using PCA')
axs[0,1].set_xlabel('Features')
axs[0,1].set_ylabel('Mean Value')
axs[0,1].plot(means.index, means.values)

means = X012_train_PCA.mean()
axs[0,2].set_title('training set(block) of images 0,1,2 after using PCA')
axs[0,2].set_xlabel('Features')
axs[0,2].set_ylabel('Mean Value')
axs[0,2].plot(means.index, means.values)

means = X012_test_PCA.mean()
axs[0,3].set_title('test set(block) of images 0,1,2 after using PCA')
axs[0,3].set_xlabel('Features')
axs[0,3].set_ylabel('Mean Value')
axs[0,3].plot(means.index, means.values)

means = XS01_train_PCA.mean()
axs[1,0].set_title('training set(Slide) of images 0,1 after using PCA')
axs[1,0].set_xlabel('Features')
axs[1,0].set_ylabel('Mean Value')
axs[1,0].plot(means.index, means.values)

means = XS01_test_PCA.mean()
axs[1,1].set_title('test set(Slide) of images 0,1 after using PCA')
axs[1,1].set_xlabel('Features')
axs[1,1].set_ylabel('Mean Value')
axs[1,1].plot(means.index, means.values)

means = XS012_train_PCA.mean()
axs[1,2].set_title('training set(Slide) of images 0,1,2 after using PCA')
axs[1,2].set_xlabel('Features')
axs[1,2].set_ylabel('Mean Value')
axs[1,2].plot(means.index, means.values)

means = XS012_test_PCA.mean()
axs[1,3].set_title('training set(Slide) of images 0,1,2 after using PCA')
axs[1,3].set_xlabel('Features')
axs[1,3].set_ylabel('Mean Value')
axs[1,3].plot(means.index, means.values)

plt.show()

fig, axs = plt.subplots(2, 4, figsize=(25, 10))

var = X01_train_PCA.var()
axs[0,0].set_title('training set(block) of images 0,1 after using PCA')
axs[0,0].set_xlabel('Features')
axs[0,0].set_ylabel('Variance Value')
axs[0,0].plot(var.index, var.values)

var = X01_test_PCA.var()
axs[0,1].set_title('test set(block) of images 0,1 after using PCA')
axs[0,1].set_xlabel('Features')
axs[0,1].set_ylabel('Variance Value')
axs[0,1].plot(var.index, var.values)

var = X012_train_PCA.var()
axs[0,2].set_title('training set(block) of images 0,1,2 after using PCA')
axs[0,2].set_xlabel('Features')
axs[0,2].set_ylabel('Variance Value')
axs[0,2].plot(var.index, var.values)

var = X012_test_PCA.var()
axs[0,3].set_title('test set(block) of images 0,1,2 after using PCA')
axs[0,3].set_xlabel('Features')
axs[0,3].set_ylabel('Variance Value')
axs[0,3].plot(var.index, var.values)

var = XS01_train_PCA.var()
axs[1,0].set_title('training set(Slide) of images 0,1 after using PCA')
axs[1,0].set_xlabel('Features')
axs[1,0].set_ylabel('Variance Value')
axs[1,0].plot(var.index, var.values)

var = XS01_test_PCA.var()
axs[1,1].set_title('test set(Slide) of images 0,1 after using PCA')
axs[1,1].set_xlabel('Features')
axs[1,1].set_ylabel('Variance Value')
axs[1,1].plot(var.index, var.values)

var = XS012_train_PCA.var()
axs[1,2].set_title('training set(Slide) of images 0,1,2 after using PCA')
axs[1,2].set_xlabel('Features')
axs[1,2].set_ylabel('Variance Value')
axs[1,2].plot(var.index, var.values)

var = XS012_test_PCA.var()
axs[1,3].set_title('training set(Slide) of images 0,1,2 after using PCA')
axs[1,3].set_xlabel('Features')
axs[1,3].set_ylabel('Variance Value')
axs[1,3].plot(var.index, var.values)

plt.show()

##############plotting histograms
fig, axs = plt.subplots(2, 4, figsize=(25, 10))

axs[0,0].set_title('training set(block) images 0,1 after using PCA')
axs[0,0].set_xlabel('Features 1 and 3')
axs[0,0].hist(X01_train_PCA[[1,3]], bins=30)

axs[0,1].set_title('test set(block) images 0,1 after using PCA')
axs[0,1].set_xlabel('Features 1 and 3')
axs[0,1].hist(X01_test_PCA[[1,3]], bins=30)

axs[0,2].set_title('training set(block) of images 0,1,2 after using PCA')
axs[0,2].set_xlabel('Features 1 and 3')
axs[0,2].hist(X012_train_PCA[[1,3]], bins=30)

axs[0,3].set_title('test set(block) of images 0,1,2 after using PCA')
axs[0,3].set_xlabel('Features 1 and 3')
axs[0,3].hist(X012_test_PCA[[1,3]], bins=30)

axs[1,0].set_title('training set(Slide) of images 0,1 after using PCA')
axs[1,0].set_xlabel('Features 1 and 3')
axs[1,0].hist(XS01_train_PCA[[1,3]], bins=30)

axs[1,1].set_title('test set(Slide) of images 0,1 after using PCA')
axs[1,1].set_xlabel('Features 1 and 3')
axs[1,1].hist(XS01_test_PCA[[1,3]], bins=30)

axs[1,2].set_title('training set(Slide) of images 0,1,2 after using PCA')
axs[1,2].set_xlabel('Features 1 and 3')
axs[1,2].hist(XS012_train_PCA[[1,3]], bins=30)

axs[1,3].set_title('test set(Slide) of images 0,1,2 after using PCA')
axs[1,3].set_xlabel('Features 1 and 3')
axs[1,3].hist(XS012_test_PCA[[1,3]], bins=30)

plt.show()

########## training the model using elastic net regression as two class classifier block feature 01

eln=ElasticNet(alpha=0.835)

#training model for block feature vectors of image 0 and image 1
modelLR01 = eln.fit(X01_train_PCA, Y01_train)
y01_predict_PCA = modelLR01.predict(X01_test_PCA)
y01_predict_PCA = y01_predict_PCA.round()

dfB01_PCA = pd.DataFrame(X01_test_PCA)
dfB01_PCA['Actual']=Y01_test.values
dfB01_PCA['Predict']=y01_predict_PCA


#training model for slide feature vectors of image 0 and image 1
modelLRS01 = eln.fit(XS01_train_PCA, YS01_train)
ys01_predict_PCA = modelLRS01.predict(XS01_test_PCA)
ys01_predict_PCA = ys01_predict_PCA.round()

dfS01_PCA = pd.DataFrame(XS01_test)
dfS01_PCA['Actual']=YS01_test.values
dfS01_PCA['Predict']=ys01_predict_PCA


#constructing confusion matrices for block feature vectors of image0 and image1 for elastic regression model
print("Classification Report for Elasticnet for block features of image0 and image1 after using PCA:")
print(classification_report(Y01_test, y01_predict_PCA))
CC_testLR = confusion_matrix(Y01_test, y01_predict_PCA, labels=[0,1])
print("Confusion Matrix for Elasticnet for block features of image0 and image1 after using PCA")
print(CC_testLR)

TN = CC_testLR[1,1]
FP = CC_testLR[1,0]
FN = CC_testLR[0,1]
TP = CC_testLR[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Qualitative Measures for Elasticnet for block features of image0 and image1 after using PCA")
print("Accuracy_Score :",Accuracy)
Precision = 1/(1+(FP/TP))
print("Precision_Score :",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Sensitivity_Score :",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Specificity_Score :",Specificity)


#constructing confusion matrices for slide feature vectors of image0 and image1 for elastic regression model
print("Classification Report for Elasticnet for slide features of image0 and image1 after using PCA:")
print(classification_report(YS01_test, ys01_predict_PCA))
CC_testLR = confusion_matrix(YS01_test, ys01_predict_PCA, labels=[0,1])
print("Confusion Matrix for Elasticnet for slide features of image0 and image1 after using PCA")
print(CC_testLR)

TN = CC_testLR[1,1]
FP = CC_testLR[1,0]
FN = CC_testLR[0,1]
TP = CC_testLR[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Qualitative Measures for Elasticnet for slide features of image0 and image1 after using PCA")
print("Accuracy_Score :",Accuracy)
Precision = 1/(1+(FP/TP))
print("Precision_Score :",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Sensitivity_Score :",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Specificity_Score :",Specificity)


########################## training the model using random forest ###########################

rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1
)

#training the model for block feature vectors of image0 and image1
modelRF = rf.fit(X01_train_PCA, Y01_train)
y_predictRF01_PCA = modelRF.predict(X01_test_PCA)
y_predictRF01_PCA = y_predictRF01_PCA.round()

dfB01_RF_PCA = pd.DataFrame(X01_test_PCA)
dfB01_RF_PCA['Actual']=Y01_test.values
dfB01_RF_PCA['Predict']=y_predictRF01_PCA

#constructing confusion matrices for block feature vectors of image0 and image1
CC_testRF = confusion_matrix(Y01_test, y_predictRF01_PCA, labels=[0,1])
print("Confusion Matrix For Random Forest for block features of image0 and image1 after using PCA")
print(CC_testRF)

TN = CC_testRF[1,1]
FP = CC_testRF[1,0]
FN = CC_testRF[0,1]
TP = CC_testRF[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score for RF :",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our_Precision_Score for RF :",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score for RF :",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score for RF :",Specificity)


#training the model for Slide feature vectors of image0 and image1
modelRF = rf.fit(XS01_train_PCA, YS01_train)
y_predictRFS01_PCA = modelRF.predict(XS01_test_PCA)
y_predictRFS01_PCA = y_predictRFS01_PCA.round()

dfS01_RF_PCA = pd.DataFrame(XS01_test)
dfS01_RF_PCA['Actual']=YS01_test.values
dfS01_RF_PCA['Predict']=y_predictRFS01_PCA

#constructing confusion matrices for block feature vectors of image0 and image1
CC_testRF = confusion_matrix(YS01_test, y_predictRFS01_PCA, labels=[0,1])
print("Confusion Matrix For Random Forest for slide features of image0 and image1 after using PCA")
print(CC_testRF)

TN = CC_testRF[1,1]
FP = CC_testRF[1,0]
FN = CC_testRF[0,1]
TP = CC_testRF[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score for RF :",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our_Precision_Score for RF :",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score for RF :",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score for RF :",Specificity)



#training the model for block feature vectors of image0, image1 and image2

modelRF = rf.fit(X012_train_PCA, Y012_train)
y_predictRF012_PCA = modelRF.predict(X012_test_PCA)
y_predictRF012_PCA = y_predictRF012_PCA.round()

dfB012_RF_PCA = pd.DataFrame(X012_test_PCA)
dfB012_RF_PCA['Actual']=Y012_test.values
dfB012_RF_PCA['Predict']=y_predictRF012_PCA

#constructing confusion matrices for block feature vectors of image0 and image1
CC_testRF = confusion_matrix(Y012_test, y_predictRF012_PCA, labels=[0,1,2])
print("Confusion Matrix For Random Forest for block features of image0 and image1, image2 after using PCA")
print(CC_testRF)

TN = CC_testRF[1,1]
FP = CC_testRF[1,0]
FN = CC_testRF[0,1]
TP = CC_testRF[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score for RF :",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our_Precision_Score for RF :",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score for RF :",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score for RF :",Specificity)



#training the model for slide feature vectors of image0, image1 and image2
modelRF = rf.fit(XS012_train_PCA, YS012_train)
y_predictRFS012_PCA = modelRF.predict(XS012_test_PCA)
y_predictRFS012_PCA = y_predictRFS012_PCA.round()

dfS012_RF_PCA = pd.DataFrame(XS012_test_PCA)
dfS012_RF_PCA['Actual']=YS012_test.values
dfS012_RF_PCA['Predict']=y_predictRFS012_PCA

#constructing confusion matrices for block feature vectors of image0 and image1
CC_testRF = confusion_matrix(YS012_test, y_predictRFS012_PCA, labels=[0,1,2])
print("Confusion Matrix For Random Forest for Slide features of image0 and image1, image2 after using PCA")
print(CC_testRF)

TN = CC_testRF[1,1]
FP = CC_testRF[1,0]
FN = CC_testRF[0,1]
TP = CC_testRF[0,0]
FPFN = FP+FN
TPTN = TP+TN
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score for RF :",Accuracy)
Precision = 1/(1+(FP/TP))
print("Our_Precision_Score for RF :",Precision)
Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score for RF :",Sensitivity)
Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score for RF :",Specificity)
