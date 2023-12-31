# libs
import os, cv2, warnings
from tqdm import tqdm
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def load_data(path):
    
    images = []
    labels=[]

    # Load images from 2 folders (horse and human)
    for filename in os.listdir(path):
        print(f'Loading {filename} images...')
        # Load imagem from current folder
        for types in tqdm(os.listdir(os.path.join(path, filename))):
            img = cv2.imread(os.path.join(path, filename, types))
            if img is not None:
                labels.append(filename)
                images.append(img)
                
    return images, labels
   
def preprocess(images):

    Image_transformed = []

    for img in tqdm(images):
        # Convert the image in RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the image in gray scale
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        Image_transformed.append(img)
        
    return Image_transformed

def extract_features(images):

    features=[]

    for image in tqdm(images):
            # Apply canny edge detector
            edges = cv2.Canny(image, 100, 250)

            # Invert the image, edge black, background white
            invertida = cv2.bitwise_not(edges)
            
            # Extract HOG features (we got the params online)
            fd, hog_image = hog(invertida, orientations = 8, pixels_per_cell = (16, 16), cells_per_block = (1, 1), visualize = True)

            features.append(fd)

    return features
  
def TrainRF(features, labels):
    
    # Define the parameter grid
    parameters = {'criterion': ['gini', 'entropy', 'log_loss'],
               'max_features':['auto','sqrt','log2'],
               'n_estimators': [100, 250, 500],}

    # Create a random forest classifier
    rf_model = RandomForestClassifier()

    # Instantiate the grid search model
    grid_search = GridSearchCV(rf_model, parameters)

    grid_search.fit(features, labels)

    best_params = grid_search.best_params_

    print("Best parameters found:", best_params)
    # The best parameters found are: {'criterion': 'log_loss', 'max_features': 'log2', 'n_estimators': 500}

    best_model = grid_search.best_estimator_

    return best_model

def Main():
    # Maybe wont work in linux
    trainPath = '.\horse-or-human\\train'
    testPath = '.\horse-or-human\\validation'

    print("[INFO] Starting Training...")
    print("[INFO] Getting images...")
    images, labels = load_data(trainPath)

    print("[INFO] Preprocessing images...")
    image_preprocessed = preprocess(images)

    print("[INFO] Extracting features...")
    trainFeatures = extract_features(image_preprocessed)

    print("[INFO] Training RF...")
    rf = TrainRF(trainFeatures, labels)
    

    print("[INFO] Starting Testing...")
    print("[INFO] Getting test images...")
    testImages, testLabels = load_data(testPath)

    print("[INFO] Preprocessing test images...")
    image_preprocessed = preprocess(testImages)

    print("[INFO] Extracting test features...")
    testFeatures = extract_features(image_preprocessed)

    print("[INFO] Predicting...")
    predictions = rf.predict(testFeatures)

    print("[INFO] Accuracy: ", accuracy_score(testLabels, predictions))
    
    cm = confusion_matrix(testLabels, predictions)

    # Plot confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=False,
                                class_names=['Horse', 'Human'])

    
    plt.title('Matriz de Confusão')
    plt.xlabel('Rótulo Predito')
    plt.ylabel('Rótulo Real')

    plt.show()

if __name__ == '__main__':
    Main()
