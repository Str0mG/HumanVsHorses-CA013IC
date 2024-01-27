## Project Title

### Classification of Humans and Horses in Images

## Team

- Breno Cesar Dupin - 1827642
- Gabriel S. Trombini - 2209926

## Dataset

- [Horses or Humans Dataset](https://www.kaggle.com/sanikamal/horses-or-humans-dataset)

## Project Description

The "Classification of Humans and Horses in Images" project aims to create a classification model capable of distinguishing images of humans and horses. This could have practical applications as the model could be applied in the livestock industry to identify and track animals, especially horses, on farms, stables, and events related to animal breeding. One of the potential limitations could be the size and diversity of the dataset and the repertoire of knowledge about classification algorithms.

## Project Repository

- [GitHub](https://github.com/Str0mG/HumanVsHorses-CA013IC)
- [Video(PT-br)](https://youtu.be/)

## Classifier and Accuracy

- Using Random Forest, with hyperparameters obtained through GridSearchCV, achieving an accuracy of 84.375%.

![Confusion Matrix](https://github.com/Str0mG/HumanVsHorses-CA013IC/blob/main/GraficoAcuracia.png)

## Installation

To run this project, follow these steps:

- Clone project of GitHub. Open you terminal and run the following command:

  ```bash
  git clone https://github.com/Str0mG/CaptchaSolver.git
  ```

- [Dowload DataSet Here!](https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset)

- Remove the folder "horse-or-human" inside the main folder because it is repetitive.

1. Set up a virtual environment. In Windows, open your terminal and run the following command:

   ```bash
   python -m venv env
   ```

2. Install the required packages from the project by running the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the project by executing the following command in Windows:

   ```bash
   python src/Canny_HOG_RF.py
   ```
