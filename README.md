## Titulo do Projeto

### Classificação de humanos e cavalos em imagens

## Equipe

- Breno Cesar Dupin - 1827642
- Gabriel S. Trombini - 2209926

## Dataset

- [Horses or Humans Dataset](https://www.kaggle.com/sanikamal/horses-or-humans-dataset)

## Descrição do Projeto

O projeto "Classificação de humanos e cavalos em imagens" tem como objetivo criar um modelo de classificação capaz de distinguir imagens de humanos e cavalos. Na qual pode ter aplicações praticas uma vez que o modelo pode ser aplicado na indústria pecuária para identificar e rastrear animais, especialmente cavalos, em fazendas, estábulos e eventos relacionados à criação de animais.
Uma das possiveis limitações seriam o tamanho e diversidade do conjunto de dados e o repertório de conhecimento sobre algoritmos de classificações.

## Repositório do projeto

- [GitHub](https://github.com/Str0mG/HumanVsHorses-CA013IC)
- [Video](https://youtu.be/)

## Classicador e Acurácia

- Usando Ramdom Forest, na qual os hiperparâmetros foram obtidos por meio de GridSearchCV, obtendo uma acurácia de 84.375%.

![Matriz de confusão](https://github.com/Str0mG/HumanVsHorses-CA013IC/blob/main/GraficoAcuracia.png)

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
