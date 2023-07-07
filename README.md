## Projeto Final de Processamento de Imagens

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
- [Video](https://drive.google.com/file/d/15BFsliGAEfBtdO-s8K7THxhgNXONJPtn/view?usp=sharing)

## Classicador e Acurácia

- Usando Ramdom Forest, na qual os hiperparâmetros foram obtidos por meio de GridSearchCV, obtendo uma acurácia de 84.375%.

![Gráfico Acurácia](https://github.com/Str0mG/HumanVsHorses-CA013IC/blob/main/results/GraficoAcuracia.png?raw=true)

Com base no gráfico, explicaremos mais sobre o código 'canny hog RF' cuja Matriz de Confusão é a seguinte:

![Matriz de Confusão](https://github.com/Str0mG/HumanVsHorses-CA013IC/blob/main/results/Canny_HOG_RF.png?raw=true)
## Instalação

Para executar este projeto, siga estes passos:

- Clone o projeto do GitHub. Abra o seu terminal e execute o seguinte comando:

  ```bash
  git clone https://github.com/Str0mG/HumanVsHorses-CA013IC
  ```

- [Baixe o dataset aqui!](https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset)

- Remova a pasta "horse-or-human" dentro da pasta principal, pois ela é repetitiva.

1. Configure um ambiente virtual. No Windows, abra o seu terminal e execute o seguinte comando:

   ```bash
   python -m venv env
   ```

2. Instale os pacotes necessários do projeto executando o seguinte comando:

   ```bash
   pip install -r requirements.txt
   ```

3. Execute o projeto executando o seguinte comando no Windows:

   ```bash
   python src/Canny_HOG_RF.py
   ```
