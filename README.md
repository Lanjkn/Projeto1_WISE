Projeto realizado para o programa de Trainee da Wise, ministrado pelos Tutores Rodrigo Quisen, Alberto Levi e Marcos Colodi.
A finalidade do projeto é analisar minhas habilidades em machine learning e ver o quão bem consigo aplicar o conteúdo ensinado através dos cursos.

O Projeto se resume a sanitização de dados e gerar modelos de classificação para a coluna "SITUAÇÃO", fazendo predições de situações de Churn ou não dependendo dos dados do cliente e também clusterização de clientes de acordo com suas features.

Os modelos supervisionados utilizados são: RandomForestClassifier, LogisticRegressor e MLPClassifier, todos do pacote ScikitLearn.

O modelo não-supervisionado utilizado é o KMeans.

Certas colunas dos dados foram removidas por sua baixa importância nas análises de features, podendo assim criar um modelo muito mais otimizado sem sacrificar sua eficiência.

Requisitos:

REQ-01 - Criar um diretório para organizar os arquivos do projeto, contendo ao menos os seguintes
itens:

● Um arquivo “README.md” contendo o descritivo do projeto e as instruções para o preparo do
ambiente; ✔️

● Um script python “main.py” para o processamento dos dados; ✔️

● O dataset utilizado; ✔️

● Um arquivo “requirements.txt” para incluir as dependências do projeto (bibliotecas utilizadas e
suas versões). 

REQ-02 - Versionar o diretório do projeto em um repositório privado no Github e inserir os tutores
como colaboradores: quisen; mcolodi; AlbertoLevi. ✔️

REQ-03 - Gerar um documento para visualização gráfica dos dados utilizando o Sweetviz ou o
Pandas Profiling e versionar os outputs. O objetivo da utilização dessas ferramentas é a identificação
de características de cada coluna e verificar quais atributos possuem valores inválidos, nulos ou
vazios para realizar a sanitização dos dados. ✔️

Para a geração dos modelos dos REQ-04, REQ-05 e REQ-06 utilizar a biblioteca Scikit-learn e
considerar com o label o atributo “SITUACAO” do dataset fornecido. ✔️

REQ-04 - Implementar funções para o treinamento e geração de um modelo utilizando Random
Forest. ✔️

REQ-05 - Implementar funções para o treinamento e geração de um modelo utilizando Logistic
Regression. ✔️

REQ-06 - Implementar funções para o treinamento e geração de um modelo utilizando Multilayer
Perceptron. ✔️

REQ-07 - Com o intuito de determinar a qualidade do modelo gerado, mplementar uma maneira
programática de avaliar e visualizar as métricas (Accuracy; F1; Precision; Recall) do modelo de
Random Forest. ✔️

REQ-08 - Com o intuito de determinar a qualidade do modelo gerado, mplementar uma maneira
programática de avaliar e visualizar as métricas (Accuracy; F1; Precision; Recall) do modelo de
Logistic Regression. ✔️

REQ-09 - Com o intuito de determinar a qualidade do modelo gerado, mplementar uma maneira
programática de avaliar e visualizar as métricas (Accuracy; F1; Precision; Recall) do modelo de
Multilayer Perceptron. ✔️

REQ-10 - Implementar o algoritmo K-means gerando a clusterização dos clientes presentes no
dataset. Analisar e apresentar as principais características dos clusteres ✔️

Descrição do Projeto:
A empresa “Fake-Dental”, que comercializa planos odontológicos, entrou em contato com a Wise Intelligence buscando integrar soluções de machine learning em seus sistemas.

Um dos principais problemas apresentados teve origem no setor de retenção de clientes da empresa. Esse setor é responsável por entrar em contato com os clientes para obter informações a respeito da sua satisfação com relação aos serviços e ao plano contratado.

Visto que existe uma grande quantidade de clientes ativos no banco de dados, a tarefa de definir quais são os clientes que devem ser priorizados para o contato dos operadores não ocorre de forma simples. Portanto é necessário, a partir dos dados dos clientes atuais, a geração de algum recurso que possa auxiliar no direcionamento estratégico da empresa.
