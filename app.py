from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import os
import tempfile
import joblib

#Imports colab
from transformers import pipeline
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dateparser
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permite requisições de outros domínios

#Dados de treino
dados = {
    'documento': [
        "O fornecedor cumpriu todos os padrões de segurança sem falhas.",  # Baixa
        "O contrato foi renovado e está em conformidade com as normas.",  # Baixa
        "A auditoria não encontrou nenhum problema nos relatórios técnicos.",  # Baixa
        "O documento apresenta falhas graves nos registros de segurança.",  # Alta
        "O contrato está expirado e nenhuma renovação foi feita.",  # Alta
        "Há pequenos atrasos na entrega de documentos, mas sem impacto na segurança.",  # Méia
        "Alguns documentos estão incompletos, mas o fornecedor está corrigindo.",  #
        "A auditoria encontrou algumas irregularidades, porém sem riscos críticos."  #
    ],
    'rótulo': ['baixa', 'baixa', 'baixa', 'alta', 'alta', 'média', 'média', 'média']
}

#treinando
vetorizador = CountVectorizer()
X = vetorizador.fit_transform(dados['documento'])
y = dados['rótulo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_naive = MultinomialNB()
modelo_naive.fit(X_train, y_train)

#extrair texto de PDFs
def extrair_texto_pdf(caminho_arquivo):
    with pdfplumber.open(caminho_arquivo) as pdf:
        texto = ''
        for pagina in pdf.pages:
            texto += pagina.extract_text()
    return texto

#modelo BERT para classificar o texto no contexto
def classificar_contexto_bert(texto):
    classifier = pipeline("zero-shot-classification", model="bert-base-multilingual-cased")
    classes = ["baixa prioridade", "média prioridade", "alta prioridade"]
    resultado = classifier(texto, classes)
    return resultado['labels'][0]  # Classe com maior pontuação

#analise de sentimento (para pergar contexto das palavras chaves)
def analisar_sentimento(texto):
    blob = TextBlob(texto)
    sentimento = blob.sentiment.polarity
    if sentimento > 0:
        return "positivo"
    elif sentimento < 0:
        return "negativo"
    else:
        return "neutro"
    
# Função para extrair datas do texto
def extrair_datas(texto):
    # Padrão para capturar datas como "07/06/2024", "07-06-2024", e "07 jun 2024"
    padrao_data = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b|\b(\d{1,2}\s(?:jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\s\d{4})\b'
    datas_encontradas = re.findall(padrao_data, texto)
    # A função re.findall retorna uma lista de tuplas, precisamos filtrar os resultados
    datas_formatadas = [data[0] if data[0] else data[1] for data in datas_encontradas]
    return datas_formatadas

# Função para verificar validade das datas com correção de comparação
def verificar_validade(datas):
    if not datas:
        return "sem data"

    data_atual = datetime.now()
    status_validade = "dentro do prazo"

    for data in datas:
        if isinstance(data, str):
            try:
                # Usar o dateparser para interpretar a data
                data_formatada = dateparser.parse(data)

                # Se a data foi corretamente interpretada
                if data_formatada:
                    # Comparar com a data atual
                    if data_formatada < data_atual:
                        return "vencido"
            except Exception as e:
                continue  # Ignorar erros de interpretação

    return status_validade

#classificar prioridade com Naive Bayes (palavras chaves)
def classificar_prioridade_naive(texto, vetorizar, modelo):
    vetor_texto = vetorizador.transform([texto])
    return modelo.predict(vetor_texto)[0]

def classificar_contexto_bert(texto):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    classes = ["baixa prioridade", "média prioridade", "alta prioridade"]
    resultado = classifier(texto, classes)
    return resultado['labels'][0]  # Classe com maior pontuação

# Função para processar o PDF e extrair informações
def analyze_document(file_path):
    texto_documento = extrair_texto_pdf(file_path)

    # Mostrar o nome do arquivo PDF
    print(f"\nAnalisando o arquivo: {file_path}")

    # BERT (contexto)
    classificacao_bert = classificar_contexto_bert(texto_documento)
    print(f"Classificação BERT: {classificacao_bert}")

    # Sentimento
    sentimento_documento = analisar_sentimento(texto_documento)
    print(f"Sentimento do documento: {sentimento_documento}")

    # Naive Bayes (palavras)
    classificacao_naive = classificar_prioridade_naive(texto_documento, vetorizador, modelo_naive)
    print(f"Classificação Naive Bayes: {classificacao_naive}")

    # Verificação de Data
    datas_extraidas = extrair_datas(texto_documento)
    validade = verificar_validade(datas_extraidas)
    print(f"Validade do documento: {validade}")

    # Combinação de classificações
    if classificacao_bert == "alta prioridade" or classificacao_naive == "alta" or sentimento_documento == "negativo":
        classificacao_final = "alta prioridade"
    elif classificacao_bert == "baixa prioridade" and classificacao_naive == "baixa" and sentimento_documento == "positivo":
        classificacao_final = "baixa prioridade"
    else:
        classificacao_final = "média prioridade"

    print(f"Classificação Final: {classificacao_final}")
    
    final = {
        "Bert":classificacao_bert,   
        "sentimento":sentimento_documento, 
        "Naive":classificacao_naive, 
        "validadeDoc":validade,
        "resultadoFinal": classificacao_final
             }
    
    return final  # Retorna apenas os primeiros 500 caracteres como exemplo

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    files = request.files.getlist('files')  # Obtenha todos os arquivos
    results = []

    for file in files:
        if file.filename == '':
            continue

        # Crie um diretório temporário para salvar o arquivo
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Chame a função de análise
        analysis_result = analyze_document(file_path)
        results.append({"filename": file.filename, "conteudo": analysis_result})

    # Retorne o resultado de todos os arquivos
    return jsonify({"resultados": results})

if __name__ == "__main__":
    app.run(debug=True)