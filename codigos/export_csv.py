import csv
import time

def salvar_dados_csv(fps_atual, num_pessoas_detectadas,resolucao, canais, nome_arquivo):
    tempo_atual = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    dados = [fps_atual, num_pessoas_detectadas, resolucao, canais, tempo_atual]
    with open(nome_arquivo, 'a', newline='') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv)
        escritor_csv.writerow(dados)