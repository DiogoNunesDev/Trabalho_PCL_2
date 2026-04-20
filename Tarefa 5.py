import os
import matplotlib.pyplot as plt
import seaborn as sns

FILE_GOLD = "eval-labels.txt"
FILE_QUESTIONS = "eval-questions.txt"

FILES_PRED = {
    "T2 - Unigramas": "Tarefa2-results/result_unigrams.txt",
    "T2 - Bigramas": "Tarefa2-results/result_bigrams.txt",
    "T2 - Smoothing": "Tarefa2-results/result_smooth.txt",
    "T3 - Unigramas": "Tarefa3-results/result_unigrams.txt",
    "T3 - Bigramas": "Tarefa3-results/result_bigrams.txt",
    "T3 - Smoothing": "Tarefa3-results/result_smooth.txt",
    "T4 - LLM Zero-Shot": "Tarefa4-results/result_zero_full.txt",
    "T4 - LLM Few-Shot": "Tarefa4-results/result_few_full.txt"
}

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]


def carregar_ficheiro_limpo(filepath):
    if not os.path.exists(filepath):
        print(f"\n[Aviso] Ficheiro não encontrado: {filepath}")
        return []

    linhas_limpas = []
    with open(filepath, "r", encoding="utf-8") as f:
        for linha in f.readlines():
            linha_formatada = linha.strip().upper()
            if linha_formatada in VALID_LABELS:
                linhas_limpas.append(linha_formatada)
            elif linha_formatada:
                pass

    return linhas_limpas


def carregar_perguntas(filepath):
    if not os.path.exists(filepath):
        print(f"\n[Aviso] Ficheiro de perguntas não encontrado: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        return [linha.strip() for linha in f.readlines() if linha.strip()]


def gerar_imagem_matriz(nome_modelo, matrix):
    tarefa_id = nome_modelo.split()[0]
    numero_tarefa = tarefa_id.replace("T", "")

    pasta_destino = f"Tarefa{numero_tarefa}_matrizes_confusão"
    os.makedirs(pasta_destino, exist_ok=True)

    array_matriz = []
    for real in VALID_LABELS:
        linha = [matrix[real][previsto] for previsto in VALID_LABELS]
        array_matriz.append(linha)

    plt.figure(figsize=(8, 6))
    sns.heatmap(array_matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[c[:4] for c in VALID_LABELS],
                yticklabels=[c[:4] for c in VALID_LABELS])

    plt.title(f"Matriz de Confusão: {nome_modelo}", pad=15, fontsize=14)
    plt.ylabel("Real", fontweight='bold')
    plt.xlabel("Previsto", fontweight='bold')
    plt.tight_layout()

    nome_ficheiro = nome_modelo.replace(" - ", "_").replace("/", "_").replace(" ", "_") + ".png"
    caminho_completo = os.path.join(pasta_destino, nome_ficheiro)

    plt.savefig(caminho_completo, dpi=300)
    plt.close()

    print(f"Imagem gerada: {caminho_completo}")


def comparar(nome_modelo, golds, preds, questions):
    if len(golds) != len(preds):
        print(f"\n[Erro em {nome_modelo}] Tamanhos diferentes! (Reais: {len(golds)} | Previstos: {len(preds)}).")
        return

    matrix = {real: {previsto: 0 for previsto in VALID_LABELS} for real in VALID_LABELS}
    acertos = 0
    erros_extensos = []

    for real, previsto, pergunta in zip(golds, preds, questions):
        matrix[real][previsto] += 1
        if real == previsto:
            acertos += 1
        else:
            erros_extensos.append((real, previsto, pergunta))

    percentagem = (acertos / len(golds)) * 100

    print("\n" + "=" * 65)
    print(f"RESULTADOS: {nome_modelo.upper()}")
    print(f"Precisão: {percentagem:.2f}% ({acertos} corretas em {len(golds)})")
    print("=" * 65)

    header = f"{'Real \\ Prev':<12} | " + " | ".join([f"{c[:4]:<4}" for c in VALID_LABELS])
    print(header)
    print("-" * len(header))

    for real in VALID_LABELS:
        row_str = f"{real:<12} | " + " | ".join([f"{matrix[real][p]:<4}" for p in VALID_LABELS])
        print(row_str)

    erros = []
    for r in VALID_LABELS:
        for p in VALID_LABELS:
            if r != p and matrix[r][p] > 0:
                erros.append((r, p, matrix[r][p]))

    erros.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTOP 3 ERROS ({nome_modelo}):")
    for r, p, contagem in erros[:3]:
        print(f" -> Era {r:<10} mas previu {p:<10} ({contagem} vezes)")

    gerar_imagem_matriz(nome_modelo, matrix)


def main():
    print("A carregar as labels e perguntas reais...")
    golds = carregar_ficheiro_limpo(FILE_GOLD)
    questions = carregar_perguntas(FILE_QUESTIONS)

    if not golds or not questions:
        print(f"Falha a carregar os ficheiros originais. Confirma que tens {FILE_GOLD} e {FILE_QUESTIONS} na pasta.")
        return

    if len(golds) != len(questions):
        print(f"Aviso: Número de labels ({len(golds)}) não bate certo com número de perguntas ({len(questions)}).")

    print(f"Sucesso! Encontradas {len(golds)} perguntas.")

    for nome, ficheiro in FILES_PRED.items():
        preds = carregar_ficheiro_limpo(ficheiro)
        if preds:
            comparar(nome, golds, preds, questions)


if __name__ == "__main__":
    main()