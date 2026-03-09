import numpy as np
import pandas as pd

np.random.seed(42)

# Vocabulário: exatamente as 4 palavras da nossa frase
df_vocabulario = pd.DataFrame({
    "palavra": ["O",  "Enzo", "é",  "programador"],
    "id":      [ 0,       1,   2,             3  ],
})

print("Vocabulário:")
print(df_vocabulario.to_string(index=False))

palavra_para_id  = dict(zip(df_vocabulario["palavra"], df_vocabulario["id"]))
tamanho_vocab    = len(df_vocabulario)   # 4

# Frase de entrada convertida para IDs
frase  = ["O", "Enzo", "é", "programador"]
ids    = [palavra_para_id[p] for p in frase]
print(f"\nFrase : {frase}")
print(f"IDs   : {ids}")

# Hiperparâmetros
d_model = 64    # tamanho do vetor de cada palavra (paper usa 512)
d_ff    = 256   # tamanho interno da rede FFN      (paper usa 2048)

#Embedding: cada palavra vira um vetor de 64 números
tabela_embeddings = np.random.randn(tamanho_vocab, d_model) * 0.01

# Tensor X: (1 frase, 4 palavras, 64 números)
X = tabela_embeddings[ids][np.newaxis, :, :]
print(f"\nTensor X: {X.shape}  → (1 frase, 4 palavras, 64 números por palavra)")

def softmax(Z):
    """Transforma números em probabilidades que somam 1."""
    e = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def normalizar_camada(X, epsilon=1e-6):
    """Mantém os números numa escala estável (média≈0, desvio≈1)."""
    media     = np.mean(X, axis=-1, keepdims=True)
    variancia = np.var(X,  axis=-1, keepdims=True)
    return (X - media) / np.sqrt(variancia + epsilon)


class MecanismoDeAtencao:
    """
    Faz cada palavra 'prestar atenção' nas outras.
    Fórmula: softmax( Q·Kᵀ / √d_k ) · V
    """
    def __init__(self, d_model):
        self.WQ  = np.random.randn(d_model, d_model) * 0.01
        self.WK  = np.random.randn(d_model, d_model) * 0.01
        self.WV  = np.random.randn(d_model, d_model) * 0.01
        self.d_k = d_model

    def calcular(self, X):
        Q, K, V      = X @ self.WQ, X @ self.WK, X @ self.WV
        pontuacoes   = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        return softmax(pontuacoes) @ V


class RedeNeuralFeedForward:
    """
    Processa cada palavra individualmente após a atenção.
    Expande de 64 → 256 (com ReLU) → 64 números.
    Fórmula: max(0, x·W1 + b1)·W2 + b2
    """
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)   * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff,   d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def calcular(self, X):
        return np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2


class CamadaEncoder:
    """
    Um bloco completo do Encoder. Fluxo:
      1. Atenção(X)
      2. LayerNorm(X + resultado)       
      3. FFN(resultado)
      4. LayerNorm(resultado + FFN)     
    """
    def __init__(self, d_model, d_ff):
        self.atencao = MecanismoDeAtencao(d_model)
        self.ffn     = RedeNeuralFeedForward(d_model, d_ff)

    def processar(self, X):
        X = normalizar_camada(X + self.atencao.calcular(X))
        X = normalizar_camada(X + self.ffn.calcular(X))
        return X

N_CAMADAS = 6
encoder   = [CamadaEncoder(d_model, d_ff) for _ in range(N_CAMADAS)]

print(f"\nPassando '{' '.join(frase)}' pelas {N_CAMADAS} camadas do Encoder...")
print(f"Entrada           : {X.shape}")

tensor_atual = X.copy()
for i, camada in enumerate(encoder):
    tensor_atual = camada.processar(tensor_atual)
    print(f"Saída da Camada {i+1} : {tensor_atual.shape}")

Z = tensor_atual


# VALIDAÇÃO DE SANIDADE
assert Z.shape == (1, len(frase), d_model), "Erro no formato do tensor!"

print(f"\n Formato correto: {Z.shape}   (1 frase, {len(frase)} palavras, {d_model} números)")
print(f"\nVetor Z — representação final de cada palavra:")
for i, palavra in enumerate(frase):
    valores = [f"{v:.4f}" for v in Z[0, i, :6]]
    print(f"  '{palavra:12}'  [{', '.join(valores)}, ...]")

print(f"\nMédia: {Z.mean():.6f} e Desvio: {Z.std():.6f}")