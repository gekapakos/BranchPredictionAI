// embedding_test.c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define E           4   // embedding dim
#define VOCAB_SIZE  5   // vocabulary size
#define T           5   // sequence length

// --- Your function (uses globals E, VOCAB_SIZE, T) ---
static inline void embedding_forward(const int *tokens, float *X, const float *Emb)
{
    for (int t = 0; t < T; ++t) {
        int idx = tokens[t];
        if (idx < 0 || idx >= VOCAB_SIZE) idx = 0; // clamp
        const float *row = Emb + idx * E;
        float *dst = X + t * E;
        for (int e = 0; e < E; ++e) dst[e] = row[e];
    }
}

// ---- helpers ----
static void print_vec_i(const char *name, const int *v, int n){
    printf("%s = [", name);
    for (int i=0;i<n;++i) printf("%s%d", i?", ":"", v[i]);
    printf("]\n");
}
static void print_mat_TxE(const char *name, const float *X, int Tn, int En){
    printf("%s:\n", name);
    for (int t=0;t<Tn;++t){
        printf("  t=%d: [", t);
        for (int e=0;e<En;++e) printf("%s%.1f", e?", ":"", X[t*En+e]);
        printf("]\n");
    }
}
static bool allequal(const float *a, const float *b, int n){
    for (int i=0;i<n;++i) if (a[i] != b[i]) return false;
    return true;
}

int main(void)
{
    // Build a tiny embedding table with easily verified rows:
    // row v = [ 100*v + 0, 100*v + 1, 100*v + 2, 100*v + 3 ]
    float Emb[VOCAB_SIZE * E];
    for (int v=0; v<VOCAB_SIZE; ++v)
        for (int e=0; e<E; ++e)
            Emb[v*E + e] = 100.0f*v + (float)e;

    // Tokens include valid, negative, and >=VOCAB indices to test clamping→0
    int tokens[T] = { 2, -3, 0, VOCAB_SIZE /*5*/, 4 };

    // Run
    float X[T*E];
    embedding_forward(tokens, X, Emb);

    // Build expected output using the same “clamp to 0” rule
    float X_exp[T*E];
    for (int t=0; t<T; ++t) {
        int idx = tokens[t];
        if (idx < 0 || idx >= VOCAB_SIZE) idx = 0;
        const float *row = Emb + idx*E;
        memcpy(&X_exp[t*E], row, sizeof(float)*E);
    }

    // Display and check
    print_vec_i("tokens", tokens, T);
    print_mat_TxE("Embedding table (first 5 rows)", Emb, VOCAB_SIZE, E);
    print_mat_TxE("X (output)", X, T, E);
    print_mat_TxE("X_exp (expected)", X_exp, T, E);

    printf("\nRESULT: %s\n", allequal(X, X_exp, T*E) ? "PASS" : "FAIL");
    return allequal(X, X_exp, T*E) ? 0 : 1;
}