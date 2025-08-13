// ------------------------------------------------------------
//  Single‑function «nn.Embedding»‑style lookup in plain C
//  Parameters match PyTorch‑like signature:
//     num_embeddings  – vocabulary size (rows in table)
//     embedding_dim   – vector size per token (cols in table)
//  Inputs:
//     weight      – pointer to a row‑major float array of size
//                    num_embeddings * embedding_dim
//     indices     – pointer to int array of length N containing the
//                    token IDs to look up (each 0 … num_embeddings‑1)
//     N           – number of indices (output rows)
//  Output:
//     out (float*) – caller‑allocated buffer of size N * embedding_dim
//
//  The function performs:  out[n][d] = weight[ indices[n] ][d ]
// ------------------------------------------------------------
#include <stddef.h> // size_t
#include <stdio.h>
#include <stdlib.h> // malloc, free
#include <math.h>  // isnan, isinf
#include <stdbool.h> // bool, true, false
#include <string.h> // memset

void embedding_lookup(const float *weight,
                      int num_embeddings,
                      int embedding_dim,
                      const int *indices,
                      int N,
                      float *out)
{
    /*  Row‑major layout helpers */
    #define WEIGHT_ROW(tok)  ( (tok) * embedding_dim )
    #define OUT_ROW(n)       ( (n)   * embedding_dim )

    for (int n = 0; n < N; ++n) {
        int tok = indices[n];
        if (tok < 0 || tok >= num_embeddings) {
            /* OOV → zero vector */
            for (int d = 0; d < embedding_dim; ++d)
                out[OUT_ROW(n)+d] = 0.0f;
            continue;
        }
        const float *src = weight + WEIGHT_ROW(tok);
        float       *dst = out    + OUT_ROW(n);
        for (int d = 0; d < embedding_dim; ++d)
            dst[d] = src[d];
    }

    #undef WEIGHT_ROW
    #undef OUT_ROW
}

/* Usage example (allocate out before calling):

    int vocab = 10000, dim = 300, N = 4;
    float *weights = malloc(vocab * dim * sizeof *weights);
    int   idx[4]   = {42, 1, 9999, 7};
    float out[4*300];

    embedding_lookup(weights, vocab, dim, idx, N, out);
*/

// int main()
// {
//     // Example usage of embedding_lookup
//     int num_embeddings = 10000; // Vocabulary size
//     int embedding_dim = 300;    // Size of each embedding vector
//     int N = 4;                  // Number of indices to look up
//     float *weight = (float *)malloc(num_embeddings * embedding_dim * sizeof(float));
//     int indices[] = {42, 1, 9999, 7}; // Example indices to look up
//     float *out = (float *)malloc(N * embedding_dim * sizeof(float));
//     // Initialize weight with some values (for demonstration purposes)
//     for (int i = 0; i < num_embeddings * embedding_dim; ++i) {
//         weight[i] = (float)(i % 100) / 100.0f; // Example initialization
//     }
//     // Call the embedding lookup function
//     embedding_lookup(weight, num_embeddings, embedding_dim, indices, N, out);
//     // Print the output
//     for (int n = 0; n < N; ++n) {
//         printf("Embedding for index %d: ", indices[n]);
//         for (int d = 0; d < embedding_dim; ++d) {
//             printf("%f ", out[n * embedding_dim + d]);
//         }
//         printf("\n");
//     }
//     return 0;
// }