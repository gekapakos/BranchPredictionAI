#ifndef EMBEDDING_H
#define EMBEDDING_H

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

float* embedding(const float *weight,
                      int num_embeddings,
                      int embedding_dim,
                      const int *indices,
                      int N)
{
    /*  Row‑major layout helpers */
    #define WEIGHT_ROW(tok)  ( (tok) * embedding_dim )
    #define OUT_ROW(n)       ( (n)   * embedding_dim )

    float* out = (float*)malloc(N * embedding_dim * sizeof(float));

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

    return out;
}

#endif // EMBEDDING_H
