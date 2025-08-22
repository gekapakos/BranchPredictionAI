#ifndef WEIGHTS_IO_H
#define WEIGHTS_IO_H

#include <stdio.h>
#include <stdlib.h>

static inline void load_bin_f32(const char *path, float *buf, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", path); exit(1); }
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) {
        fprintf(stderr, "read count mismatch: %s (got %zu, want %zu)\n", path, got, n);
        exit(1);
    }
}

static inline void load_bin_i32(const char *path, int *buf, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open failed: %s\n", path); exit(1); }
    size_t got = fread(buf, sizeof(int), n, f);
    fclose(f);
    if (got != n) {
        fprintf(stderr, "read count mismatch: %s (got %zu, want %zu)\n", path, got, n);
        exit(1);
    }
}

#endif
