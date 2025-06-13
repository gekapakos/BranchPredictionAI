#ifndef RELU_H
#define RELU_H

#include <stdio.h>
#include <stdlib.h>

int relu(int x) {
    return (x < 0) ? 0 : x;
}

#endif // RELU_H
