#ifndef RELU_H
#define RELU_H

#include <stdio.h>
#include <stdlib.h>

float relu(float x) {
    return (x < 0) ? 0 : x;
}

#endif // RELU_H
