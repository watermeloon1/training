#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define LOG 1

typedef float Train[3];

float train_and[][3] = {
    {0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {1.0f, 1.0f, 1.0f}
};

float train_or[][3] = {
    {0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f}
};

float train_nand[][3] = {
    {0.0f, 0.0f, 1.0f},
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 0.0f}
};

float train_xor[][3] = {
    {0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 0.0f}
};

const size_t train_length = sizeof(train_and) / sizeof(train_and[0]);

float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float value) {
    return 1.0f / (1.0f + expf(-value));
}

float cost(Train *train, float w1, float w2, float bias) {
    float c = 0.0f;
    for (size_t i = 0; i < train_length; ++i) {
        float estimate_x1 = train[i][0] * w1;
        float estimate_x2 = train[i][1] * w2;
        float delta = sigmoidf(estimate_x1 + estimate_x2 + bias) - train[i][2];
        c += delta * delta;
    }
    c /= train_length;
    return c;
}

void table_print(float w1, float w2, float bias) {
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            printf("%zu | %zu = %f\n", i, j, sigmoidf(i*w1 + j*w2 + bias));
        }
    }
}

void learn_gate(Train *train) {
    srand(42);
    float w1 = randf();
    float w2 = randf();
    float b = randf();

    float eps = 1e-01;
    float rate = 1e-01;

    size_t iterations = 10 * 1000;

    for (size_t i = 0; i < iterations; ++i) {
        float c = cost(train, w1, w2, b);

        float dcost_w1 = (cost(train, w1 + eps, w2, b) - c) / eps;
        float dcost_w2 = (cost(train, w1, w2 + eps, b) - c) / eps;
        float dcost_b = (cost(train, w1, w2, b + eps) - c) / eps;

        w1 -= rate * dcost_w1;
        w2 -= rate * dcost_w2;
        b -= rate * dcost_b;
    }

    table_print(w1, w2, b);
}

int main() {

    learn_gate(train_xor);
        
    return 0;
}
