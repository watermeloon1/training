#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SECRET 3.0f

float table[][2] = {
    {0, 0},
    {1, 3},
    {2, 6},
    {3, 9}
};
size_t length = sizeof(table) / sizeof(table[0]);

float rand_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

float cost(float w) {
    float cost = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        float estimate = table[i][0] * w;
        float d = estimate - table[i][1];
        cost += d * d;
    }
    cost /= length;
    return cost;
}

void print_table(float w) {
    for (size_t i = 0; i < length; ++i) {
        printf("%f * %f = %f\n", table[i][0], SECRET, table[i][0] * w);
    }
}

int main()
{
     //srand(time(0));
     srand(42);
     float w = rand_float() * 10.0f;
     float eps = 1e-03;
     float rate = 1e-03;


     printf("cost: %f w: %f\n", cost(w), w);
     printf("-----------------------------\n");
     print_table(w);

     for (size_t i = 0; i < 1000; ++i) {
         float dcost = (cost(w + eps) - cost(w)) / eps;
         w -= rate * dcost;
         // printf("[%zu] cost: %f w: %f\n", i + 1, cost(w), w);
     }

     printf("\ncost: %f w: %f\n", cost(w), w);
     printf("-----------------------------\n");
     print_table(w);

     return 0;
}
