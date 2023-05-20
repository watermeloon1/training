#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

typedef struct {
    size_t rows;
    size_t cols;
    float* fe;
} Mat;

Mat Mat_alloc(const size_t rows, const size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    size_t malloc_bytes = sizeof(*m.fe) * rows * cols;
    // printf("Size of malloc: %zu bytes\n", malloc_bytes);
    m.fe = malloc(malloc_bytes);
    assert(m.fe != NULL);
    return m;
}

void Mat_print(Mat m) {
    for (size_t sr = 0; sr < m.rows; ++sr) {
        for (size_t sc = 0; sc < m.cols; ++sc) {
            printf("%f ", *(m.fe + (sr * m.cols) + sc));
        }
        printf("\n");
    }
}

void Mat_dot(Mat dest, Mat a, Mat b) {
    if (a.cols != b.rows) {
        fprintf(stderr,
                "ERROR: can not calculate dot product of matrices\n");
        return;
    }

    if (a.rows != dest.rows || b.cols != dest.cols) {
        fprintf(stderr,
                "ERROR: incorrect dest matrix size\n");
        return;
    }

    for (size_t sr = 0; sr < dest.rows; ++sr) {
        for (size_t sc = 0; sc < dest.cols; ++sc) {
            for (size_t sc_a = 0; sc_a < a.cols; ++sc_a) {
                *(dest.fe + (sr * dest.cols) + sc) +=
                    *(a.fe + (sr * a.cols) + sc_a) *
                    *(b.fe + (sc_a * b.cols) + sc);
            }
        }
    }
}

void Mat_scalar(Mat m, float scalar) {
    for (size_t sr = 0; sr < m.rows; ++sr) {
        for (size_t sc = 0; sc < m.cols; ++sc) {
            *(m.fe + (sr * m.cols) + sc) *= scalar;
        }
    }
}

void Mat_sum(Mat dest, Mat a) {
    if (dest.rows != a.rows || dest.cols != a.cols) {
        fprintf(stderr,
                "ERROR: matrix rows must be equal to add\n");
        return;
    }

    for (size_t sr = 0; sr < dest.rows; ++sr) {
        for (size_t sc = 0; sc < dest.cols; ++sc) {
            *(dest.fe + (sr * dest.cols) + sc) +=
                *(a.fe + (sr * a.cols) + sc);
        }
    }
}

float sigmoidf(float x) {
    return 1.0f / (expf(-x) + 1.0f);
}

void Mat_sig(Mat m) {
    for (size_t sr = 0; sr < m.rows; ++sr) {
        for (size_t sc = 0; sc < m.cols; ++sc) {
            *(m.fe + (sr * m.cols) + sc) =
                sigmoidf(*(m.fe + (sr * m.cols) + sc));
        }
    }    
}

float randf() {
    return (float)rand() / (float)RAND_MAX;
}

void Mat_randf(Mat m) {
    for (size_t sr = 0; sr < m.rows; ++sr) {
        for (size_t sc = 0; sc < m.cols; ++sc) {
            *(m.fe + (sr * m.cols) + sc) = randf();
        }
    }    
}

void Mat_fill(Mat m, float value) {
    for (size_t sr = 0; sr < m.rows; ++sr) {
        for (size_t sc = 0; sc < m.cols; ++sc) {
            *(m.fe + (sr * m.cols) + sc) = value;
        }
    }
}

// TODO: error handling
void Mat_init(Mat m, float* values) {
    for (size_t sr = 0; sr < m.rows; ++sr) {
        for (size_t sc = 0; sc < m.cols; ++sc) {
            *(m.fe + (sr * m.cols) + sc) =
                *(values + (sr * m.cols) + sc);
        }
    } 
}

Mat Mat_cpy(Mat m) {
    Mat c;
    c.rows = m.rows;
    c.cols = m.cols;
    size_t malloc_bytes = sizeof(*m.fe) * m.rows * m.cols;
    // printf("Size of malloc: %zu bytes\n", malloc_bytes);
    c.fe = malloc(malloc_bytes);
    assert(c.fe != NULL);

    for (size_t sr = 0; sr < m.rows; ++sr) {
        for (size_t sc = 0; sc < m.cols; ++sc) {
            *(c.fe + (sr * m.cols) + sc) =
                *(m.fe + (sr * m.cols) + sc);
        }
    }

    return c;
}

// TODO: error handling
void Mat_eq(Mat dest, Mat from) {
    for (size_t sr = 0; sr < dest.rows; ++sr) {
        for (size_t sc = 0; sc < dest.cols; ++sc) {
            *(dest.fe + (dest.cols * sr) + sc) =
                *(from.fe + (from.cols * sr) + sc);
        }
    }
}

void Mat_free(Mat m) {
    free(m.fe);
    m.fe = NULL;
}

typedef struct {    
    Mat weight;
    Mat bias;
} Layer;

Layer Layer_alloc(size_t weight_rows, size_t weight_cols,
                  size_t bias_rows, size_t bias_cols) {
    Layer layer;
    layer.weight = Mat_alloc(weight_rows, weight_cols);
    layer.bias = Mat_alloc(bias_rows, bias_cols);

    Mat_randf(layer.weight);
    Mat_randf(layer.bias);

    return layer;
}

void Layer_free(Layer layer) {
    Mat_free(layer.weight);
    Mat_free(layer.bias);
}

typedef struct {
    size_t size;
    Layer* layers;
} Model;

// TODO: generalize
Model Model_alloc(const size_t size) {
    Model model;
    model.size = size;
    model.layers = malloc(sizeof(*model.layers) * size);
  
    *model.layers = Layer_alloc(2, 2, 1, 2);
    *(model.layers + 1) = Layer_alloc(2, 1, 1, 1);
    
    return model;
}

void Model_free(Model model) {
    for (size_t st = 0; st < model.size; ++st) {
        Layer_free(*(model.layers + st));
    }
    free(model.layers);
    model.layers = NULL;
}

float Model_forward(Model model, Mat input) {
    // forward all layers
    Mat input_buffer = Mat_cpy(input);
    for (size_t st = 0; st < model.size; ++st) {
        Mat output = Mat_alloc(input_buffer.rows, (*(model.layers + st)).weight.cols);
        Mat_dot(output, input_buffer, (*(model.layers + st)).weight);
        Mat_sum(output, (*(model.layers + st)).bias);
        Mat_sig(output);
        
        Mat_free(input_buffer);
        input_buffer = Mat_cpy(output);
        Mat_free(output);
    }
    float model_prediction = *(input_buffer.fe);
    Mat_free(input_buffer);

    return model_prediction;
}

float Model_costf(Model model, Mat train, const size_t train_size) {
    float cost = 0.0f;
    for (size_t st = 0; st < train_size; ++st) {
        Mat input = Mat_alloc(1, 2);
        *(input.fe) = *(train.fe + (st * train.cols));
        *(input.fe + 1) = *(train.fe + (st * train.cols) + 1);
        
        float fw = Model_forward(model, input);
        float delta = *(train.fe + (st * train.cols) + 2) - fw;
        cost += delta * delta;

        Mat_free(input);
    }
    cost /= (float)train_size;
    return cost;
}

Model Model_cpy(Model model) {
    Model c;
    c.size = model.size;
    c.layers = malloc(sizeof(*model.layers) * model.size);
    assert(c.layers != NULL);

    for (size_t sl = 0; sl < model.size; ++sl) {
        (*(c.layers + sl)).weight = Mat_cpy((*(model.layers + sl)).weight);
        (*(c.layers + sl)).bias = Mat_cpy((*(model.layers + sl)).bias);
    }

    return c;
}

void Model_swap(Model dest, Model from) {
    if (dest.size != from.size) {
        fprintf(stderr,
                "ERROR: the two models can't be swapped, due to difference in size\n");
        return;
    }
    
    // for all layers
    for (size_t sl = 0; sl < dest.size; ++sl) {
        Layer dest_layer = *(dest.layers + sl);
        Layer from_layer = *(from.layers + sl);

        // copy values but no malloc
        Mat_eq(dest_layer.weight, from_layer.weight);
        Mat_eq(dest_layer.bias, from_layer.bias);
    }
}

void Model_fit(Model model, Mat train, const size_t train_size,
               float cost, const float eps, const float rate) {

    Model buffer = Model_cpy(model);
    
    // for all layers in the model
    for (size_t sl = 0; sl < model.size; ++sl) {
        Layer layer = *(model.layers + sl);
        
        // for every weight
        for (size_t sr = 0; sr < layer.weight.rows; ++sr) {
            for (size_t sc = 0; sc < layer.weight.cols; ++sc) {
                float* iter = layer.weight.fe + (sr * layer.weight.cols) + sc;
                float save_weight = *(iter);
                
                *(iter) += eps;
                float delta = (Model_costf(model, train, train_size) - cost) / eps;
                *(iter) = save_weight;

                *((*(buffer.layers + sl)).weight.fe + (sr * (*(buffer.layers + sl)).weight.cols) + sc) -= rate * delta;
            }    
        }
        
        // for every bias
        for (size_t sr = 0; sr < layer.bias.rows; ++sr) {
            for (size_t sc = 0; sc < layer.bias.cols; ++sc) {
                float* iter = layer.bias.fe + (sr * layer.bias.cols) + sc;
                float save_bias = *(iter);
                
                *(iter) += eps;
                float delta = (Model_costf(model, train, train_size) - cost) / eps;
                *(iter) = save_bias;
                
                *((*(buffer.layers + sl)).bias.fe + (sr * (*(buffer.layers + sl)).bias.cols) + sc) -= rate * delta;
            }    
        }
    }

    // swap buffers
    Model_swap(model, buffer);
    Model_free(buffer);
}

void Model_learn(Model model, Mat train, const size_t train_size,
           const size_t iterations, const float eps, const float rate) {

    
    for (size_t it = 0; it < iterations; ++it) {
        float cost = Model_costf(model, train, train_size);
        // printf("cost: %f\n", cost);
        Model_fit(model, train, train_size, cost, eps, rate);
    }
}

void Model_print(Model model) {
    for (float ui = 0; ui < 2; ++ui) {
        for (float uj = 0; uj < 2; ++uj) {
            Mat input = Mat_alloc(1, 2);
            *(input.fe) = ui;
            *(input.fe + 1) = uj;
                
            printf("%d | %d = %f\n", (int)ui, (int)uj, Model_forward(model, input));
            Mat_free(input);
        }
    }
}

float xor[] = {
    0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 0.0f
};

int main() {
    srand(42);

    Mat train = Mat_alloc(4, 3);
    const size_t train_size = 4;
    Mat_init(train, xor);

    Model model = Model_alloc(2);

    const size_t iteration = 100 * 1000;
    const float eps = 1e-01;
    const float rate = 1e-01;

    Model_learn(model, train, train_size, iteration, eps, rate);

    Mat_free(train);
    
    Model_print(model);
    Model_free(model);
    
    return 0;
}
