#ifndef __DENSE_H
#define __DENSE_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Dense Layer as two matrices and two vectors:

    input vec{x}         weights W             masks M
 [ x1 x2 x3 x4 1 ]  [ w11 w12 w13 w14 ]  [ m11 m12 m13 m14 ]
                    [ w21 w22 w23 w24 ]  [ m21 m22 m23 m24 ]
                    [ w31 w32 w33 w34 ]  [ m31 m32 m33 m34 ]
                    [ w41 w42 w43 w44 ]  [ m41 m42 m43 m44 ]
                    [ w51 w52 w53 w54 ]  [  1   1   1   1  ]

                    activation function
                         vector f
               [ func1 func2 func3 func4 ]

                     auxiliary vector
                          alpha
               [ param1 param2 param3 param4 ]

 Broadcast W and M = W'
 vec{x} dot W' = x'
 vec{output} is func[i](x'[i], param[i]) for each i

 Not all activation functions need a parameter. It's just a nice feature we like to offer.

 Note that this file does NOT seed the randomizer. That should be done by the parent program.
***************************************************************************************************/
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "cblas.h"

#define RELU                 0                                      /* [ 0.0, inf) */
#define LEAKY_RELU           1                                      /* (-inf, inf) */
#define SIGMOID              2                                      /* ( 0.0, 1.0) */
#define HYPERBOLIC_TANGENT   3                                      /* [-1.0, 1.0] */
#define SOFTMAX              4                                      /* [ 0.0, 1.0] */
#define SYMMETRICAL_SIGMOID  5                                      /* (-1.0, 1.0) */
#define THRESHOLD            6                                      /* { 0.0, 1.0} */
#define LINEAR               7                                      /* (-inf, inf) */

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __NEURON_DEBUG 1
*/

typedef struct DenseLayerType
  {
    unsigned int i;                                                 //  Number of inputs--NOT COUNTING the added bias-1
    unsigned int n;                                                 //  Number of processing units in this layer
    double* W;                                                      //  ((i + 1) x n) matrix
    double* M;                                                      //  ((i + 1) x n) matrix, all either 0.0 or 1.0
    unsigned char* f;                                               //  n-array
    double* alpha;                                                  //  n-array
    char name[LAYER_NAME_LEN];
    double* out;
  } DenseLayer;

/**************************************************************************************************
 Prototypes  */

void setW_Dense(double*, DenseLayer*);                              //  Set entirety of layer's weight matrix
void setW_i_Dense(double*, unsigned int, DenseLayer*);              //  Set entirety of weights for i-th column/neuron/unit
void setW_ij_Dense(double, unsigned int, unsigned int, DenseLayer*);//  Set element [i, j] of layer's weight matrix
void setM_Dense(bool*, DenseLayer*);                                //  Set entirety of layer's mask matrix
void setM_i_Dense(bool*, unsigned int, DenseLayer*);                //  Set entirety of masks for i-th column/neuron/unit
void setM_ij_Dense(bool, unsigned int, unsigned int, DenseLayer*);  //  Set element [i, j] of layer's mask matrix
void setF_i_Dense(unsigned char, unsigned int, DenseLayer*);        //  Set activation function of i-th neuron/unit
void setA_i_Dense(double, unsigned int, DenseLayer*);               //  Set activation function auxiliary parameter of i-th neuron/unit
void setName_Dense(char*, DenseLayer*);
void print_Dense(DenseLayer*);
unsigned int outputLen_Dense(DenseLayer*);
unsigned int run_Dense(double*, DenseLayer*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 Dense-Layers  */

/* Set entirety of layer's weight matrix.
   Input buffer 'w' is expected to be ROW-MAJOR
        weights W
   [ w0  w1  w2  w3  ]
   [ w4  w5  w6  w7  ]
   [ w8  w9  w10 w11 ]
   [ w12 w13 w14 w15 ]
   [ w16 w17 w18 w19 ]  <--- biases
*/
void setW_Dense(double* w, DenseLayer* layer)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("setW_Dense()\n");
    #endif

    for(i = 0; i < (layer->i + 1) * layer->n; i++)
      layer->W[i] = w[i];
    return;
  }

/* Set entirety of weights for i-th column/neuron/unit. */
void setW_i_Dense(double* w, unsigned int i, DenseLayer* layer)
  {
    unsigned int j;

    #ifdef __NEURON_DEBUG
    printf("setW_i_Dense()\n");
    #endif

    for(j = 0; j <= layer->i; j++)
      layer->W[j * layer->n + i] = w[j];
    return;
  }

/* Set unit[i], weight[j] of the given layer */
void setW_ij_Dense(double w, unsigned int i, unsigned int j, DenseLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setW_ij_Dense(%f, %d, %d)\n", w, i, j);
    #endif

    if(j * layer->n + i < (layer->i + 1) * layer->n)
      layer->W[j * layer->n + i] = w;
    return;
  }

/* Set entirety of layer's mask matrix
   Input buffer 'm' is expected to be ROW-MAJOR
        masks M
   [ m0  m1  m2  m3  ]
   [ m4  m5  m6  m7  ]
   [ m8  m9  m10 m11 ]
   [ m12 m13 m14 m15 ]
   [ m16 m17 m18 m19 ]  <--- biases
*/
void setM_Dense(bool* m, DenseLayer* layer)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("setM_Dense()\n");
    #endif

    for(i = 0; i < (layer->i + 1) * layer->n; i++)
      {
        if(m[i])                                                    //  TRUE means UNMASKED
          layer->M[i] = 1.0;
        else                                                        //  FALSE means MASKED
          layer->M[i] = 0.0;
      }
    return;
  }

/* Set entirety of masks for i-th column/neuron/unit */
void setM_i_Dense(bool* m, unsigned int i, DenseLayer* layer)
  {
    unsigned int j;

    #ifdef __NEURON_DEBUG
    printf("setM_i_Dense()\n");
    #endif

    for(j = 0; j <= layer->i; j++)
      {
        if(m[j])
          layer->M[j * layer->n + i] = 1.0;
        else
          layer->M[j * layer->n + i] = 0.0;
      }
    return;
  }

/* Set unit[i], weight[j] of the given layer */
void setM_ij_Dense(bool m, unsigned int i, unsigned int j, DenseLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    if(m)
      printf("setM_ij_Dense(LIVE, %d, %d)\n", i, j);
    else
      printf("setM_ij_Dense(MASKED, %d, %d)\n", i, j);
    #endif

    if(j * layer->n + i < (layer->i + 1) * layer->n)
      {
        if(m)
          layer->M[j * layer->n + i] = 1.0;
        else
          layer->M[j * layer->n + i] = 0.0;
      }
    return;
  }

/* Set the activation function for unit[i] of the given layer */
void setF_i_Dense(unsigned char func, unsigned int i, DenseLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setF_i_Dense(%d, %d)\n", func, i);
    #endif

    if(i < layer->n)
      layer->f[i] = func;
    return;
  }

/* Set the activation function parameter for unit[i] of the given layer */
void setA_i_Dense(double a, unsigned int i, DenseLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setA_i_Dense(%f, %d)\n", a, i);
    #endif

    if(i < layer->n)
      layer->alpha[i] = a;
    return;
  }

/* Set the name of the given Dense Layer */
void setName_Dense(char* n, DenseLayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

/* Print the details of the given DenseLayer 'layer' */
void print_Dense(DenseLayer* layer)
  {
    unsigned int i, j;

    #ifdef __NEURON_DEBUG
    printf("print_Dense()\n");
    #endif

    for(i = 0; i < layer->i + 1; i++)
      {
        if(i == layer->i)
          printf("bias [");
        else
          printf("     [");
        for(j = 0; j < layer->n; j++)
          {
            if(layer->W[j * (layer->i + 1) + i] >= 0.0)
              printf(" %.5f ", layer->W[j * (layer->i + 1) + i]);
            else
              printf("%.5f ", layer->W[j * (layer->i + 1) + i]);
          }
        printf("]\n");
      }
    printf("f = [");
    for(i = 0; i < layer->n; i++)
      {
        switch(layer->f[i])
          {
            case RELU:                printf("ReLU   ");  break;
            case LEAKY_RELU:          printf("L.ReLU ");  break;
            case SIGMOID:             printf("Sig.   ");  break;
            case HYPERBOLIC_TANGENT:  printf("tanH   ");  break;
            case SOFTMAX:             printf("SoftMx ");  break;
            case SYMMETRICAL_SIGMOID: printf("SymSig ");  break;
            case THRESHOLD:           printf("Thresh ");  break;
            case LINEAR:              printf("Linear ");  break;
          }
      }
    printf("]\n");
    printf("a = [");
    for(i = 0; i < layer->n; i++)
      printf("%.4f ", layer->alpha[i]);
    printf("]\n");

    return;
  }

/* Return the layer's output length
   (For Dense layers, this is the number of units) */
unsigned int outputLen_Dense(DenseLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("outputLen_Dense()\n");
    #endif

    return layer->n;
  }

/* Run the given input vector 'x' of length 'layer'->'i' through the DenseLayer 'layer'.
   Output is stored internally in layer->out. */
unsigned int run_Dense(double* x, DenseLayer* layer)
  {
                                                                    //  Input vector augmented with additional (bias) 1.0
    double* xprime;                                                 //  (1 * (length-of-input + 1))
    double* Wprime;                                                 //  ((length-of-input + 1) * nodes)
    double softmaxdenom = 0.0;                                      //  Accumulate exp()'s to normalize any softmax
    unsigned int i;
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE transa;

    #ifdef __NEURON_DEBUG
    printf("run_Dense(%d)\n", layer->i);
    #endif

    order = CblasRowMajor;                                          //  Describe how the data in W are stored
    transa = CblasTrans;

    if((xprime = (double*)malloc((layer->i + 1) * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate augmented input vector\n");
        exit(1);
      }
    for(i = 0; i < layer->i; i++)                                   //  Append 1.0 to input vector
      xprime[i] = x[i];
    xprime[i] = 1.0;
                                                                    //  Allocate W' matrix
    if((Wprime = (double*)malloc((layer->i + 1) * layer->n * sizeof(double))) == NULL)
      {
        free(xprime);
        printf("ERROR: Unable to allocate masked-weight matrix\n");
        exit(1);
      }

    //                       weights W                                                  masks M
    //     i = 0 ----------------------------> layer->nodes        i = 0 ----------------------------> layer->nodes
    //   j   [ A+0      A+((len+1)*i)        A+((len+1)*i)+j ]   j   [ A+0      A+((len+1)*i)        A+((len+1)*i)+j ]
    //   |   [ A+1      A+((len+1)*i)+1      A+((len+1)*i)+j ]   |   [ A+1      A+((len+1)*i)+1      A+((len+1)*i)+j ]
    //   |   [ A+2      A+((len+1)*i)+2      A+((len+1)*i)+j ]   |   [ A+2      A+((len+1)*i)+2      A+((len+1)*i)+j ]
    //   |   [ ...      ...                  ...             ]   |   [ ...      ...                  ...             ]
    //   V   [ A+len    A+((len+1)*i)+len    A+((len+1)*i)+j ]   V   [ A+len    A+((len+1)*i)+len    A+((len+1)*i)+j ]
    // len+1 [ A+len+1  A+((len+1)*i)+len+1  A+((len+1)*i)+j ] len+1 [  1        1                    1              ]
    for(i = 0; i < (layer->i + 1) * layer->n; i++)                  //  Broadcast weights and masks into W'
      Wprime[i] = layer->W[i] * layer->M[i];
                                                                    //  Dot-product xprime Wprime ---> layer->out
    cblas_dgemv(order,                                              //  The order in which data in Wprime are stored
                transa,                                             //  Transpose
                layer->i + 1,                                       //  Number of ROWS in Wprime = number of inputs + 1 row of biases
                layer->n,                                           //  Number of COLUMNS in Wprime = number of layer units
                1.0,                                                //  alpha (ignore this)
                Wprime,
                layer->n,                                           //  Stride in Wprime equals the number of COLUMNS when order == CblasRowMajor
                xprime, 1, 0.0, layer->out, 1);

    for(i = 0; i < layer->n; i++)                                   //  In case one of the units is a softmax unit,
      softmaxdenom += pow(M_E, layer->out[i]);                      //  compute all exp()'s so we can sum them.

    for(i = 0; i < layer->n; i++)                                   //  Run each element in out through appropriate function
      {                                                             //  with corresponding parameter
        switch(layer->f[i])
          {
            case RELU:                 layer->out[i] = (layer->out[i] > 0.0) ? layer->out[i] : 0.0;
                                       break;
            case LEAKY_RELU:           layer->out[i] = (layer->out[i] > 0.0) ? layer->out[i] : layer->out[i] * layer->alpha[i];
                                       break;
            case SIGMOID:              layer->out[i] = 1.0 / (1.0 + pow(M_E, -layer->out[i] * layer->alpha[i]));
                                       break;
            case HYPERBOLIC_TANGENT:   layer->out[i] = (2.0 / (1.0 + pow(M_E, -2.0 * layer->out[i] * layer->alpha[i]))) - 1.0;
                                       break;
            case SOFTMAX:              layer->out[i] = pow(M_E, layer->out[i]) / softmaxdenom;
                                       break;
            case SYMMETRICAL_SIGMOID:  layer->out[i] = (1.0 - pow(M_E, -layer->out[i] * layer->alpha[i])) / (1.0 + pow(M_E, -layer->out[i] * layer->alpha[i]));
                                       break;
            case THRESHOLD:            layer->out[i] = (layer->out[i] > layer->alpha[i]) ? 1.0 : 0.0;
                                       break;
                                                                    //  (Includes LINEAR)
            default:                   layer->out[i] *= layer->alpha[i];
          }
      }

    free(xprime);
    free(Wprime);

    return layer->n;                                                //  Return the length of layer->out
  }

#endif
