#ifndef __LSTM_H
#define __LSTM_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model an LSTM Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec{x}       weights Wi              weights Wo              weights Wf              weights Wc
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)                (h by d)
                 [ wi11 wi12 wi13 wi14 ] [ wo11 wo12 wo13 wo14 ] [ wf11 wf12 wf13 wf14 ] [ wc11 wc12 wc13 wc14 ]
                 [ wi21 wi22 wi23 wi24 ] [ wo21 wo22 wo23 wo24 ] [ wf21 wf22 wf23 wf24 ] [ wc21 wc22 wc23 wc24 ]
                 [ wi31 wi32 wi33 wi34 ] [ wo31 wo32 wo33 wo34 ] [ wf31 wf32 wf33 wf34 ] [ wc31 wc32 wc33 wc34 ]

                       weights Ui              weights Uo              weights Uf              weights Uc
                        (h by h)                (h by h)                (h by h)                (h by h)
                 [ ui11 ui12 ui13 ]      [ uo11 uo12 uo13 ]      [ uf11 uf12 uf13 ]      [ uc11 uc12 uc13 ]
                 [ ui21 ui22 ui23 ]      [ uo21 uo22 uo23 ]      [ uf21 uf22 uf23 ]      [ uc21 uc22 uc23 ]
                 [ ui31 ui32 ui33 ]      [ uo31 uo32 uo33 ]      [ uf31 uf32 uf33 ]      [ uc31 uc32 uc33 ]

                     bias h-vec{bi}          bias h-vec{bo}          bias h-vec{bf}          bias h-vec{bc}
                 [ bi1 ]                 [ bo1 ]                 [ bf1 ]                 [ bc1 ]
                 [ bi2 ]                 [ bo2 ]                 [ bf2 ]                 [ bc2 ]
                 [ bi3 ]                 [ bo3 ]                 [ bf3 ]                 [ bc3 ]

         H state cache (times 1, 2, 3, 4 = columns 0, 1, 2, 3)
        (h by cache)
 [ H11 H12 H13 H14 ]
 [ H21 H22 H23 H24 ]
 [ H31 H32 H33 H34 ]

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

typedef struct LSTMLayerType
  {
    unsigned int d;                                                 //  Dimensionality of input vector
    unsigned int h;                                                 //  Dimensionality of hidden state vector
    unsigned int cache;                                             //  The number of states to keep in memory:
                                                                    //  when 't' exceeds this, shift out.
    unsigned int t;                                                 //  The time step
                                                                    //  W matrices are (h by d)
    double* Wi;                                                     //  Input gate weights
    double* Wo;                                                     //  Output gate weights
    double* Wf;                                                     //  Forget gate weights
    double* Wc;                                                     //  Memory cell weights
                                                                    //  U matrices are (h by h)
    double* Ui;                                                     //  Recurrent connection input gate weights
    double* Uo;                                                     //  Recurrent connection output gate weights
    double* Uf;                                                     //  Recurrent connection forget gate weights
    double* Uc;                                                     //  Recurrent connection memory cell weights
                                                                    //  Bias vectors are length h
    double* bi;                                                     //  Input gate bias
    double* bo;                                                     //  Output gate bias
    double* bf;                                                     //  Forget gate bias
    double* bc;                                                     //  Memory cell bias

    double* c;                                                      //  Cell state vector, length h
    double* H;                                                      //  Hidden state cache matrix (h by cache)
    char name[LAYER_NAME_LEN];
  } LSTMLayer;

/**************************************************************************************************
 Prototypes  */

void setWi_LSTM(double*, LSTMLayer*);                               //  Set entirety of Wi weight matrix
void setWo_LSTM(double*, LSTMLayer*);                               //  Set entirety of Wo weight matrix
void setWf_LSTM(double*, LSTMLayer*);                               //  Set entirety of Wf weight matrix
void setWc_LSTM(double*, LSTMLayer*);                               //  Set entirety of Wc weight matrix
void setWi_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Wi weight matrix
void setWo_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Wo weight matrix
void setWf_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Wf weight matrix
void setWc_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Wc weight matrix
void setUi_LSTM(double*, LSTMLayer*);                               //  Set entirety of Ui weight matrix
void setUo_LSTM(double*, LSTMLayer*);                               //  Set entirety of Uo weight matrix
void setUf_LSTM(double*, LSTMLayer*);                               //  Set entirety of Uf weight matrix
void setUc_LSTM(double*, LSTMLayer*);                               //  Set entirety of Uc weight matrix
void setUi_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Ui weight matrix
void setUo_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Uo weight matrix
void setUf_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Uf weight matrix
void setUc_ij_LSTM(double, unsigned int, unsigned int, LSTMLayer*); //  Set element [i, j] of Uc weight matrix
void setbi_LSTM(double*, LSTMLayer*);                               //  Set entirety of bi bias vector
void setbo_LSTM(double*, LSTMLayer*);                               //  Set entirety of bo bias vector
void setbf_LSTM(double*, LSTMLayer*);                               //  Set entirety of bf bias vector
void setbc_LSTM(double*, LSTMLayer*);                               //  Set entirety of bc bias vector
void setbi_i_LSTM(double, unsigned int, LSTMLayer*);                //  Set i-th element of bi bias vector
void setbo_i_LSTM(double, unsigned int, LSTMLayer*);                //  Set i-th element of bo bias vector
void setbf_i_LSTM(double, unsigned int, LSTMLayer*);                //  Set i-th element of bf bias vector
void setbc_i_LSTM(double, unsigned int, LSTMLayer*);                //  Set i-th element of bc bias vector
void setName_LSTM(char*, LSTMLayer*);
void print_LSTM(LSTMLayer*);
unsigned int outputLen_LSTM(LSTMLayer*);
unsigned int run_LSTM(double*, LSTMLayer*);
void reset_LSTM(LSTMLayer*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 LSTM-Layers  */

/* Set the entirety of the Wi matrix of the given layer using the given array */
void setWi_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->d * layer->h; i++)
      layer->Wi[i] = w[i];
    return;
  }

/* Set the entirety of the Wo matrix of the given layer using the given array */
void setWo_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->d * layer->h; i++)
      layer->Wo[i] = w[i];
    return;
  }

/* Set the entirety of the Wf matrix of the given layer using the given array */
void setWf_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->d * layer->h; i++)
      layer->Wf[i] = w[i];
    return;
  }

/* Set the entirety of the Wc matrix of the given layer using the given array */
void setWc_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->d * layer->h; i++)
      layer->Wc[i] = w[i];
    return;
  }

/* Set column[i], row[j] of the given layer, Wi matrix */
void setWi_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWi_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->d)
      layer->Wi[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Wo matrix */
void setWo_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWo_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->d)
      layer->Wo[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Wf matrix */
void setWf_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWf_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->d)
      layer->Wf[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Wc matrix */
void setWc_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWc_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->d)
      layer->Wc[i * layer->h + j] = w;
    return;
  }

/* Set the entirety of the Ui matrix of the given layer using the given array */
void setUi_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h * layer->h; i++)
      layer->Ui[i] = w[i];
    return;
  }

/* Set the entirety of the Uo matrix of the given layer using the given array */
void setUo_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h * layer->h; i++)
      layer->Uo[i] = w[i];
    return;
  }

/* Set the entirety of the Uf matrix of the given layer using the given array */
void setUf_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h * layer->h; i++)
      layer->Uf[i] = w[i];
    return;
  }

/* Set the entirety of the Uc matrix of the given layer using the given array */
void setUc_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h * layer->h; i++)
      layer->Uc[i] = w[i];
    return;
  }

/* Set column[i], row[j] of the given layer, Ui matrix */
void setUi_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUi_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->h)
      layer->Ui[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Uo matrix */
void setUo_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUo_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->h)
      layer->Uo[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Uf matrix */
void setUf_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUf_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->h)
      layer->Uf[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Uc matrix */
void setUc_ij_LSTM(double w, unsigned int i, unsigned int j, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUc_ij_LSTM(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->h)
      layer->Uc[i * layer->h + j] = w;
    return;
  }

/* Set the entirety of the bi vector of the given layer using the given array */
void setbi_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h; i++)
      layer->bi[i] = w[i];
    return;
  }

/* Set the entirety of the bo vector of the given layer using the given array */
void setbo_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h; i++)
      layer->bo[i] = w[i];
    return;
  }

/* Set the entirety of the bf vector of the given layer using the given array */
void setbf_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h; i++)
      layer->bf[i] = w[i];
    return;
  }

/* Set the entirety of the bc vector of the given layer using the given array */
void setbc_LSTM(double* w, LSTMLayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h; i++)
      layer->bc[i] = w[i];
    return;
  }

/* Set element [i] of the given layer, bi vector */
void setbi_i_LSTM(double w, unsigned int i, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setbi_i_LSTM(%f, %d)\n", w, i);
    #endif

    if(i < layer->h)
      layer->bi[i] = w;
    return;
  }

/* Set element [i] of the given layer, bo vector */
void setbo_i_LSTM(double w, unsigned int i, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setbo_i_LSTM(%f, %d)\n", w, i);
    #endif

    if(i < layer->h)
      layer->bo[i] = w;
    return;
  }

/* Set element [i] of the given layer, bf vector */
void setbf_i_LSTM(double w, unsigned int i, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setbf_i_LSTM(%f, %d)\n", w, i);
    #endif

    if(i < layer->h)
      layer->bf[i] = w;
    return;
  }

/* Set element [i] of the given layer, bc vector */
void setbc_i_LSTM(double w, unsigned int i, LSTMLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setbc_i_LSTM(%f, %d)\n", w, i);
    #endif

    if(i < layer->h)
      layer->bc[i] = w;
    return;
  }

void setName_LSTM(char* n, LSTMLayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

/* Print the details of the given LSTMLayer 'layer' */
void print_LSTM(LSTMLayer* layer)
  {
    unsigned int i, j;

    #ifdef __NEURON_DEBUG
    printf("print_LSTM()\n");
    #endif

    printf("Input dimensionality d = %d\n", layer->d);
    printf("State dimensionality h = %d\n", layer->h);
    printf("State cache size       = %d\n", layer->cache);

    printf("Wi (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->d; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Wi[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Wf (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->d; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Wf[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Wc (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->d; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Wc[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Wo (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->d; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Wo[i * layer->h + j]);
        printf(" ]\n");
      }

    printf("Ui (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Ui[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Uf (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Uf[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Uc (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Uc[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Uo (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Uo[i * layer->h + j]);
        printf(" ]\n");
      }

    printf("bi (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->bi[i]);
    printf("bf (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->bf[i]);
    printf("bc (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->bc[i]);
    printf("bo (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->bo[i]);

    return;
  }

unsigned int outputLen_LSTM(LSTMLayer* layer)
  {
    return layer->h;
  }

/* Run the given input vector 'x' of length 'layer'->'d' through the LSTMLayer 'layer'.
   Output is stored internally in layer->H.
   Write to the 'layer'->'t'-th column and increment t.
   If 'layer'->'t' exceeds 'layer'->'cache', shift everything down. */
unsigned int run_LSTM(double* x, LSTMLayer* layer)
  {
    unsigned int n, m;
    double* i;
    double* f;
    double* c;                                                      //  Time t
    double* o;
    double* ct_1;                                                   //  Time t - 1
    double* ht_1;                                                   //  Time t - 1
    unsigned int t_1;                                               //  Where we READ FROM
    unsigned int t;                                                 //  Where we WRITE TO
                                                                    //  layer->t increases indefinitely
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE transa;

    #ifdef __NEURON_DEBUG
    printf("run_LSTM(%d)\n", layer->h);
    #endif

    order = CblasColMajor;
    transa = CblasTrans;

    if((i = (double*)malloc(layer->h * sizeof(double))) == NULL)    //  Allocate vec{i}
      {
        printf("ERROR: Unable to allocate LSTM vector i\n");
        exit(1);
      }
    if((f = (double*)malloc(layer->h * sizeof(double))) == NULL)    //  Allocate vec{f}
      {
        printf("ERROR: Unable to allocate LSTM vector f\n");
        exit(1);
      }
    if((c = (double*)malloc(layer->h * sizeof(double))) == NULL)    //  Allocate vec{c}
      {
        printf("ERROR: Unable to allocate LSTM vector c\n");
        exit(1);
      }
    if((o = (double*)malloc(layer->h * sizeof(double))) == NULL)    //  Allocate vec{o}
      {
        printf("ERROR: Unable to allocate LSTM vector o\n");
        exit(1);
      }
    if((ht_1 = (double*)malloc(layer->h * sizeof(double))) == NULL) //  Allocate vec{ht_1}
      {
        printf("ERROR: Unable to allocate LSTM vector ht_1\n");
        exit(1);
      }
    if((ct_1 = (double*)malloc(layer->h * sizeof(double))) == NULL) //  Allocate vec{ct_1}
      {
        printf("ERROR: Unable to allocate LSTM vector ct_1\n");
        exit(1);
      }

    if(layer->t == 0)                                               //  Timestep layer->t = 0 uses the zero-vectors for t - 1
      {
        t_1 = 0;
        t = 0;
        for(n = 0; n < layer->h; n++)                               //  Write zeroes to h(t-1) and c(t-1)
          {
            ht_1[n] = 0.0;
            ct_1[n] = 0.0;
          }
      }
    else                                                            //  Timestep t > 0 uses the previous state
      {                                                             //  Consider that we may have shifted states
        if(layer->t >= layer->cache)                                //  out of the matrix
          {
            t_1 = layer->cache - 1;                                 //  Read from the rightmost column
                                                                    //  (then shift everything left)
            t = layer->cache - 1;                                   //  Write to the rightmost column
          }
        else                                                        //  We've not yet maxed out cache
          {
            t_1 = layer->t - 1;                                     //  Read from the previous column
            t = layer->t;                                           //  Write to the targeted column
          }
        for(n = 0; n < layer->h; n++)
          {
            ht_1[n] = layer->H[ t_1 * layer->h + n ];
            ct_1[n] = layer->c[n];
          }
      }

    for(n = 0; n < layer->h; n++)                                   //  Write biases to vectors
      {
        i[n] = layer->bi[n];
        f[n] = layer->bf[n];
        c[n] = layer->bc[n];
        o[n] = layer->bo[n];
      }
                                                                    //  Add Ui dot ht_1 to i
    cblas_dgemv(order, transa, layer->h, layer->h, 1.0, layer->Ui, layer->h, ht_1, 1, 1.0, i, 1);
                                                                    //  Add Uf dot ht_1 to f
    cblas_dgemv(order, transa, layer->h, layer->h, 1.0, layer->Uf, layer->h, ht_1, 1, 1.0, f, 1);
                                                                    //  Add Uc dot ht_1 to c
    cblas_dgemv(order, transa, layer->h, layer->h, 1.0, layer->Uc, layer->h, ht_1, 1, 1.0, c, 1);
                                                                    //  Add Uo dot ht_1 to o
    cblas_dgemv(order, transa, layer->h, layer->h, 1.0, layer->Uo, layer->h, ht_1, 1, 1.0, o, 1);

    if(layer->d == 1)
      {
        for(n = 0; n < layer->h; n++)
          {
            i[n] += layer->Wi[n] * x[0];                            //  Add Wi dot x to i
            f[n] += layer->Wf[n] * x[0];                            //  Add Wf dot x to f
            c[n] += layer->Wc[n] * x[0];                            //  Add Wc dot x to c
            o[n] += layer->Wo[n] * x[0];                            //  Add Wo dot x to o
          }
      }
    else
      {
                                                                    //  Add Wi dot x to i
        cblas_dgemv(order, transa, layer->h, layer->d, 1.0, layer->Wi, layer->h, x, 1, 1.0, i, 1);
                                                                    //  Add Wf dot x to f
        cblas_dgemv(order, transa, layer->h, layer->d, 1.0, layer->Wf, layer->h, x, 1, 1.0, f, 1);
                                                                    //  Add Wc dot x to c
        cblas_dgemv(order, transa, layer->h, layer->d, 1.0, layer->Wc, layer->h, x, 1, 1.0, c, 1);
                                                                    //  Add Wo dot x to o
        cblas_dgemv(order, transa, layer->h, layer->d, 1.0, layer->Wo, layer->h, x, 1, 1.0, o, 1);
      }

    //  We have allocated h-by-cache space for 'H', but the structure and routine should not crash if
    //  we write more than 'cache' states. Shift everything down one column and write to the end.
    if(layer->t >= layer->cache)
      {
        for(m = 1; m < layer->cache; m++)                           //  Shift down
          {
            for(n = 0; n < layer->h; n++)
              layer->H[(m - 1) * layer->h + n] = layer->H[m * layer->h + n];
          }
      }

    for(n = 0; n < layer->h; n++)
      {
        i[n] = 1.0 / (1.0 + pow(M_E, -i[n]));                       //  i = sig(Wi*x + Ui*ht_1 + bi)
        f[n] = 1.0 / (1.0 + pow(M_E, -f[n]));                       //  f = sig(Wf*x + Uf*ht_1 + bf)
                                                                    //  c = f*ct_1 + i*tanh(Wc*x + Uc*ht_1 + bc)
        layer->c[n] = f[n] * ct_1[n] + i[n] * ((2.0 / (1.0 + pow(M_E, -2.0 * c[n]))) - 1.0);
        o[n] = 1.0 / (1.0 + pow(M_E, -o[n]));                       //  o = sig(Wo*x + Uo*ht_1 + bo)
                                                                    //  h = o*tanh(c)
        layer->H[ t * layer->h + n ] = o[n] * ((2.0 / (1.0 + pow(M_E, -2.0 * layer->c[n]))) - 1.0);
      }

    free(i);
    free(f);
    free(c);
    free(o);
    free(ct_1);
    free(ht_1);

    layer->t++;                                                     //  Increment time step

    return layer->h;                                                //  Return the size of the state
  }

void reset_LSTM(LSTMLayer* layer)
  {
    unsigned int i;
    layer->t = 0;
    for(i = 0; i < layer->h; i++)
      layer->c[i] = 0.0;
    for(i = 0; i < layer->h * layer->cache; i++)
      layer->H[i] = 0.0;
    return;
  }

#endif
