#ifndef __GRU_H
#define __GRU_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a GRU Layer as matrices and vectors:
  d = the length of a single input instance
      (that is, we may have an indefinitely long sequence of word-vectors, but each is an input instance of length 'd')
  h = the length of internal state vectors
  cache = the number of previous states to track and store

  input d-vec{x}       weights Wz              weights Wr              weights Wh
 [ x1 x2 x3 x4 ]        (h by d)                (h by d)                (h by d)
                 [ wz11 wz12 wz13 wz14 ] [ wr11 wr12 wr13 wr14 ] [ wh11 wh12 wh13 wh14 ]
                 [ wz21 wz22 wz23 wz24 ] [ wr21 wr22 wr23 wr24 ] [ wh21 wh22 wh23 wh24 ]
                 [ wz31 wz32 wz33 wz34 ] [ wr31 wr32 wr33 wr34 ] [ wh31 wh32 wh33 wh34 ]

                       weights Uz              weights Ur              weights Uh
                        (h by h)                (h by h)                (h by h)
                 [ uz11 uz12 uz13 ]      [ ur11 ur12 ur13 ]      [ uh11 uh12 uh13 ]
                 [ uz21 uz22 uz23 ]      [ ur21 ur22 ur23 ]      [ uh21 uh22 uh23 ]
                 [ uz31 uz32 uz33 ]      [ ur31 ur32 ur33 ]      [ uh31 uh32 uh33 ]

                     bias h-vec{bz}          bias h-vec{br}          bias h-vec{bh}
                 [ bz1 ]                 [ br1 ]                 [ bh1 ]
                 [ bz2 ]                 [ br2 ]                 [ bh2 ]
                 [ bz3 ]                 [ br3 ]                 [ bh3 ]

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

typedef struct GRULayerType
  {
    unsigned int d;                                                 //  Dimensionality of input vector
    unsigned int h;                                                 //  Dimensionality of hidden state vector
    unsigned int cache;                                             //  The number of states to keep in memory:
                                                                    //  when 't' exceeds this, shift out.
    unsigned int t;                                                 //  The time step
                                                                    //  W matrices are (h by d)
    double* Wz;                                                     //  Update gate weights
    double* Wr;                                                     //  Reset gate weights
    double* Wh;                                                     //  Output gate weights
                                                                    //  U matrices are (h by h)
    double* Uz;                                                     //  Recurrent connection update gate weights
    double* Ur;                                                     //  Recurrent connection reset gate weights
    double* Uh;                                                     //  Recurrent connection output gate weights
                                                                    //  Bias vectors are length h
    double* bz;                                                     //  Update gate bias
    double* br;                                                     //  Reset gate bias
    double* bh;                                                     //  Output gate bias

    double* H;                                                      //  Hidden state cache matrix (h by cache)
    char name[LAYER_NAME_LEN];
  } GRULayer;

/**************************************************************************************************
 Prototypes  */

void setWz_GRU(double*, GRULayer*);                                 //  Set entirety of Wz weight matrix
void setWr_GRU(double*, GRULayer*);                                 //  Set entirety of Wr weight matrix
void setWh_GRU(double*, GRULayer*);                                 //  Set entirety of Wh weight matrix
void setWz_ij_GRU(double, unsigned int, unsigned int, GRULayer*);   //  Set element [i, j] of Wz weight matrix
void setWr_ij_GRU(double, unsigned int, unsigned int, GRULayer*);   //  Set element [i, j] of Wr weight matrix
void setWh_ij_GRU(double, unsigned int, unsigned int, GRULayer*);   //  Set element [i, j] of Wh weight matrix
void setUz_GRU(double*, GRULayer*);                                 //  Set entirety of Uz weight matrix
void setUr_GRU(double*, GRULayer*);                                 //  Set entirety of Ur weight matrix
void setUh_GRU(double*, GRULayer*);                                 //  Set entirety of Uh weight matrix
void setUz_ij_GRU(double, unsigned int, unsigned int, GRULayer*);   //  Set element [i, j] of Uz weight matrix
void setUr_ij_GRU(double, unsigned int, unsigned int, GRULayer*);   //  Set element [i, j] of Ur weight matrix
void setUh_ij_GRU(double, unsigned int, unsigned int, GRULayer*);   //  Set element [i, j] of Uh weight matrix
void setbz_GRU(double*, GRULayer*);                                 //  Set entirety of bz bias vector
void setbr_GRU(double*, GRULayer*);                                 //  Set entirety of br bias vector
void setbh_GRU(double*, GRULayer*);                                 //  Set entirety of bh bias vector
void setbz_i_GRU(double, unsigned int, GRULayer*);                  //  Set i-th element of bz bias vector
void setbr_i_GRU(double, unsigned int, GRULayer*);                  //  Set i-th element of br bias vector
void setbh_i_GRU(double, unsigned int, GRULayer*);                  //  Set i-th element of bh bias vector
void setName_GRU(char*, GRULayer*);
void print_GRU(GRULayer*);
unsigned int outputLen_GRU(GRULayer*);
unsigned int run_GRU(double*, GRULayer*);
void reset_GRU(GRULayer*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 GRU-Layers  */

/* Set the entirety of the Wz matrix of the given layer using the given array */
void setWz_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->d * layer->h; i++)
      layer->Wz[i] = w[i];
    return;
  }

/* Set the entirety of the Wr matrix of the given layer using the given array */
void setWr_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->d * layer->h; i++)
      layer->Wr[i] = w[i];
    return;
  }

/* Set the entirety of the Wh matrix of the given layer using the given array */
void setWh_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->d * layer->h; i++)
      layer->Wh[i] = w[i];
    return;
  }

/* Set column[i], row[j] of the given layer, Wz matrix */
void setWz_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWz_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(j * layer->d + i < layer->h * layer->d)
      layer->Wz[j * layer->d + i] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Wr matrix */
void setWr_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWr_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(j * layer->d + i < layer->h * layer->d)
      layer->Wr[j * layer->d + i] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Wh matrix */
void setWh_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWh_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(j * layer->d + i < layer->h * layer->d)
      layer->Wh[j * layer->d + i] = w;
    return;
  }

/* Set the entirety of the Uz matrix of the given layer using the given array */
void setUz_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h * layer->h; i++)
      layer->Uz[i] = w[i];
    return;
  }

/* Set the entirety of the Ur matrix of the given layer using the given array */
void setUr_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h * layer->h; i++)
      layer->Ur[i] = w[i];
    return;
  }

/* Set the entirety of the Uh matrix of the given layer using the given array */
void setUh_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h * layer->h; i++)
      layer->Uh[i] = w[i];
    return;
  }

/* Set column[i], row[j] of the given layer, Uz matrix */
void setUz_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUz_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(j * layer->h + i < layer->h * layer->h)
      layer->Uz[j * layer->h + i] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Ur matrix */
void setUr_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUr_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(j * layer->h + i < layer->h * layer->h)
      layer->Ur[j * layer->h + i] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Uh matrix */
void setUh_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUh_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(j * layer->h + i < layer->h * layer->h)
      layer->Uh[j * layer->h + i] = w;
    return;
  }

/* Set the entirety of the bz vector of the given layer using the given array */
void setbz_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h; i++)
      layer->bz[i] = w[i];
    return;
  }

/* Set the entirety of the br vector of the given layer using the given array */
void setbr_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h; i++)
      layer->br[i] = w[i];
    return;
  }

/* Set the entirety of the bh vector of the given layer using the given array */
void setbh_GRU(double* w, GRULayer* layer)
  {
    unsigned int i;
    for(i = 0; i < layer->h; i++)
      layer->bh[i] = w[i];
    return;
  }

/* Set element [i] of the given layer, bz vector */
void setbz_i_GRU(double w, unsigned int i, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setbz_i_GRU(%f, %d)\n", w, i);
    #endif

    if(i < layer->h)
      layer->bz[i] = w;
    return;
  }

/* Set element [i] of the given layer, br vector */
void setbr_i_GRU(double w, unsigned int i, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setbr_i_GRU(%f, %d)\n", w, i);
    #endif

    if(i < layer->h)
      layer->br[i] = w;
    return;
  }

/* Set element [i] of the given layer, bh vector */
void setbh_i_GRU(double w, unsigned int i, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setbh_i_GRU(%f, %d)\n", w, i);
    #endif

    if(i < layer->h)
      layer->bh[i] = w;
    return;
  }

void setName_GRU(char* n, GRULayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

/* Print the details of the given GRULayer 'layer' */
void print_GRU(GRULayer* layer)
  {
    unsigned int i, j;

    #ifdef __NEURON_DEBUG
    printf("print_GRU()\n");
    #endif

    printf("Input dimensionality d = %d\n", layer->d);
    printf("State dimensionality h = %d\n", layer->h);
    printf("State cache size       = %d\n", layer->cache);

    printf("Wz (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->d; j++)
          printf(" %.5f", layer->Wz[i * layer->d + j]);
        printf(" ]\n");
      }
    printf("\nWr (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->d; j++)
          printf(" %.5f", layer->Wr[i * layer->d + j]);
        printf(" ]\n");
      }
    printf("\nWh (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->d; j++)
          printf(" %.5f", layer->Wh[i * layer->d + j]);
        printf(" ]\n");
      }

    printf("\nUz (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Uz[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("\nUr (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Ur[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("\nUh (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Uh[i * layer->h + j]);
        printf(" ]\n");
      }

    printf("\nbz (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->bz[i]);
    printf("\nbr (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->br[i]);
    printf("\nbh (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->bh[i]);

    return;
  }

unsigned int outputLen_GRU(GRULayer* layer)
  {
    return layer->h;
  }

/* Run the given input vector 'x' of length 'layer'->'d' through the GRULayer 'layer'.
   Output is stored internally in layer->H.
   Write to the 'layer'->'t'-th column and increment t.
   If 'layer'->'t' exceeds 'layer'->'cache', shift everything down. */
unsigned int run_GRU(double* x, GRULayer* layer)
  {
    unsigned int n, m;
    double* z;
    double* r;
    double* h;
    double* hprime;                                                 //  Intermediate Hadamard product r * ht_1
    double* ht_1;                                                   //  Time t - 1
    unsigned int t_1;                                               //  Where we READ FROM
    unsigned int t;                                                 //  Where we WRITE TO
                                                                    //  layer->t increases indefinitely
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE transa;

    #ifdef __NEURON_DEBUG
    printf("run_GRU(%d)\n", layer->h);
    #endif

    order = CblasRowMajor;
    transa = CblasTrans;

    if((z = (double*)malloc(layer->h * sizeof(double))) == NULL)    //  Allocate vec{z}
      {
        printf("ERROR: Unable to allocate GRU vector z\n");
        exit(1);
      }
    if((r = (double*)malloc(layer->h * sizeof(double))) == NULL)    //  Allocate vec{r}
      {
        printf("ERROR: Unable to allocate GRU vector r\n");
        exit(1);
      }
    if((h = (double*)malloc(layer->h * sizeof(double))) == NULL)    //  Allocate vec{h}
      {
        printf("ERROR: Unable to allocate GRU vector h\n");
        exit(1);
      }
                                                                    //  Allocate vec{h'}
    if((hprime = (double*)malloc(layer->h * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRU vector hprime\n");
        exit(1);
      }
    if((ht_1 = (double*)malloc(layer->h * sizeof(double))) == NULL) //  Allocate vec{ht_1}
      {
        printf("ERROR: Unable to allocate GRU vector ht_1\n");
        exit(1);
      }

    if(layer->t == 0)                                               //  Timestep layer->t = 0 uses the zero-vectors for t - 1
      {
        t_1 = 0;
        t = 0;
        for(n = 0; n < layer->h; n++)                               //  Write zeroes to h(t-1)
          ht_1[n] = 0.0;
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
          ht_1[n] = layer->H[ n * layer->cache + t_1 ];
      }

    for(n = 0; n < layer->h; n++)                                   //  Write biases to vectors z and r
      {
        z[n] = layer->bz[n];
        r[n] = layer->br[n];
        h[n] = 0.0;                                                 //  Blank out h
      }
                                                                    //  Add Uz dot ht_1 to z
    cblas_dgemv(order, transa, layer->h, layer->h, 1.0, layer->Uz, layer->h, ht_1, 1, 1.0, z, 1);
                                                                    //  Add Ur dot ht_1 to r
    cblas_dgemv(order, transa, layer->h, layer->h, 1.0, layer->Ur, layer->h, ht_1, 1, 1.0, r, 1);

    if(layer->d == 1)                                               //  Add Wz dot x to z; Wr dot x to r
      {
        for(n = 0; n < layer->h; n++)
          {
            z[n] += layer->Wz[n] * x[0];                            //  Add Wz dot x to z
            r[n] += layer->Wr[n] * x[0];                            //  Add Wr dot x to r
          }
      }
    else
      {
                                                                    //  Add Wz dot x to z
        cblas_dgemv(order, transa, layer->h, layer->d, 1.0, layer->Wz, layer->h, x, 1, 1.0, z, 1);
                                                                    //  Add Wr dot x to r
        cblas_dgemv(order, transa, layer->h, layer->d, 1.0, layer->Wr, layer->h, x, 1, 1.0, r, 1);
      }
    for(n = 0; n < layer->h; n++)                                   //  Apply sigmoid function to z and r vectors
      {
        z[n] = 1.0 / (1.0 + pow(M_E, -z[n]));                       //  z = sig(Wz.x + Uz.ht_1 + bz)
        r[n] = 1.0 / (1.0 + pow(M_E, -r[n]));                       //  r = sig(Wr.x + Ur.ht_1 + br)
      }

    for(n = 0; n < layer->h; n++)                                   //  h' = r * ht_1
      hprime[n] = r[n] * ht_1[n];
                                                                    //  Set h = Uh.h' = Uh.(r * ht_1)
    cblas_dgemv(order, transa, layer->h, layer->h, 1.0, layer->Uh, layer->h, hprime, 1, 1.0, h, 1);
    if(layer->d == 1)                                               //  Add Wh dot x to h
      {
        for(n = 0; n < layer->h; n++)
          h[n] += layer->Wh[n] * x[0];                              //  Add Wh dot x to h
      }
    else
      cblas_dgemv(order, transa, layer->h, layer->d, 1.0, layer->Wh, layer->h, x, 1, 1.0, h, 1);
    for(n = 0; n < layer->h; n++)                                   //  Add bias to h vector
      h[n] += layer->bh[n];

    //  Now h = Wh.x + Uh.(r * ht_1) + bh

    //  We have allocated h-by-cache space for 'H', but the structure and routine should not crash if
    //  we write more than 'cache' states. Shift everything down one column and write to the end.
    if(layer->t >= layer->cache)
      {
        for(m = 1; m < layer->cache; m++)                           //  Shift down
          {
            for(n = 0; n < layer->h; n++)
              layer->H[n * layer->cache + m - 1] = layer->H[n * layer->cache + m];
          }
      }

    for(n = 0; n < layer->h; n++)
      {
                                                                    //  h = z*ht_1 + (1-z)*tanh(h)
        layer->H[ n * layer->cache + t ] = z[n] * ht_1[n] + (1.0 - z[n]) * ((2.0 / (1.0 + pow(M_E, -2.0 * h[n]))) - 1.0);
      }

    free(z);
    free(r);
    free(hprime);
    free(h);
    free(ht_1);

    layer->t++;                                                     //  Increment time step

    return layer->h;                                                //  Return the size of the state
  }

void reset_GRU(GRULayer* layer)
  {
    unsigned int i;
    layer->t = 0;
    for(i = 0; i < layer->h * layer->cache; i++)
      layer->H[i] = 0.0;
    return;
  }

#endif
