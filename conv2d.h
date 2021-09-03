#ifndef __CONV2D_H
#define __CONV2D_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Convolutional Layer as an array of one or more 2D filters:

  input mat{X} w, h       filter1      activation function vector f
 [ x11 x12 x13 x14 ]    [ w11 w12 ]   [ func1 func2 ]
 [ x21 x22 x23 x24 ]    [ w21 w22 ]
 [ x31 x32 x33 x34 ]    [ bias ]       auxiliary vector alpha
 [ x41 x42 x43 x44 ]                  [ param1 param2 ]
 [ x51 x52 x53 x54 ]      filter2
                      [ w11 w12 w13 ]
                      [ w21 w22 w23 ]
                      [ w31 w32 w33 ]
                      [ bias ]

 Filters needn't be arranged from smallest to largest; this is just for illustration.

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

typedef struct Filter2DType
  {
    unsigned int w;                                                 //  Width of the filter
    unsigned int h;                                                 //  Height of the filter

    unsigned int stride_h;                                          //  Stride by which we move the filter left to right
    unsigned int stride_v;                                          //  Stride by which we move the filter top to bottom

    unsigned char f;                                                //  Function flag, in {RELU, LEAKY_RELU, ..., THRESHOLD, LINEAR}
    double alpha;                                                   //  Function parameter (not always applicable)

    double* W;                                                      //  Array of (w * h) weights, arranged row-major, +1 for the bias
  } Filter2D;

typedef struct Conv2DLayerType
  {
    unsigned int inputW, inputH;                                    //  Dimensions of the input
    unsigned int n;                                                 //  Number of processing units in this layer =
                                                                    //  number of filters in this layer
    Filter2D* filters;                                              //  Array of 2D filter structs

    char name[LAYER_NAME_LEN];
    unsigned int outlen;                                            //  Length of the output buffer
    double* out;
  } Conv2DLayer;

/**************************************************************************************************
 Prototypes  */

unsigned int add_Conv2DFilter(unsigned int, unsigned int, Conv2DLayer*);
void setW_i_Conv2D(double*, unsigned int, Conv2DLayer*);             //  Set entirety of i-th filter; w is length width * height + 1
void setW_ij_Conv2D(double, unsigned int, unsigned int, Conv2DLayer*);
void setHorzStride_i_Conv2D(unsigned int, unsigned int, Conv2DLayer*);
void setVertStride_i_Conv2D(unsigned int, unsigned int, Conv2DLayer*);
void setF_i_Conv2D(unsigned char, unsigned int, Conv2DLayer*);       //  Set activation function of i-th filter
void setA_i_Conv2D(double, unsigned int, Conv2DLayer*);              //  Set activation function auxiliary parameter of i-th filter
void setName_Conv2D(char*, Conv2DLayer*);
void print_Conv2D(Conv2DLayer*);
unsigned int outputLen_Conv2D(Conv2DLayer*);
unsigned int run_Conv2D(double*, Conv2DLayer*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 2D-Convolutional-Layers  */

/* Add a Filter2D to an existing Conv2DLayer.
   The new filter shall have dimensions 'filterW' by 'filterH'. */
unsigned int add_Conv2DFilter(unsigned int filterW, unsigned int filterH, Conv2DLayer* layer)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("add_Conv2DFilter(%d, %d)\n", filterW, filterH);
    #endif

    layer->n++;                                                     //  Increment the number of filters/units
    if(layer->n == 1)
      {
                                                                    //  Allocate filter in 'filters' array
        if((layer->filters = (Filter2D*)malloc(sizeof(Filter2D))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer's filters array\n");
            exit(1);
          }
      }
    else
      {
                                                                    //  Allocate another filter in 'filters' array
        if((layer->filters = (Filter2D*)realloc(layer->filters, layer->n * sizeof(Filter2D))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Conv2DLayer's filters array\n");
            exit(1);
          }
      }

    layer->filters[layer->n - 1].w = filterW;                       //  Set newest filter's dimensions
    layer->filters[layer->n - 1].h = filterH;
    layer->filters[layer->n - 1].stride_h = 1;                      //  Default to stride (1, 1)
    layer->filters[layer->n - 1].stride_v = 1;
    layer->filters[layer->n - 1].f = RELU;                          //  Default to ReLU
    layer->filters[layer->n - 1].alpha = 1.0;                       //  Default to 1.0

                                                                    //  Allocate the filter matrix plus bias
    if((layer->filters[layer->n - 1].W = (double*)malloc((filterW * filterH + 1) * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate Conv2DLayer's filter\n");
        exit(1);
      }
    for(i = 0; i < filterW * filterH; i++)                          //  Generate random numbers in [ -1.0, 1.0 ]
      layer->filters[layer->n - 1].W[i] = -1.0 + ((double)rand() / ((double)RAND_MAX * 0.5));
    layer->filters[layer->n - 1].W[i] = 0.0;                        //  Defaut bias = 0.0

    layer->outlen = outputLen_Conv2D(layer);                        //  Update this layer's output buffer

    if(layer->n > 1)
      free(layer->out);

    if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate Conv2DLayer's internal output array\n");
        exit(1);
      }

    return layer->n;
  }

/* Set entirety of i-th filter; w is length width * height + 1.
   Input array 'w' is expected to be ROW-MAJOR:
        filter
   [ w0  w1  w2  ]
   [ w3  w4  w5  ]
   [ w6  w7  w8  ]  [ bias (w9) ]  */
void setW_i_Conv2D(double* w, unsigned int i, Conv2DLayer* layer)
  {
    unsigned int j;

    #ifdef __NEURON_DEBUG
    printf("setW_i_Conv2D(%d)\n", i);
    #endif

    for(j = 0; j < layer->filters[i].w * layer->filters[i].h + 1; j++)
      layer->filters[i].W[j] = w[j];
    return;
  }

/* Set filter[i], weight[j] of the given layer */
void setW_ij_Conv2D(double w, unsigned int i, unsigned int j, Conv2DLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setW_ij_Conv2D(%f, %d, %d)\n", w, i, j);
    #endif

    if(i < layer->n && j <= layer->filters[i].w * layer->filters[i].h)
      layer->filters[i].W[j] = w;
    return;
  }

/* Set filter[i]'s horizontal stride for the given layer */
void setHorzStride_i_Conv2D(unsigned int stride, unsigned int i, Conv2DLayer* layer)
  {
    if(i < layer->n)
      {
        layer->filters[i].stride_h = stride;

        layer->outlen = outputLen_Conv2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

/* Set filter[i]'s vertical stride for the given layer */
void setVertStride_i_Conv2D(unsigned int stride, unsigned int i, Conv2DLayer* layer)
  {
    if(i < layer->n)
      {
        layer->filters[i].stride_v = stride;

        layer->outlen = outputLen_Conv2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

/* Set the activation function for unit[i] of the given layer */
void setF_i_Conv2D(unsigned char func, unsigned int i, Conv2DLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setF_i_Conv2D(%d, %d)\n", func, i);
    #endif

    if(i < layer->n)
      layer->filters[i].f = func;
    return;
  }

/* Set the activation function parameter for unit[i] of the given layer */
void setA_i_Conv2D(double a, unsigned int i, Conv2DLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setA_i_Conv2D(%f, %d)\n", a, i);
    #endif

    if(i < layer->n)
      layer->filters[i].alpha = a;
    return;
  }

/* Set the name of the given Convolutional Layer */
void setName_Conv2D(char* n, Conv2DLayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

/* Print the details of the given Conv2DLayer 'layer' */
void print_Conv2D(Conv2DLayer* layer)
  {
    unsigned int i, x, y;

    #ifdef __NEURON_DEBUG
    printf("print_Conv2D()\n");
    #endif

    for(i = 0; i < layer->n; i++)                                   //  Draw each filter
      {
        printf("Filter %d\n", i);
        for(y = 0; y < layer->filters[i].h; y++)
          {
            printf("  [");
            for(x = 0; x < layer->filters[i].w; x++)
              {
                if(layer->filters[i].W[y * layer->filters[i].w + x] >= 0.0)
                  printf(" %.5f ", layer->filters[i].W[y * layer->filters[i].w + x]);
                else
                  printf("%.5f ", layer->filters[i].W[y * layer->filters[i].w + x]);
              }
            printf("]\n");
          }
        printf("  Func:  ");
        switch(layer->filters[i].f)
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
        printf("\n");
        printf("  Param: %.5f\n", layer->filters[i].alpha);
        printf("  Bias:  %.5f\n", layer->filters[i].W[layer->filters[i].h * layer->filters[i].w]);
      }
    return;
  }

/* Return the layer's output length */
unsigned int outputLen_Conv2D(Conv2DLayer* layer)
  {
    unsigned int ctr = 0;
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("outputLen_Conv2D()\n");
    #endif

    for(i = 0; i < layer->n; i++)
      ctr += (unsigned int)( (floor((double)(layer->inputW - layer->filters[i].w) / (double)layer->filters[i].stride_h) + 1.0) *
                             (floor((double)(layer->inputH - layer->filters[i].h) / (double)layer->filters[i].stride_v) + 1.0) );

    return ctr;
  }

/* Run the given input vector 'x' of length 'layer'->'inputW' * 'layer'->'inputH' through the Conv2DLayer 'layer'.
   The understanding for this function is that convolution never runs off the edge of the input,
   and that there is only one "color-channel."
   Output is stored internally in layer->out. */
unsigned int run_Conv2D(double* xvec, Conv2DLayer* layer)
  {
    unsigned int i, o = 0, c;                                       //  Iterators for the filters, the output vector, and the cache
    unsigned int s;                                                 //  Cache iterator
    unsigned int x, y;                                              //  Input iterators
    unsigned int m, n;                                              //  Filter iterators
    unsigned int filterOutputLen = 0;                               //  Length of a single filter's output vector
    double* cache;                                                  //  Output array for a single filter
    double softmaxdenom;
    double val;

    #ifdef __NEURON_DEBUG
    printf("run_Conv2D(%d, %d)\n", layer->inputW, layer->inputH);
    #endif

    for(i = 0; i < layer->n; i++)                                   //  For each filter
      {
        c = 0;
        softmaxdenom = 0.0;
        filterOutputLen = (unsigned int)( (floor((double)(layer->inputW - layer->filters[i].w) / (double)layer->filters[i].stride_h) + 1.0) *
                                          (floor((double)(layer->inputH - layer->filters[i].h) / (double)layer->filters[i].stride_v) + 1.0) );
        if((cache = (double*)malloc(filterOutputLen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate filter output buffer\n");
            exit(1);
          }

        for(y = 0; y <= layer->inputH - layer->filters[i].h; y += layer->filters[i].stride_v)
          {
            for(x = 0; x <= layer->inputW - layer->filters[i].w; x += layer->filters[i].stride_h)
              {
                val = 0.0;
                for(n = 0; n < layer->filters[i].h; n++)
                  {
                    for(m = 0; m < layer->filters[i].w; m++)
                      val += layer->filters[i].W[n * layer->filters[i].w + m] * xvec[(y + n) * layer->inputW + x + m];
                  }
                                                                    //  Add bias
                val += layer->filters[i].W[layer->filters[i].w * layer->filters[i].h];
                cache[c] = val;                                     //  Add the value to the cache
                c++;
              }
          }

        for(s = 0; s < c; s++)                                      //  In case one of the units is a softmax unit,
          softmaxdenom += pow(M_E, cache[s]);                       //  compute all exp()'s so we can sum them.

        for(s = 0; s < c; s++)
          {
            switch(layer->filters[i].f)
              {
                case RELU:                 layer->out[o] = (cache[s] > 0.0) ? cache[s] : 0.0;
                                           break;
                case LEAKY_RELU:           layer->out[o] = (cache[s] > 0.0) ? cache[s] : cache[s] * layer->filters[i].alpha;
                                           break;
                case SIGMOID:              layer->out[o] = 1.0 / (1.0 + pow(M_E, -cache[s] * layer->filters[i].alpha));
                                           break;
                case HYPERBOLIC_TANGENT:   layer->out[o] = (2.0 / (1.0 + pow(M_E, -2.0 * cache[s] * layer->filters[i].alpha))) - 1.0;
                                           break;
                case SOFTMAX:              layer->out[o] = pow(M_E, cache[s]) / softmaxdenom;
                                           break;
                case SYMMETRICAL_SIGMOID:  layer->out[o] = (1.0 - pow(M_E, -cache[s] * layer->filters[i].alpha)) / (1.0 + pow(M_E, -cache[s] * layer->filters[i].alpha));
                                           break;
                case THRESHOLD:            layer->out[o] = (cache[s] > layer->filters[i].alpha) ? 1.0 : 0.0;
                                           break;
                                                                    //  (Includes LINEAR)
                default:                   layer->out[o] = cache[s] * layer->filters[i].alpha;
              }
            o++;
          }

        free(cache);                                                //  Release the cache for this filter
      }

    return layer->outlen;
  }

#endif
