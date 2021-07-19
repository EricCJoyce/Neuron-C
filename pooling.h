#ifndef __POOLING_H
#define __POOLING_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 Model a Pooling Layer as 2D input dimensions and an array of 2D pools.
  inputW = width of the input
  inputH = height of the input

 Each pool has a 2D shape, two dimensions for stride, and a function/type:
  stride_h = horizontal stride of the pool
  stride_v = vertical stride of the pool
  f = {MAX_POOL, AVG_POOL, MIN_POOL, MEDIAN_POOL}

    input mat{X}          pool     output for s = (1, 1)     output for s = (2, 2)
 [ x11 x12 x13 x14 ]    [ . . ]   [ y11  y12  y13 ]         [ y11  y12 ]
 [ x21 x22 x23 x24 ]    [ . . ]   [ y21  y22  y23 ]         [ y21  y22 ]
 [ x31 x32 x33 x34 ]              [ y31  y32  y33 ]
 [ x41 x42 x43 x44 ]              [ y41  y42  y43 ]
 [ x51 x52 x53 x54 ]

 Pools needn't be arranged from smallest to largest or in any order.

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

#define MAX_POOL     0
#define MIN_POOL     1
#define AVG_POOL     2
#define MEDIAN_POOL  3

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __NEURON_DEBUG 1
*/

typedef struct Pool2DType
  {
    unsigned int w;                                                 //  Width of the pool
    unsigned int h;                                                 //  Height of the pool
    unsigned int stride_h;                                          //  Stride by which we move the pool left to right
    unsigned int stride_v;                                          //  Stride by which we move the pool top to bottom
    unsigned char f;                                                //  In {MAX_POOL, MIN_POOL, AVG_POOL, MEDIAN_POOL}
  } Pool2D;

typedef struct Pool2DLayerType
  {
    unsigned int inputW, inputH;                                    //  Dimensions of the input
    Pool2D* pools;                                                  //  Array of Pool2Ds
    unsigned int n;                                                 //  Length of that array

    double* out;
    unsigned int outlen;                                            //  Length of the output buffer
    char name[LAYER_NAME_LEN];
  } Pool2DLayer;

/**************************************************************************************************
 Prototypes  */

unsigned int add_2DPool(unsigned int, unsigned int, Pool2DLayer*);
void setW_Pool2D(unsigned int, unsigned int, Pool2DLayer*);
void setH_Pool2D(unsigned int, unsigned int, Pool2DLayer*);
void setHorzStride_Pool2D(unsigned int, unsigned int, Pool2DLayer*);
void setVertStride_Pool2D(unsigned int, unsigned int, Pool2DLayer*);
void setFunc_Pool2D(unsigned char, unsigned int, Pool2DLayer*);
void setName_Pool2D(char*, Pool2DLayer*);
void print_Pool2D(Pool2DLayer*);
unsigned int outputLen_Pool2D(Pool2DLayer*);
unsigned int run_Pool2D(double*, Pool2DLayer*);
void pooling_quicksort(bool, double**, unsigned int, unsigned int);
unsigned int pooling_partition(bool, double**, unsigned int, unsigned int);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 Pooling-Layers  */

/* Add a Pool2D to an existing Pool2DLayer.
   The new pool shall have dimensions 'poolW' by 'poolH'. */
unsigned int add_2DPool(unsigned int poolW, unsigned int poolH, Pool2DLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("add_2DPool(%d, %d)\n", poolW, poolH);
    #endif

    layer->n++;                                                     //  Increment the number of pools/units
    if(layer->n == 1)
      {
                                                                    //  Allocate pool in 'pools' array
        if((layer->pools = (Pool2D*)malloc(sizeof(Pool2D))) == NULL)
          {
            printf("ERROR: Unable to allocate Pool2DLayer's pools array\n");
            exit(1);
          }
      }
    else
      {
                                                                    //  Allocate another pool in 'pools' array
        if((layer->pools = (Pool2D*)realloc(layer->pools, layer->n * sizeof(Pool2D))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Pool2DLayer's pools array\n");
            exit(1);
          }
      }

    layer->pools[layer->n - 1].w = poolW;                           //  Set newest filter's dimensions
    layer->pools[layer->n - 1].h = poolH;
    layer->pools[layer->n - 1].stride_h = 1;                        //  Default to stride (1, 1)
    layer->pools[layer->n - 1].stride_v = 1;
    layer->pools[layer->n - 1].f = MAX_POOL;                        //  Default to max-pooling

    layer->outlen = outputLen_Pool2D(layer);

    if(layer->n > 1)
      free(layer->out);

    if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate Pool2DLayer's internal output array\n");
        exit(1);
      }

    return layer->n;
  }

/* (Re)Set the width of the i-th pool in the given layer. */
void setW_Pool2D(unsigned int w, unsigned int i, Pool2DLayer* layer)
  {
    if(i < layer->n)
      {
        layer->pools[i].w = w;

        layer->outlen = outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Pool2DLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

/* (Re)Set the height of the i-th pool in the given layer. */
void setH_Pool2D(unsigned int h, unsigned int i, Pool2DLayer* layer)
  {
    if(i < layer->n)
      {
        layer->pools[i].h = h;

        layer->outlen = outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Pool2DLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

/* Set the horizontal stride for the i-th pool in the given layer. */
void setHorzStride_Pool2D(unsigned int stride, unsigned int i, Pool2DLayer* layer)
  {
    if(i < layer->n)
      {
        layer->pools[i].stride_h = stride;

        layer->outlen = outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Pool2DLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

/* Set the vertical stride for the i-th pool in the given layer. */
void setVertStride_Pool2D(unsigned int stride, unsigned int i, Pool2DLayer* layer)
  {
    if(i < layer->n)
      {
        layer->pools[i].stride_v = stride;

        layer->outlen = outputLen_Pool2D(layer);                    //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Pool2DLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

/* Set the function for the i-th pool in the given layer. */
void setFunc_Pool2D(unsigned char f, unsigned int i, Pool2DLayer* layer)
  {
    if(i < layer->n)
      layer->pools[i].f = f;
    return;
  }

/* Set the name of the given Pooling Layer */
void setName_Pool2D(char* n, Pool2DLayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

/* Print the details of the given Pool2DLayer 'layer' */
void print_Pool2D(Pool2DLayer* layer)
  {
    unsigned int i, x, y;

    #ifdef __NEURON_DEBUG
    printf("print_Pool2D()\n");
    #endif

    printf("Input Shape = (%d, %d)\n", layer->inputW, layer->inputH);

    for(i = 0; i < layer->n; i++)                                   //  Draw each pool
      {
        printf("Pool %d\n", i);
        for(y = 0; y < layer->pools[i].h; y++)
          {
            printf("  [");
            for(x = 0; x < layer->pools[i].w; x++)
              printf(" . ");
            printf("]");
            if(y < layer->pools[i].h - 1)
              printf("\n");
          }
        printf("  Func:  ");
        switch(layer->pools[i].f)
          {
            case MAX_POOL:     printf("max.  ");  break;
            case MIN_POOL:     printf("min.  ");  break;
            case AVG_POOL:     printf("avg.  ");  break;
            case MEDIAN_POOL:  printf("med.  ");  break;
          }
        printf("Stride: (%d, %d)\n", layer->pools[i].stride_h, layer->pools[i].stride_v);
      }

    return;
  }

/* Return the layer's output length */
unsigned int outputLen_Pool2D(Pool2DLayer* layer)
  {
    unsigned int i;
    unsigned int ctr = 0;

    for(i = 0; i < layer->n; i++)
      ctr += (unsigned int)(floor((double)(layer->inputW - layer->pools[i].w + 1) / (double)layer->pools[i].stride_h) *
                            floor((double)(layer->inputH - layer->pools[i].h + 1) / (double)layer->pools[i].stride_v));

    return ctr;
  }

/* Run the given input vector 'x' of length 'layer'->'inputW' * 'layer'->'inputH' through the Pool2DLayer 'layer'.
   The understanding for this function is that pooling never runs off the edge of the input, and that there is
   only one "color-channel."
   Output is stored internally in layer->out. */
unsigned int run_Pool2D(double* xvec, Pool2DLayer* layer)
  {
    unsigned int i = 0;                                             //  Pool array iterator
    unsigned int o = 0;                                             //  Output iterator
    unsigned int ctr;                                               //  Only used in median pooling
    unsigned int x, y;                                              //  2D input iterators
    unsigned int m, n;                                              //  2D pool iterators

    double* cache;                                                  //  Intermediate buffer
    unsigned int cacheLen;                                          //  Length of that buffer
    bool cacheLenEven = false;
    unsigned int index;                                             //  Used in median pooling
    double val;

    #ifdef __NEURON_DEBUG
    printf("run_Pool2D(%d, %d)\n", layer->inputW, layer->inputH);
    #endif

    for(i = 0; i < layer->n; i++)                                   //  For each pool
      {
        switch(layer->pools[i].f)                                   //  Prefer one "if" per layer to one "if" per iteration
          {
            case MAX_POOL:    for(y = 0; y <= layer->inputH - layer->pools[i].h; y += layer->pools[i].stride_v)
                                {
                                  for(x = 0; x <= layer->inputW - layer->pools[i].w; x += layer->pools[i].stride_h)
                                    {
                                      val = -INFINITY;
                                      for(n = 0; n < layer->pools[i].h; n++)
                                        {
                                          for(m = 0; m < layer->pools[i].w; m++)
                                            {
                                              if(xvec[(y + n) * layer->inputW + x + m] > val)
                                                val = xvec[(y + n) * layer->inputW + x + m];
                                            }
                                        }
                                      layer->out[o] = val;
                                      o++;
                                    }
                                }
                              break;
            case MIN_POOL:    for(y = 0; y <= layer->inputH - layer->pools[i].h; y += layer->pools[i].stride_v)
                                {
                                  for(x = 0; x <= layer->inputW - layer->pools[i].w; x += layer->pools[i].stride_h)
                                    {
                                      val = INFINITY;
                                      for(n = 0; n < layer->pools[i].h; n++)
                                        {
                                          for(m = 0; m < layer->pools[i].w; m++)
                                            {
                                              if(xvec[(y + n) * layer->inputW + x + m] < val)
                                                val = xvec[(y + n) * layer->inputW + x + m];
                                            }
                                        }
                                      layer->out[o] = val;
                                      o++;
                                    }
                                }
                              break;
            case AVG_POOL:    for(y = 0; y <= layer->inputH - layer->pools[i].h; y += layer->pools[i].stride_v)
                                {
                                  for(x = 0; x <= layer->inputW - layer->pools[i].w; x += layer->pools[i].stride_h)
                                    {
                                      val = 0.0;
                                      for(n = 0; n < layer->pools[i].h; n++)
                                        {
                                          for(m = 0; m < layer->pools[i].w; m++)
                                            val += xvec[(y + n) * layer->inputW + x + m];
                                        }
                                      layer->out[o] = val / (layer->pools[i].w * layer->pools[i].h);
                                      o++;
                                    }
                                }
                              break;
            case MEDIAN_POOL: cacheLen = layer->pools[i].w * layer->pools[i].h;
                              if((cache = malloc(cacheLen * sizeof(double))) == NULL)
                                {
                                  printf("ERROR: Unable to allocate median pooling cache array\n");
                                  exit(1);
                                }
                              cacheLenEven = (cacheLen % 2 == 0);
                              if(cacheLenEven)
                                index = cacheLenEven / 2 - 1;
                              else
                                index = (cacheLenEven - 1) / 2;

                              for(y = 0; y <= layer->inputH - layer->pools[i].h; y += layer->pools[i].stride_v)
                                {
                                  for(x = 0; x <= layer->inputW - layer->pools[i].w; x += layer->pools[i].stride_h)
                                    {
                                      ctr = 0;
                                      for(n = 0; n < layer->pools[i].h; n++)
                                        {
                                          for(m = 0; m < layer->pools[i].w; m++)
                                            {
                                              cache[ctr] = xvec[(y + n) * layer->inputW + x + m];
                                              ctr++;
                                            }
                                        }

                                      pooling_quicksort(true, &cache, 0, cacheLen - 1);

                                      if(cacheLenEven)
                                        layer->out[o] = 0.5 * (cache[index] + cache[index + 1]);
                                      else
                                        layer->out[o] = cache[index];

                                      o++;
                                    }
                                }
                              free(cache);
                              break;
          }
      }

    return layer->outlen;
  }

void pooling_quicksort(bool desc, double** a, unsigned int lo, unsigned int hi)
  {
    unsigned int p;

    if(lo < hi)
      {
        p = pooling_partition(desc, a, lo, hi);

        if(p > 0)                                                   //  PREVENT ROLL-OVER TO UINT_MAX
          pooling_quicksort(desc, a, lo, p - 1);                    //  Left side: start quicksort
        if(p < UINT_MAX)                                            //  PREVENT ROLL-OVER TO 0x0000
          pooling_quicksort(desc, a, p + 1, hi);                    //  Right side: start quicksort
      }

    return;
  }

unsigned int pooling_partition(bool desc, double** a, unsigned int lo, unsigned int hi)
  {
    double pivot = (*a)[hi];
    unsigned int i = lo;
    unsigned int j;
    double tmpFloat;
    bool trigger;

    for(j = lo; j < hi; j++)
      {
        if(desc)
          trigger = ((*a)[j] > pivot);                              //  SORT DESCENDING
        else
          trigger = ((*a)[j] < pivot);                              //  SORT ASCENDING

        if(trigger)
          {
            tmpFloat = (*a)[i];                                     //  Swap a[i] with a[j]
            (*a)[i]   = (*a)[j];
            (*a)[j]   = tmpFloat;

            i++;
          }
      }

    tmpFloat = (*a)[i];                                             //  Swap a[i] with a[hi]
    (*a)[i]  = (*a)[hi];
    (*a)[hi] = tmpFloat;

    return i;
  }

#endif
