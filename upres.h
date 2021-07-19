#ifndef __UPRES_H
#define __UPRES_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 An upres layer serves to prepare input for (transposed) convolution.
  s = stride
  p = padding

    input mat{X}         output for s = 1, p = 0        output for s = 1, p = 1
 [ x11 x12 x13 x14 ]    [ x11 0 x12 0 x13 0 x14 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x21 x22 x23 x24 ]    [  0  0  0  0  0  0  0  ]    [ 0 x11 0 x12 0 x13 0 x14 0 ]
 [ x31 x32 x33 x34 ]    [ x21 0 x22 0 x23 0 x24 ]    [ 0  0  0  0  0  0  0  0  0 ]
 [ x41 x42 x43 x44 ]    [  0  0  0  0  0  0  0  ]    [ 0 x21 0 x22 0 x23 0 x24 0 ]
 [ x51 x52 x53 x54 ]    [ x31 0 x32 0 x33 0 x34 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x31 0 x32 0 x33 0 x34 0 ]
                        [ x41 0 x42 0 x43 0 x44 ]    [ 0  0  0  0  0  0  0  0  0 ]
                        [  0  0  0  0  0  0  0  ]    [ 0 x41 0 x42 0 x43 0 x44 0 ]
                        [ x51 0 x52 0 x53 0 x54 ]    [ 0  0  0  0  0  0  0  0  0 ]
                                                     [ 0 x51 0 x52 0 x53 0 x54 0 ]
                                                     [ 0  0  0  0  0  0  0  0  0 ]

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

#define FILL_ZERO    0                                              /* Fill strides or pad using zeroes */
#define FILL_SAME    1                                              /* Fill strides or pad using duplicates of the nearest value */
#define FILL_INTERP  2                                              /* Fill strides or pad using bilinear interpolation */

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __NEURON_DEBUG 1
*/

typedef struct UpresParamsType
  {
    unsigned int stride_h;                                          //  Horizontal Stride: number of columns to put between input columns
    unsigned int stride_v;                                          //  Vertical Stride: number of rows to put between input rows
    unsigned int padding_h;                                         //  Horizontal Padding: depth of pixels appended to the source border, left and right
    unsigned int padding_v;                                         //  Vertical Padding: depth of pixels appended to the source border, top and bottom

    unsigned char sMethod;                                          //  In {FILL_ZERO, FILL_SAME, FILL_INTERP}
    unsigned char pMethod;
  } UpresParams;

typedef struct UpresLayerType
  {
    unsigned int inputW, inputH;                                    //  Dimensions of the input
    UpresParams* params;                                            //  Array of Up-resolution parameters structures
    unsigned int n;                                                 //  Length of that array

    double* out;
    unsigned int outlen;                                            //  Length of the output buffer
    char name[LAYER_NAME_LEN];
  } UpresLayer;

/**************************************************************************************************
 Prototypes  */

unsigned int add_UpResParams(unsigned int, unsigned int, UpresLayer*);
void setHorzStride_Upres(unsigned int, unsigned int, UpresLayer*);
void setVertStride_Upres(unsigned int, unsigned int, UpresLayer*);
void setHorzPad_Upres(unsigned int, unsigned int, UpresLayer*);
void setVertPad_Upres(unsigned int, unsigned int, UpresLayer*);
void setStrideMethod_Upres(unsigned char, unsigned int, UpresLayer*);
void setPaddingMethod_Upres(unsigned char, unsigned int, UpresLayer*);
void setName_Upres(char*, UpresLayer*);
void print_Upres(UpresLayer*);
unsigned int outputLen_Upres(UpresLayer*);
unsigned int run_Upres(double*, UpresLayer*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 Upres-Layers  */

/* Add an "up-ressing" to an existing UpresLayer.
   The new "up-ressing" shall have stride 'stride' and padding 'padding'. */
unsigned int add_UpResParams(unsigned int stride, unsigned int padding, UpresLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("add_UpResParams(%d, %d)\n", stride, padding);
    #endif

    layer->n++;                                                     //  Increment the number of params/units
    if(layer->n == 1)
      {
                                                                    //  Allocate UpresParams in 'params' array
        if((layer->params = (UpresParams*)malloc(sizeof(UpresParams))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresParams's parameters array\n");
            exit(1);
          }
      }
    else
      {
                                                                    //  Allocate another UpresParams in 'params' array
        if((layer->params = (UpresParams*)realloc(layer->params, layer->n * sizeof(UpresParams))) == NULL)
          {
            printf("ERROR: Unable to re-allocate UpresParams's parameters array\n");
            exit(1);
          }
      }

    layer->params[layer->n - 1].stride_h = stride;                  //  Set newest up-ressing's stride (the same)
    layer->params[layer->n - 1].stride_v = stride;

    layer->params[layer->n - 1].padding_h = padding;                //  Set newest up-ressing's padding (the same)
    layer->params[layer->n - 1].padding_v = padding;

    layer->params[layer->n - 1].sMethod = FILL_ZERO;                //  Default to filling the strided rows and columns with zero
    layer->params[layer->n - 1].pMethod = FILL_ZERO;                //  Default to padding the input rows and columns with zero

    layer->outlen = outputLen_Upres(layer);

    if(layer->n > 1)
      free(layer->out);

    if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate UpresLayer's internal output array\n");
        exit(1);
      }

    return layer->n;
  }

void setHorzStride_Upres(unsigned int stride, unsigned int i, UpresLayer* layer)
  {
    if(i < layer->n)
      {
        layer->params[i].stride_h = stride;

        layer->outlen = outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

void setVertStride_Upres(unsigned int stride, unsigned int i, UpresLayer* layer)
  {
    if(i < layer->n)
      {
        layer->params[i].stride_v = stride;

        layer->outlen = outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

void setHorzPad_Upres(unsigned int pad, unsigned int i, UpresLayer* layer)
  {
    if(i < layer->n)
      {
        layer->params[i].padding_h = pad;

        layer->outlen = outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

void setVertPad_Upres(unsigned int pad, unsigned int i, UpresLayer* layer)
  {
    if(i < layer->n)
      {
        layer->params[i].padding_v = pad;

        layer->outlen = outputLen_Upres(layer);                     //  Re-compute layer's output length and reallocate its output buffer
        free(layer->out);
        if((layer->out = (double*)malloc(layer->outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresLayer's internal output array\n");
            exit(1);
          }
      }
    return;
  }

void setStrideMethod_Upres(unsigned char strideMethod, unsigned int i, UpresLayer* layer)
  {
    if(i < layer->n)
      layer->params[i].sMethod = strideMethod;
    return;
  }

void setPaddingMethod_Upres(unsigned char padMethod, unsigned int i, UpresLayer* layer)
  {
    if(i < layer->n)
      layer->params[i].pMethod = padMethod;
    return;
  }

/* Set the name of the given Upres Layer */
void setName_Upres(char* n, UpresLayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

/* Print the details of the given UpresLayer 'layer' */
void print_Upres(UpresLayer* layer)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("print_Upres()\n");
    #endif

    printf("Input Shape = (%d, %d)\n", layer->inputW, layer->inputH);

    for(i = 0; i < layer->n; i++)                                   //  Draw each up-ressing
      {
        printf("Parameters %d\n", i);
        printf("  H.stride  = %d\n", layer->params[i].stride_h);
        printf("  V.stride  = %d\n", layer->params[i].stride_v);
        printf("  H.padding = %d\n", layer->params[i].padding_h);
        printf("  V.padding = %d\n", layer->params[i].padding_v);
        printf("  Stride    = ");
        switch(layer->params[i].sMethod)
          {
            case FILL_ZERO:    printf("zero");         break;
            case FILL_SAME:    printf("same");         break;
            case FILL_INTERP:  printf("interpolate");  break;
          }
        printf("\n  Padding   = ");
        switch(layer->params[i].pMethod)
          {
            case FILL_ZERO:    printf("zero");         break;
            case FILL_SAME:    printf("same");         break;
            case FILL_INTERP:  printf("interpolate");  break;
          }
        printf("\n");
      }

    return;
  }

/* Return the layer's output length */
unsigned int outputLen_Upres(UpresLayer* layer)
  {
    unsigned int i;
    unsigned int ctr = 0;

    for(i = 0; i < layer->n; i++)
      ctr += (layer->inputW * (layer->params[i].stride_h + 1) - layer->params[i].stride_h + layer->params[i].padding_h + layer->params[i].padding_h) *
             (layer->inputH * (layer->params[i].stride_v + 1) - layer->params[i].stride_v + layer->params[i].padding_v + layer->params[i].padding_v);

    return ctr;
  }

/* Run the given input vector 'x' of length 'layer'->'inputW' * 'layer'->'inputH' through the UpresLayer 'layer'.
   The understanding for this function is that there is only one "color-channel."
   Output is stored internally in layer->out. */
unsigned int run_Upres(double* xvec, UpresLayer* layer)
  {
    unsigned int i;                                                 //  Up-ressing iterator
    unsigned int o = 0;                                             //  Output iterator
    unsigned int x, y;                                              //  Iterators
    unsigned int x_src, y_src;                                      //  Used in zero-fill stride to iterate over source
    unsigned int cache_w, cache_h;                                  //  Dimensions of the inner-rectangle (without padding applied yet)
    unsigned int output_w, output_h;                                //  Dimensions per up-ressing of the padded output
    double x_prime, y_prime;                                        //  Inter-pixel locations in source
    double a, b;                                                    //  Fractional parts of clipped doubles
    double sc_inv_h, sc_inv_v;                                      //  Scaling factors of the inverse transformation
    double val;                                                     //  Stores and compares neighboring pixel influence
    double* cache;                                                  //  The "inner rectangle" we compute first
    unsigned int ctr;

    #ifdef __NEURON_DEBUG
    printf("run_Upres(%d, %d)\n", layer->inputW, layer->inputH);
    #endif

    for(ctr = 0; ctr < layer->outlen; ctr++)                        //  Blank out the output buffer
      layer->out[ctr] = 0.0;

    for(i = 0; i < layer->n; i++)                                   //  For each up-ressing, write the inner rectangle to cache and then wreath with padding.
      {
                                                                    //  Compute the shape of the inner rectangle
        cache_w = layer->inputW * (layer->params[i].stride_h + 1) - layer->params[i].stride_h;
        cache_h = layer->inputH * (layer->params[i].stride_v + 1) - layer->params[i].stride_v;

        output_w = cache_w + 2 * layer->params[i].padding_h;        //  Compute the shape of the padded rectangle
        output_h = cache_h + 2 * layer->params[i].padding_v;
                                                                    //  Allocate cache for the inner rectangle
        if((cache = (double*)malloc(cache_w * cache_h * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresLayer cache array\n");
            exit(1);
          }

        ctr = 0;                                                    //  Reset counter: this now acts as our temporary output iterator

        if(layer->params[i].sMethod == FILL_INTERP)                 //  Fill strides using bilinear interpolation
          {
            sc_inv_h = (double)layer->inputW / (double)(cache_w);
            sc_inv_v = (double)layer->inputH / (double)(cache_h);

            for(y = 0; y < cache_h; y++)
              {
                for(x = 0; x < cache_w; x++)
                  {
                    x_prime = (double)x * sc_inv_h;                 //  Where in the source does this pixel fall?
                    y_prime = (double)y * sc_inv_v;

                    a = x_prime - (double)((unsigned int)x_prime);  //  Clip the fractional parts, store them in a and b:
                    b = y_prime - (double)((unsigned int)y_prime);  //  weigh the influences of neighboring pixels.

                    cache[ctr] = ((1.0 - a) * (1.0 - b)) * xvec[ (unsigned int)y_prime      * layer->inputW + (unsigned int)x_prime    ] +
                                 ((1.0 - a) * b)         * xvec[((unsigned int)y_prime + 1) * layer->inputW + (unsigned int)x_prime    ] +
                                 (a * (1.0 - b))         * xvec[ (unsigned int)y_prime      * layer->inputW + (unsigned int)x_prime + 1] +
                                 (a * b)                 * xvec[((unsigned int)y_prime + 1) * layer->inputW + (unsigned int)x_prime + 1];

                    ctr++;
                  }
              }
          }
        else if(layer->params[i].sMethod == FILL_SAME)              //  Fill strides in by duplicating the nearest source element
          {
            sc_inv_h = (double)layer->inputW / (double)(cache_w);
            sc_inv_v = (double)layer->inputH / (double)(cache_h);

            for(y = 0; y < cache_h; y++)
              {
                for(x = 0; x < cache_w; x++)
                  {
                    x_prime = (double)x * sc_inv_h;                 //  Where in the source does this pixel fall?
                    y_prime = (double)y * sc_inv_v;

                    a = x_prime - (double)((unsigned int)x_prime);  //  Clip the fractional parts, store them in a and b:
                    b = y_prime - (double)((unsigned int)y_prime);  //  weigh the influences of neighboring pixels.

                    val = ((1.0 - a) * (1.0 - b));                  //  Initial assumption: this pixel is nearest
                    cache[ctr]     = xvec[ (unsigned int)y_prime      * layer->inputW + (unsigned int)x_prime    ];

                    if(((1.0 - a) * b) > val)                       //  Does this pixel have greater influence?
                      {
                        val = ((1.0 - a) * b);
                        cache[ctr] = xvec[((unsigned int)y_prime + 1) * layer->inputW + (unsigned int)x_prime    ];
                      }
                    if((a * (1.0 - b)) > val)                       //  Does this pixel have greater influence?
                      {
                        val = (a * (1.0 - b));
                        cache[ctr] = xvec[ (unsigned int)y_prime      * layer->inputW + (unsigned int)x_prime + 1];
                      }
                    if((a * b) > val)                               //  Does this pixel have greater influence?
                      {                                             //  (No point storing 'val' anymore.)
                        cache[ctr] = xvec[((unsigned int)y_prime + 1) * layer->inputW + (unsigned int)x_prime + 1];
                      }

                    ctr++;
                  }
              }
          }
        else                                                        //  Fill strides in with zeroes
          {
            x_src = 0;                                              //  Initialize source-iterators
            y_src = 0;

            for(y = 0; y < cache_h; y += (layer->params[i].stride_v + 1))
              {
                for(x = 0; x < cache_w; x += (layer->params[i].stride_h + 1))
                  {
                                                                    //  Copy source pixel
                    cache[ctr] = xvec[y_src * layer->inputW + x_src];
                    x_src++;                                        //  Increment source x-iterator
                    ctr += (layer->params[i].stride_h + 1);         //  Advance output-iterator by horizontal stride
                  }
                x_src = 0;                                          //  Reset source x-iterator
                y_src++;                                            //  Increment source y-iterator
                ctr += (layer->params[i].stride_v + 1) * cache_w;   //  Advance output-iterator by vertical stride
              }
          }

        if(layer->params[i].pMethod != FILL_ZERO)                   //  Duplicate extrema
          {
                                                                    //  First fill in the sides
            for(y = layer->params[i].stride_v; y <= output_h - layer->params[i].stride_v; y++)
              {
                for(x = 0; x < layer->params[i].stride_h; x++)      //  Duplicate left side
                  layer->out[o + output_w * y + x] = cache[(y - layer->params[i].stride_v) * cache_w];
                                                                    //  Duplicate right side
                for(x = layer->params[i].stride_h + cache_w; x < output_w; x++)
                  layer->out[o + output_w * y + layer->params[i].stride_h + cache_w + x] = cache[(y - layer->params[i].stride_v) * cache_w + cache_w - 1];
              }
                                                                    //  Then fill the top and bottom
            for(y = 0; y < layer->params[i].stride_v; y++)          //  Fill top by referring to the first side-padded row
              {
                for(x = 0; x < output_w; x++)
                  layer->out[o + y * output_w + x] = layer->out[o + layer->params[i].stride_v * output_w + x];
              }
                                                                    //  Fill bottom by referring to the last side-padded row
            for(y = layer->params[i].stride_v + cache_h + 1; y < output_h; y++)
              {
                for(x = 0; x < output_w; x++)
                  layer->out[o + y * output_w + x] = layer->out[o + (layer->params[i].stride_v + cache_h) * output_w + x];
              }
          }
                                                                    //  Now, whether we had fancy padding or not, set cache into output buffer
        x_src = 0;                                                  //  Reset; these now iterate over the cached inner rectangle
        y_src = 0;
        for(y = 0; y < output_h; y++)                               //  For every row in the padded output for the current up-ressing
          {                                                         //  if we have passed the topmost padding and not yet reached the bottommost
            if(y >= layer->params[i].stride_v && y < output_h - layer->params[i].stride_v)
              {
                for(x = 0; x < output_w; x++)                       //  For every column in the padded output for the current up-ressing
                  {                                                 //  if we have passed the leftmost padding and not yet reached the rightmost
                    if(x >= layer->params[i].stride_h && x < output_w - layer->params[i].stride_h)
                      {
                                                                    //  Copy from cache
                        layer->out[o] = cache[y_src * cache_w + x_src];
                        x_src++;                                    //  Increment cache's x-iterator
                      }
                    o++;                                            //  Increment output buffer iterator
                  }
                x_src = 0;                                          //  Reset cache's x-iterator
                y_src++;                                            //  Increment cache's y-iterator
              }
            else                                                    //  Otherwise, skip a whole output row
              o += output_w;
          }

        free(cache);                                                //  Release
      }

    return layer->outlen;
  }

#endif
