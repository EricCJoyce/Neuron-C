#ifndef __NORMAL_H
#define __NORMAL_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

 A normalizing layer applies the four learned parameters to its input.
  m = learned mean
  s = learned standard deviation
  g = learned coefficient
  b = learned constant

 input vec{x}    output vec{y}
   [ x1 ]     [ g*((x1 - m)/s)+b ]
   [ x2 ]     [ g*((x2 - m)/s)+b ]
   [ x3 ]     [ g*((x3 - m)/s)+b ]
   [ x4 ]     [ g*((x4 - m)/s)+b ]
   [ x5 ]     [ g*((x5 - m)/s)+b ]

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

#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */

/*
#define __NEURON_DEBUG 1
*/

typedef struct NormalLayerType
  {
    unsigned int i;                                                 //  Number of inputs
    double m;                                                       //  Mu: the mean learned during training
    double s;                                                       //  Sigma: the standard deviation learned during training
    double g;                                                       //  The factor learned during training
    double b;                                                       //  The constant learned during training

    double* out;
    char name[LAYER_NAME_LEN];
  } NormalLayer;

/**************************************************************************************************
 Prototypes  */

void setM_Normal(double, NormalLayer*);
void setS_Normal(double, NormalLayer*);
void setG_Normal(double, NormalLayer*);
void setB_Normal(double, NormalLayer*);
void setName_Normal(char* n, NormalLayer*);
void print_Normal(NormalLayer*);
unsigned int run_Normal(double*, NormalLayer*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 Normalization-Layers  */

void setM_Normal(double m, NormalLayer* layer)
  {
    layer->m = m;
    return;
  }

void setS_Normal(double s, NormalLayer* layer)
  {
    layer->s = s;
    return;
  }

void setG_Normal(double g, NormalLayer* layer)
  {
    layer->g = g;
    return;
  }

void setB_Normal(double b, NormalLayer* layer)
  {
    layer->b = b;
    return;
  }

void setName_Normal(char* n, NormalLayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

void print_Normal(NormalLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("print_Normal()\n");
    #endif

    printf("Input Length = %d\n", layer->i);
    printf("Mean = %f\n", layer->m);
    printf("Std.dev = %f\n", layer->s);
    printf("Coefficient = %f\n", layer->g);
    printf("Constant = %f\n", layer->b);
    printf("\n");

    return;
  }

unsigned int run_Normal(double* xvec, NormalLayer* layer)
  {
    unsigned int j;
    for(j = 0; j < layer->i; j++)
      layer->out[j] = layer->g * ((xvec[j] - layer->m) / layer->s) + layer->b;
    return layer->i;
  }

#endif
