#ifndef __ACCUM_H
#define __ACCUM_H

/**************************************************************************************************
 Neural Network library, by Eric C. Joyce

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

typedef struct AccumLayerType
  {
    unsigned int i;                                                 //  Number of inputs--ACCUMULATORS GET NO bias-1
    char name[LAYER_NAME_LEN];
    double* out;
  } AccumLayer;

/**************************************************************************************************
 Prototypes  */

void setName_Accum(char* n, AccumLayer*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 Accumulator-Layers  */

/* Set the name of the given Accumulator Layer */
void setName_Accum(char* n, AccumLayer* layer)
  {
    unsigned char i;
    unsigned char lim;
    lim = (strlen(n) < LAYER_NAME_LEN) ? strlen(n) : LAYER_NAME_LEN;
    for(i = 0; i < lim; i++)
      layer->name[i] = n[i];
    layer->name[i] = '\0';
    return;
  }

#endif
