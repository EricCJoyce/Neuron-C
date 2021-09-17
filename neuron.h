#ifndef __NEURON_H
#define __NEURON_H

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

#include "accum.h"                                                  /* Include Accumulator Layer library */
#include "conv2d.h"                                                 /* Include 2D-Convolutional Layer library */
#include "dense.h"                                                  /* Include Dense Layer library */
#include "gru.h"                                                    /* Include GRU Layer library */
#include "lstm.h"                                                   /* Include LSTM Layer library */
#include "normalization.h"                                          /* Include Normalization Layer library */
#include "pooling.h"                                                /* Include Pooling Layer library */
#include "upres.h"                                                  /* Include Up-Res (a.k.a. Transpose Convolution) Layer library */

#define INPUT_ARRAY   0                                             /* Flag refers to network input */
#define DENSE_ARRAY   1                                             /* Flag refers to 'denselayers' */
#define CONV2D_ARRAY  2                                             /* Flag refers to 'convlayers' */
#define ACCUM_ARRAY   3                                             /* Flag refers to 'accumlayers' */
#define LSTM_ARRAY    4                                             /* Flag refers to 'lstmlayers' */
#define GRU_ARRAY     5                                             /* Flag refers to 'grulayers' */
#define POOL_ARRAY    6                                             /* Flag refers to 'poollayers' */
#define UPRES_ARRAY   7                                             /* Flag refers to 'upreslayers' */
#define NORMAL_ARRAY  8                                             /* Flag refers to 'normallayers' */

#define VARSTR_LEN      16                                          /* Length of a Variable key string */
#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */
#define COMMSTR_LEN     64                                          /* Length of a Network Comment string */

/*
#define __NEURON_DEBUG 1
*/

typedef struct VariableType
  {
    char key[VARSTR_LEN];                                           //  String for variable key/symbol
    double value;                                                   //  Variable's value
  } Variable;

typedef struct NodeType                                             //  Really just used in connectivity tests
  {
    unsigned char type;                                             //  Which network array to look in
    unsigned int index;                                             //  Index into that array
  } Node;

typedef struct EdgeType
  {
    unsigned char srcType;                                          //  Indicates in which array to find the source
    unsigned int srcIndex;                                          //  Index into that array

    unsigned int selectorStart;                                     //  From (and including) this array element...
    unsigned int selectorEnd;                                       //  ...to (but excluding) this array element.

    unsigned char dstType;                                          //  Indicates in which array to find the destination
    unsigned int dstIndex;                                          //  Index into that array
  } Edge;

typedef struct NeuralNetType
  {
    unsigned int i;                                                 //  Number of Network inputs

    Edge* edgelist;                                                 //  Edge list
    unsigned int len;                                               //  Length of edge list

    DenseLayer* denselayers;                                        //  Array of Dense Layers
    unsigned int denseLen;                                          //  Length of that array

    Conv2DLayer* convlayers;                                        //  Array of Conv2D Layers
    unsigned int convLen;                                           //  Length of that array

    AccumLayer* accumlayers;                                        //  Array of Accum Layers
    unsigned int accumLen;                                          //  Length of that array

    LSTMLayer* lstmlayers;                                          //  Array of LSTM Layers
    unsigned int lstmLen;                                           //  Length of that array

    GRULayer* grulayers;                                            //  Array of GRU Layers
    unsigned int gruLen;                                            //  Length of that array

    Pool2DLayer* poollayers;                                        //  Array of Pooling Layers
    unsigned int poolLen;                                           //  Length of that array

    UpresLayer* upreslayers;                                        //  Array of Upres Layers
    unsigned int upresLen;                                          //  Length of that array

    NormalLayer* normlayers;                                        //  Array of Normal Layers
    unsigned int normalLen;                                         //  Length of that array

    Variable* variables;                                            //  Array of Network Variables
    unsigned char vars;                                             //  Length of that array

    unsigned int gen;                                               //  Network generation/epoch
    double fit;                                                     //  Network fitness
    char comment[COMMSTR_LEN];                                      //  Network comment
  } NeuralNet;

/**************************************************************************************************
 Prototypes  */

bool init_NN(NeuralNet**, unsigned int);
void free_NN(NeuralNet*);
unsigned int run_NN(double*, NeuralNet*, double**);
bool linkLayers(unsigned char, unsigned int, unsigned int, unsigned int, unsigned char, unsigned int, NeuralNet*);
bool load_NN(char*, NeuralNet*);
bool write_NN(char*, NeuralNet*);
void sortEdges(NeuralNet*);
unsigned int nameIndex(char*, NeuralNet*);
unsigned char nameType(char*, NeuralNet*);
void printEdgeList(NeuralNet*);
void print_NN(NeuralNet*);
void printLayerName(unsigned char, unsigned int, NeuralNet*);

unsigned int add_Dense(unsigned int, unsigned int, NeuralNet*);
bool read_Dense(DenseLayer*, unsigned int, FILE*);
bool write_Dense(DenseLayer*, unsigned int, FILE*);

unsigned int add_Conv2D(unsigned int, unsigned int, NeuralNet*);
bool read_Conv2D(Conv2DLayer*, unsigned int, FILE*);
bool write_Conv2D(Conv2DLayer*, unsigned int, FILE*);

unsigned int add_Accum(unsigned int, NeuralNet*);
bool read_Accum(AccumLayer*, unsigned int, FILE*);
bool write_Accum(AccumLayer*, unsigned int, FILE*);

unsigned int add_LSTM(unsigned int, unsigned int, unsigned int, NeuralNet*);
bool read_LSTM(LSTMLayer*, unsigned int, FILE*);
bool write_LSTM(LSTMLayer*, unsigned int, FILE*);

unsigned int add_GRU(unsigned int, unsigned int, unsigned int, NeuralNet*);
bool read_GRU(GRULayer*, unsigned int, FILE*);
bool write_GRU(GRULayer*, unsigned int, FILE*);

unsigned int add_Pool(unsigned int, unsigned int, NeuralNet*);
bool read_Pool2D(Pool2DLayer*, unsigned int, FILE*);
bool write_Pool2D(Pool2DLayer*, unsigned int, FILE*);

unsigned int add_Upres(unsigned int, unsigned int, NeuralNet*);
bool read_Upres(UpresLayer*, unsigned int, FILE*);
bool write_Upres(UpresLayer*, unsigned int, FILE*);

unsigned int add_Normal(unsigned int, NeuralNet*);
bool read_Normal(NormalLayer*, unsigned int, FILE*);
bool write_Normal(NormalLayer*, unsigned int, FILE*);

/**************************************************************************************************
 Globals  */

/**************************************************************************************************
 Neural Network  */

/* Initialize a deep network for ad hoc construction. */
bool init_NN(NeuralNet** nn, unsigned int inputs)
  {
    unsigned char i;

    #ifdef __NEURON_DEBUG
    printf("init_NN(%d)\n", inputs);
    #endif

    if(inputs == 0)
      return false;

    if(((*nn) = (NeuralNet*)malloc(sizeof(NeuralNet))) == NULL)
      return false;

    (*nn)->i = inputs;                                              //  Save number of inputs
    (*nn)->len = 0;                                                 //  Initially zero edges
    (*nn)->denseLen = 0;                                            //  Initially zero DenseLayers
    (*nn)->convLen = 0;                                             //  Initially zero Conv2DLayers
    (*nn)->accumLen = 0;                                            //  Initially zero AccumLayers
    (*nn)->lstmLen = 0;                                             //  Initially zero LSTMLayers
    (*nn)->gruLen = 0;                                              //  Initially zero GRULayers
    (*nn)->poolLen = 0;                                             //  Initially zero Pool2DLayers
    (*nn)->upresLen = 0;                                            //  Initially zero UpresLayers
    (*nn)->normalLen = 0;                                           //  Initially zero NormalLayers
    (*nn)->vars = 0;                                                //  Initially zero variables
    (*nn)->gen = 0;                                                 //  Initialize generation to zero
    (*nn)->fit = 0.0;                                               //  Initialize fitness to zero
    for(i = 0; i < COMMSTR_LEN; i++)                                //  Fill comment buffer with NULLs
      (*nn)->comment[i] = '\0';

    return true;
  }

/* Release network, all arrays and structures. */
void free_NN(NeuralNet* nn)
  {
    unsigned int i, j;

    if(nn != NULL)
      {
        if(nn->len > 0)
          free(nn->edgelist);

        if(nn->denseLen > 0)                                        //  Clean up Dense Layers array
          {
            for(i = 0; i < nn->denseLen; i++)
              {
                free(nn->denselayers[i].W);                         //  Free the weights matrix
                free(nn->denselayers[i].M);                         //  Free the mask matrix
                free(nn->denselayers[i].f);                         //  Free activation function flags array
                free(nn->denselayers[i].alpha);                     //  Free activation parameters array
                free(nn->denselayers[i].out);                       //  Free output buffer
              }
            free(nn->denselayers);                                  //  Release the layers array
          }

        if(nn->convLen > 0)                                         //  Clean up 2D-Convolutional Layers array
          {
            for(i = 0; i < nn->convLen; i++)                        //  For each Conv2DLayer...
              {
                if(nn->convlayers[i].n > 0)
                  {
                    for(j = 0; j < nn->convlayers[i].n; j++)        //  For each Filter2D...
                      free(nn->convlayers[i].filters[j].W);         //  Free filter weights
                    if(nn->convlayers[i].outlen > 0)
                      free(nn->convlayers[i].out);                  //  Free output buffer
                    free(nn->convlayers[i].filters);                //  Free Filter2D array
                  }
              }
            free(nn->convlayers);                                   //  Release the layers array
          }

        if(nn->accumLen > 0)                                        //  Clean up Accumulator Layers array
          {
            for(i = 0; i < nn->accumLen; i++)
              free(nn->accumlayers[i].out);                         //  Free output buffer
            free(nn->accumlayers);                                  //  Release the layers array
          }

        if(nn->lstmLen > 0)                                         //  Clean up LSTM Layers array
          {
            for(i = 0; i < nn->lstmLen; i++)                        //  For each layer
              {
                free(nn->lstmlayers[i].Wi);                         //  Free the Wi matrix
                free(nn->lstmlayers[i].Wo);                         //  Free the Wo matrix
                free(nn->lstmlayers[i].Wf);                         //  Free the Wf matrix
                free(nn->lstmlayers[i].Wc);                         //  Free the Wc matrix

                free(nn->lstmlayers[i].Ui);                         //  Free the Ui matrix
                free(nn->lstmlayers[i].Uo);                         //  Free the Uo matrix
                free(nn->lstmlayers[i].Uf);                         //  Free the Uf matrix
                free(nn->lstmlayers[i].Uc);                         //  Free the Uc matrix

                free(nn->lstmlayers[i].bi);                         //  Free the bi vector
                free(nn->lstmlayers[i].bo);                         //  Free the bo vector
                free(nn->lstmlayers[i].bf);                         //  Free the bf vector
                free(nn->lstmlayers[i].bc);                         //  Free the bc vector

                free(nn->lstmlayers[i].c);                          //  Free the c vector
                free(nn->lstmlayers[i].H);                          //  Free the H cache
              }
            free(nn->lstmlayers);                                   //  Release the layers array
          }

        if(nn->gruLen > 0)                                          //  Clean up GRU Layers array
          {
            for(i = 0; i < nn->gruLen; i++)                         //  For each layer
              {
                free(nn->grulayers[i].Wz);                          //  Free the Wz matrix
                free(nn->grulayers[i].Wr);                          //  Free the Wr matrix
                free(nn->grulayers[i].Wh);                          //  Free the Wh matrix

                free(nn->grulayers[i].Uz);                          //  Free the Uz matrix
                free(nn->grulayers[i].Ur);                          //  Free the Ur matrix
                free(nn->grulayers[i].Uh);                          //  Free the Uh matrix

                free(nn->grulayers[i].bz);                          //  Free the bz vector
                free(nn->grulayers[i].br);                          //  Free the br vector
                free(nn->grulayers[i].bh);                          //  Free the bh vector

                free(nn->grulayers[i].H);                           //  Free the H cache
              }
            free(nn->grulayers);                                    //  Release the layers array
          }

        if(nn->poolLen > 0)                                         //  Clean up Pooling Layers array
          {
            for(i = 0; i < nn->poolLen; i++)                        //  For each layer
              {
                if(nn->poollayers[i].n > 0)
                  {
                    free(nn->poollayers[i].pools);                  //  Free Pool2D array
                    free(nn->poollayers[i].out);                    //  Free the output array
                  }
              }
            free(nn->poollayers);                                   //  Release the layers array
          }

        if(nn->upresLen > 0)                                        //  Clean up Upres Layers array
          {
            for(i = 0; i < nn->upresLen; i++)                       //  For each layer
              {
                if(nn->upreslayers[i].n > 0)
                  {
                    free(nn->upreslayers[i].params);                //  Free the Upres Parameters array
                    free(nn->upreslayers[i].out);                   //  Free the output buffer
                  }
              }
            free(nn->upreslayers);                                  //  Release the layers array
          }

        if(nn->normalLen > 0)                                       //  Clean up Normalization Layers array
          {
            for(i = 0; i < nn->normalLen; i++)
              free(nn->normlayers[i].out);                          //  Free the output buffer
            free(nn->normlayers);                                   //  Release the layers array
          }

        if(nn->vars > 0)                                            //  Clean up Variables
          free(nn->variables);

        free(nn);                                                   //  Finally, release the Neural Network pointer
      }

    return;
  }

/* Input vector 'x' has length = 'nn'->'i'.
   Output vector 'z' will have length = # of units/outputs in last layer */
unsigned int run_NN(double* x, NeuralNet* nn, double** z)
  {
    double* in;
    unsigned int inLen = 0;
    unsigned int outLen = 0;
    unsigned int i, j, k, l;
    unsigned int last = 0;
    unsigned int t;                                                 //  For reading from LSTMs

    #ifdef __NEURON_DEBUG
    printf("run_NN()\n");
    #endif

    i = 0;
    while(i < nn->len)                                              //  For each edge in nn->edgelist
      {
                                                                    //  Set the length of the input vector to the input
                                                                    //  size of the current destination layer.
        switch(nn->edgelist[i].dstType)                             //  Which array contains the destination layer?
          {
            case DENSE_ARRAY:   inLen = nn->denselayers[nn->edgelist[i].dstIndex].i;
                                break;
            case CONV2D_ARRAY:  inLen = nn->convlayers[nn->edgelist[i].dstIndex].inputW *
                                        nn->convlayers[nn->edgelist[i].dstIndex].inputH;
                                break;
            case ACCUM_ARRAY:   inLen = nn->accumlayers[nn->edgelist[i].dstIndex].i;
                                break;
            case LSTM_ARRAY:    inLen = nn->lstmlayers[nn->edgelist[i].dstIndex].d;
                                break;
            case GRU_ARRAY:     inLen = nn->grulayers[nn->edgelist[i].dstIndex].d;
                                break;
            case POOL_ARRAY:    inLen = nn->poollayers[nn->edgelist[i].dstIndex].inputW *
                                        nn->poollayers[nn->edgelist[i].dstIndex].inputH;
                                break;
            case UPRES_ARRAY:   inLen = nn->upreslayers[nn->edgelist[i].dstIndex].inputW *
                                        nn->upreslayers[nn->edgelist[i].dstIndex].inputH;
                                break;
            case NORMAL_ARRAY:  inLen = nn->normlayers[nn->edgelist[i].dstIndex].i;
                                break;
          }

        if((in = (double*)malloc(inLen * sizeof(double))) == NULL)  //  Allocate the vector
          {
            printf("ERROR: Unable to allocate input vector for layer(%d, %d)\n", nn->edgelist[i].dstType, nn->edgelist[i].dstIndex);
            exit(1);
          }

        k = 0;                                                      //  Point to head of input buffer
        j = i;                                                      //  Advance j to encompass all inputs to current layer
        while(j < nn->len && nn->edgelist[i].dstType == nn->edgelist[j].dstType &&
                             nn->edgelist[i].dstIndex == nn->edgelist[j].dstIndex)
          {
            switch(nn->edgelist[j].srcType)
              {
                case INPUT_ARRAY:                                   //  Receiving from network input
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = x[l];
                                           k++;
                                         }
                                       break;
                case DENSE_ARRAY:                                   //  Receiving from a dense layer
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = nn->denselayers[nn->edgelist[j].srcIndex].out[l];
                                           k++;
                                         }
                                       break;
                case CONV2D_ARRAY:                                  //  Receiving from a convolutional layer
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = nn->convlayers[nn->edgelist[j].srcIndex].out[l];
                                           k++;
                                         }
                                       break;
                case ACCUM_ARRAY:                                   //  Receiving from an accumulator layer
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = nn->accumlayers[nn->edgelist[j].srcIndex].out[l];
                                           k++;
                                         }
                                       break;
                case LSTM_ARRAY:                                    //  Receiving from an LSTM layer
                                       if(nn->lstmlayers[nn->edgelist[j].srcIndex].t >= nn->lstmlayers[nn->edgelist[j].srcIndex].cache)
                                         t = nn->lstmlayers[nn->edgelist[j].srcIndex].cache - 1;
                                       else
                                         t = nn->lstmlayers[nn->edgelist[j].srcIndex].t - 1;
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                                                    //  Read from the LAST time step
                                           in[k] = nn->lstmlayers[nn->edgelist[j].srcIndex].H[ l * nn->lstmlayers[nn->edgelist[j].srcIndex].cache + t ];
                                           k++;
                                         }
                                       break;
                case GRU_ARRAY:                                     //  Receiving from a GRU layer
                                       if(nn->grulayers[nn->edgelist[j].srcIndex].t >= nn->grulayers[nn->edgelist[j].srcIndex].cache)
                                         t = nn->grulayers[nn->edgelist[j].srcIndex].cache - 1;
                                       else
                                         t = nn->grulayers[nn->edgelist[j].srcIndex].t - 1;
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                                                    //  Read from the LAST time step
                                           in[k] = nn->grulayers[nn->edgelist[j].srcIndex].H[ l * nn->grulayers[nn->edgelist[j].srcIndex].cache + t ];
                                           k++;
                                         }
                                       break;
                case POOL_ARRAY:                                    //  Receiving from a pooling layer
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = nn->poollayers[nn->edgelist[j].srcIndex].out[l];
                                           k++;
                                         }
                                       break;
                case UPRES_ARRAY:                                   //  Receiving from an upres layer
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = nn->upreslayers[nn->edgelist[j].srcIndex].out[l];
                                           k++;
                                         }
                                       break;
                case NORMAL_ARRAY:                                  //  Receiving from a normalization layer
                                       for(l = nn->edgelist[j].selectorStart; l < nn->edgelist[j].selectorEnd; l++)
                                         {
                                           in[k] = nn->normlayers[nn->edgelist[j].srcIndex].out[l];
                                           k++;
                                         }
                                       break;
              }
            j++;
          }

        switch(nn->edgelist[i].dstType)                             //  Which array contains the destination layer?
          {
            case DENSE_ARRAY:   outLen = run_Dense(in, nn->denselayers + nn->edgelist[i].dstIndex);
                                break;
            case CONV2D_ARRAY:  outLen = run_Conv2D(in, nn->convlayers + nn->edgelist[i].dstIndex);
                                break;
            case ACCUM_ARRAY:   outLen = inLen;
                                #ifdef __NEURON_DEBUG
                                printf("run_Accum()\n");
                                #endif
                                for(k = 0; k < inLen; k++)
                                  nn->accumlayers[nn->edgelist[i].dstIndex].out[k] = in[k];
                                break;
            case LSTM_ARRAY:    outLen = run_LSTM(in, nn->lstmlayers + nn->edgelist[i].dstIndex);
                                break;
            case GRU_ARRAY:     outLen = run_GRU(in, nn->grulayers + nn->edgelist[i].dstIndex);
                                break;
            case POOL_ARRAY:    outLen = run_Pool2D(in, nn->poollayers + nn->edgelist[i].dstIndex);
                                break;
            case UPRES_ARRAY:   outLen = run_Upres(in, nn->upreslayers + nn->edgelist[i].dstIndex);
                                break;
            case NORMAL_ARRAY:  outLen = run_Normal(in, nn->normlayers + nn->edgelist[i].dstIndex);
                                break;
          }

        free(in);                                                   //  Release input vector

        last = i;                                                   //  Save the index of the previous edge
        i = j;                                                      //  Increment 'i'
      }

    if(((*z) = (double*)malloc(outLen * sizeof(double))) == NULL)   //  Copy from last (internal) out to 'z'
      {
        printf("ERROR: Unable to allocate network output array\n");
        exit(1);
      }
    switch(nn->edgelist[last].dstType)
      {
        case DENSE_ARRAY:   for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->denselayers[nn->edgelist[last].dstIndex].out[i];
                            break;
        case CONV2D_ARRAY:  for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->convlayers[nn->edgelist[last].dstIndex].out[i];
                            break;
        case ACCUM_ARRAY:   for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->accumlayers[nn->edgelist[last].dstIndex].out[i];
                            break;
        case LSTM_ARRAY:    if(nn->lstmlayers[nn->edgelist[last].dstIndex].t >= nn->lstmlayers[nn->edgelist[last].dstIndex].cache)
                              t = nn->lstmlayers[nn->edgelist[last].dstIndex].cache - 1;
                            else
                              t = nn->lstmlayers[nn->edgelist[last].dstIndex].t - 1;
                            for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->lstmlayers[nn->edgelist[last].dstIndex].H[t * nn->lstmlayers[nn->edgelist[last].dstIndex].h + i];
                            break;
        case GRU_ARRAY:     if(nn->grulayers[nn->edgelist[last].dstIndex].t >= nn->grulayers[nn->edgelist[last].dstIndex].cache)
                              t = nn->grulayers[nn->edgelist[last].dstIndex].cache - 1;
                            else
                              t = nn->grulayers[nn->edgelist[last].dstIndex].t - 1;
                            for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->grulayers[nn->edgelist[last].dstIndex].H[t * nn->grulayers[nn->edgelist[last].dstIndex].h + i];
                            break;
        case POOL_ARRAY:    for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->poollayers[nn->edgelist[last].dstIndex].out[i];
                            break;
        case UPRES_ARRAY:   for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->upreslayers[nn->edgelist[last].dstIndex].out[i];
                            break;
        case NORMAL_ARRAY:  for(i = 0; i < outLen; i++)
                              (*z)[i] = nn->normlayers[nn->edgelist[last].dstIndex].out[i];
                            break;
      }

    return outLen;
  }

/* Connect layer srcIndex to layer dstIndex.
   We have to identify the types of src and dst so that we can identify which arrays the layer structs are in.
   We specify the slice from src as [selectorStart, selectorEnd].
   e.g.  linkLayers(INPUT_ARRAY, 0, 0, 63, CONV2D_ARRAY, 0)         //  From input[0:63] to convolution layer
         linkLayers(CONV2D_ARRAY, 0, 0, 91, ACCUM_ARRAY, 0)         //  From convolution layer to accumulator
         linkLayers(INPUT_ARRAY, 0, 63, 64, ACCUM_ARRAY, 0)         //  From input[63:64] to accumulator
         linkLayers(ACCUM_ARRAY, 0, 0, 92, DENSE_ARRAY, 0)          //  From accumulator to dense layer
         linkLayers(DENSE_ARRAY, 0, 0, 40, DENSE_ARRAY, 1)          //  From dense layer to dense layer
         linkLayers(DENSE_ARRAY, 1, 0, 10, DENSE_ARRAY, 2)          //  From dense layer to dense layer */
bool linkLayers(unsigned char srcFlag, unsigned int src,
                unsigned int selectorStart, unsigned int selectorEnd,
                unsigned char dstFlag, unsigned int dst,
                NeuralNet* nn)
  {
    unsigned int i, j;
    Node* queue;
    Node* tmp;
    unsigned int len = 0;
    Node* node;
    Node* visited = NULL;
    unsigned int vlen = 0;

    #ifdef __NEURON_DEBUG
    printf("linkLayers(");
    switch(srcFlag)
      {
        case INPUT_ARRAY:   printf("Input, ");  break;
        case DENSE_ARRAY:   printf("Dense, ");  break;
        case CONV2D_ARRAY:  printf("Conv2D, "); break;
        case ACCUM_ARRAY:   printf("Accum, ");  break;
        case LSTM_ARRAY:    printf("LSTM, ");  break;
        case GRU_ARRAY:     printf("GRU, ");  break;
        case POOL_ARRAY:    printf("Pool, ");  break;
        case UPRES_ARRAY:   printf("Upres, ");  break;
        case NORMAL_ARRAY:  printf("Normal, ");  break;
      }
    printf("%d, %d, %d, ", src, selectorStart, selectorEnd);
    switch(dstFlag)
      {
        case DENSE_ARRAY:   printf("Dense, ");  break;
        case CONV2D_ARRAY:  printf("Conv2D, "); break;
        case ACCUM_ARRAY:   printf("Accum, ");  break;
        case LSTM_ARRAY:    printf("LSTM, ");  break;
        case GRU_ARRAY:     printf("GRU, ");  break;
        case POOL_ARRAY:    printf("Pool, ");  break;
        case UPRES_ARRAY:   printf("Upres, ");  break;
        case NORMAL_ARRAY:  printf("Normal, ");  break;
      }
    printf("%d)\n", dst);
    #endif

    if(srcFlag == DENSE_ARRAY && src >= nn->denseLen)               //  Is the request out of bounds?
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for dense layers.\n");
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && src >= nn->convLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for convolutional layers.\n");
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && src >= nn->accumLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for accumulations layers.\n");
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && src >= nn->lstmLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for LSTM layers.\n");
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && src >= nn->gruLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for GRU layers.\n");
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && src >= nn->poolLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for pooling layers.\n");
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && src >= nn->upresLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for upres layers.\n");
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && src >= nn->normalLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given source is out of bounds for normalization layers.\n");
        #endif
        return false;
      }

    if(dstFlag == DENSE_ARRAY && dst >= nn->denseLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for dense layers.\n");
        #endif
        return false;
      }
    if(dstFlag == CONV2D_ARRAY && dst >= nn->convLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for convolutional layers.\n");
        #endif
        return false;
      }
    if(dstFlag == ACCUM_ARRAY && dst >= nn->accumLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for accumulator layers.\n");
        #endif
        return false;
      }
    if(dstFlag == LSTM_ARRAY && dst >= nn->lstmLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for LSTM layers.\n");
        #endif
        return false;
      }
    if(dstFlag == GRU_ARRAY && dst >= nn->gruLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for GRU layers.\n");
        #endif
        return false;
      }
    if(dstFlag == POOL_ARRAY && dst >= nn->poolLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for pooling layers.\n");
        #endif
        return false;
      }
    if(dstFlag == UPRES_ARRAY && dst >= nn->upresLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for upres layers.\n");
        #endif
        return false;
      }
    if(dstFlag == NORMAL_ARRAY && dst >= nn->normalLen)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given destination is out of bounds for normalization layers.\n");
        #endif
        return false;
      }

    if(srcFlag == DENSE_ARRAY && selectorStart >= nn->denselayers[src].n)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for dense layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && selectorStart >= outputLen_Conv2D(nn->convlayers + src))
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for convolutional layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && selectorStart >= nn->accumlayers[src].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for accumulator layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && selectorStart >= nn->lstmlayers[src].h)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for LSTM layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && selectorStart >= nn->grulayers[src].h)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for GRU layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && selectorStart >= outputLen_Pool2D(nn->poollayers + src))
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for pooling layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && selectorStart >= outputLen_Upres(nn->upreslayers + src))
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for upres layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && selectorStart >= nn->normlayers[src].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is out of bounds for normalization layer %d.\n", src);
        #endif
        return false;
      }

    if(srcFlag == DENSE_ARRAY && selectorEnd > nn->denselayers[src].n)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for dense layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && selectorEnd > outputLen_Conv2D(nn->convlayers + src))
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for convolutional layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && selectorEnd > nn->accumlayers[src].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for accumulator layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && selectorEnd > nn->lstmlayers[src].h)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for LSTM layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && selectorEnd > nn->grulayers[src].h)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for GRU layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && selectorEnd > outputLen_Pool2D(nn->poollayers + src))
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for pooling layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && selectorEnd > outputLen_Upres(nn->upreslayers + src))
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for upres layer %d.\n", src);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && selectorEnd > nn->normlayers[src].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection end is out of bounds for normalization layer %d.\n", src);
        #endif
        return false;
      }

    if(selectorEnd < selectorStart)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because the given selection start is greater than the selection end.\n");
        #endif
        return false;
      }
                                                                    //  Check output-input shapes match
    if(srcFlag == DENSE_ARRAY && dstFlag == DENSE_ARRAY &&          //  Dense-->Dense
       outputLen_Dense(nn->denselayers + src) != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == CONV2D_ARRAY &&         //  Dense-->Conv2D
       outputLen_Dense(nn->denselayers + src) != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for convolutional layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == ACCUM_ARRAY &&          //  Dense-->Accumulator
       outputLen_Dense(nn->denselayers + src) > nn->accumlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == LSTM_ARRAY &&           //  Dense-->LSTM
       outputLen_Dense(nn->denselayers + src) != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == GRU_ARRAY &&            //  Dense-->GRU
       outputLen_Dense(nn->denselayers + src) != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == POOL_ARRAY &&           //  Dense-->Pool
       outputLen_Dense(nn->denselayers + src) != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for pooling layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == UPRES_ARRAY &&          //  Dense-->Upres
       outputLen_Dense(nn->denselayers + src) != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == NORMAL_ARRAY &&         //  Dense-->Normalization
       outputLen_Dense(nn->denselayers + src) != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    if(srcFlag == CONV2D_ARRAY && dstFlag == DENSE_ARRAY &&         //  Conv2D-->Dense
       outputLen_Conv2D(nn->convlayers + src) != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == CONV2D_ARRAY &&        //  Conv2D-->Conv2D
       outputLen_Conv2D(nn->convlayers + src) != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == ACCUM_ARRAY &&         //  Conv2D-->Accumulator
       outputLen_Conv2D(nn->convlayers + src) > nn->accumlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == LSTM_ARRAY &&          //  Conv2D-->LSTM
       outputLen_Conv2D(nn->convlayers + src) != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == GRU_ARRAY &&           //  Conv2D-->GRU
       outputLen_Conv2D(nn->convlayers + src) != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == POOL_ARRAY &&          //  Conv2D-->Pool
       outputLen_Conv2D(nn->convlayers + src) != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for pool layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == UPRES_ARRAY &&         //  Conv2D-->Upres
       outputLen_Conv2D(nn->convlayers + src) != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == NORMAL_ARRAY &&        //  Conv2D-->Normalization
       outputLen_Conv2D(nn->convlayers + src) != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    if(srcFlag == ACCUM_ARRAY && dstFlag == DENSE_ARRAY &&          //  Accumulator-->Dense
       nn->accumlayers[src].i != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from accumulator layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && dstFlag == CONV2D_ARRAY &&         //  Accumulator-->Conv2D
       nn->accumlayers[src].i != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from accumulator layer %d does not match input for convolutional layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && dstFlag == ACCUM_ARRAY &&          //  Accumulator-->Accumulator
       nn->accumlayers[src].i > nn->accumlayers[dst].i)             //  Incoming layer free to be < Accumulator size
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from accumulator layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && dstFlag == LSTM_ARRAY &&           //  Accumulator-->LSTM
       nn->accumlayers[src].i != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from accumulator layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && dstFlag == GRU_ARRAY &&            //  Accumulator-->GRU
       nn->accumlayers[src].i != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from accumulator layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && dstFlag == POOL_ARRAY &&           //  Accumulator-->Pool
       nn->accumlayers[src].i != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from accumulator layer %d does not match input for pool layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && dstFlag == UPRES_ARRAY &&          //  Accumulator-->Upres
       nn->accumlayers[src].i != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from accumulator layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == ACCUM_ARRAY && dstFlag == NORMAL_ARRAY &&         //  Accumulator-->Normalization
       nn->accumlayers[src].i != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected bcause output from accumulator layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    if(srcFlag == LSTM_ARRAY && dstFlag == DENSE_ARRAY &&           //  LSTM-->Dense
       nn->lstmlayers[src].h != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && dstFlag == CONV2D_ARRAY &&          //  LSTM-->Conv2D
       nn->lstmlayers[src].h != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d does not match input for convolutional layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && dstFlag == ACCUM_ARRAY &&           //  LSTM-->Accumulator
       nn->lstmlayers[src].h > nn->accumlayers[dst].i)              //  Incoming layer free to be < Accumulator size
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && dstFlag == LSTM_ARRAY &&            //  LSTM-->LSTM
       nn->lstmlayers[src].h != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && dstFlag == GRU_ARRAY &&             //  LSTM-->GRU
       nn->lstmlayers[src].h != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && dstFlag == POOL_ARRAY &&            //  LSTM-->Pool
       nn->lstmlayers[src].h != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d does not match input for pool layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && dstFlag == UPRES_ARRAY &&           //  LSTM-->Upres
       nn->lstmlayers[src].h != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == LSTM_ARRAY && dstFlag == NORMAL_ARRAY &&          //  LSTM-->Normalization
       nn->lstmlayers[src].h != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from LSTM layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    if(srcFlag == GRU_ARRAY && dstFlag == DENSE_ARRAY &&            //  GRU-->Dense
       nn->grulayers[src].h != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && dstFlag == CONV2D_ARRAY &&           //  GRU-->Conv2D
       nn->grulayers[src].h != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d does not match input for convolutional layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && dstFlag == ACCUM_ARRAY &&            //  GRU-->Accumulator
       nn->grulayers[src].h > nn->accumlayers[dst].i)               //  Incoming layer free to be < Accumulator size
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && dstFlag == LSTM_ARRAY &&             //  GRU-->LSTM
       nn->grulayers[src].h != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && dstFlag == GRU_ARRAY &&              //  GRU-->GRU
       nn->grulayers[src].h != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && dstFlag == POOL_ARRAY &&             //  GRU-->Pool
       nn->grulayers[src].h != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d does not match input for pool layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && dstFlag == UPRES_ARRAY &&            //  GRU-->Upres
       nn->grulayers[src].h != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == GRU_ARRAY && dstFlag == NORMAL_ARRAY &&           //  GRU-->Normalization
       nn->grulayers[src].h != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from GRU layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    if(srcFlag == POOL_ARRAY && dstFlag == DENSE_ARRAY &&           //  Pool-->Dense
       outputLen_Pool2D(nn->poollayers + src) != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && dstFlag == CONV2D_ARRAY &&          //  Pool-->Conv2D
       outputLen_Pool2D(nn->poollayers + src) != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d does not match input for convolutional layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && dstFlag == ACCUM_ARRAY &&           //  Pool-->Accumulator
                                                                    //  Incoming layer free to be < Accumulator size
       outputLen_Pool2D(nn->poollayers + src) > nn->accumlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && dstFlag == LSTM_ARRAY &&            //  Pool-->LSTM
       outputLen_Pool2D(nn->poollayers + src) != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && dstFlag == GRU_ARRAY &&             //  Pool-->GRU
       outputLen_Pool2D(nn->poollayers + src) != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && dstFlag == POOL_ARRAY &&            //  Pool-->Pool
       outputLen_Pool2D(nn->poollayers + src) != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d does not match input for pool layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && dstFlag == UPRES_ARRAY &&           //  Pool-->Upres
       outputLen_Pool2D(nn->poollayers + src) != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == POOL_ARRAY && dstFlag == NORMAL_ARRAY &&          //  Pool-->Normalization
       outputLen_Pool2D(nn->poollayers + src) != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from pool layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    if(srcFlag == UPRES_ARRAY && dstFlag == DENSE_ARRAY &&          //  Upres-->Dense
       outputLen_Upres(nn->upreslayers + src) != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && dstFlag == CONV2D_ARRAY &&         //  Upres-->Conv2D
       outputLen_Upres(nn->upreslayers + src) != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d does not match input for convolutional layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && dstFlag == ACCUM_ARRAY &&          //  Upres-->Accumulator
                                                                    //  Incoming layer free to be < Accumulator size
       outputLen_Upres(nn->upreslayers + src) > nn->accumlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && dstFlag == LSTM_ARRAY &&           //  Upres-->LSTM
       outputLen_Upres(nn->upreslayers + src) != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && dstFlag == GRU_ARRAY &&            //  Upres-->GRU
       outputLen_Upres(nn->upreslayers + src) != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && dstFlag == POOL_ARRAY &&           //  Upres-->Pool
       outputLen_Upres(nn->upreslayers + src) != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d does not match input for pool layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && dstFlag == UPRES_ARRAY &&          //  Upres-->Upres
       outputLen_Upres(nn->upreslayers + src) != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == UPRES_ARRAY && dstFlag == NORMAL_ARRAY &&         //  Upres-->Normalization
       outputLen_Upres(nn->upreslayers + src) != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from upres layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    if(srcFlag == NORMAL_ARRAY && dstFlag == DENSE_ARRAY &&         //  Normalization-->Dense
       nn->normlayers[src].i != nn->denselayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d does not match input for dense layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && dstFlag == CONV2D_ARRAY &&        //  Normalization-->Conv2D
       nn->normlayers[src].i != nn->convlayers[dst].inputW * nn->convlayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d does not match input for convolutional layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && dstFlag == ACCUM_ARRAY &&         //  Normalization-->Accumulator
       nn->normlayers[src].i > nn->accumlayers[dst].i)              //  Incoming layer free to be < Accumulator size
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d exceeds input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && dstFlag == LSTM_ARRAY &&          //  Normalization-->LSTM
       nn->normlayers[src].i != nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && dstFlag == GRU_ARRAY &&           //  Normalization-->GRU
       nn->normlayers[src].i != nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d does not match input for GRU layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && dstFlag == POOL_ARRAY &&          //  Normalization-->Pool
       nn->normlayers[src].i != nn->poollayers[dst].inputW * nn->poollayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d does not match input for pool layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && dstFlag == UPRES_ARRAY &&         //  Normalization-->Upres
       nn->normlayers[src].i != nn->upreslayers[dst].inputW * nn->upreslayers[dst].inputH)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d does not match input for upres layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == NORMAL_ARRAY && dstFlag == NORMAL_ARRAY &&        //  Normalization-->Normalization
       nn->normlayers[src].i != nn->normlayers[dst].i)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from normalization layer %d does not match input for normalization layer %d.\n", src, dst);
        #endif
        return false;
      }

    i = 0;                                                          //  Does this edge exist already?
    while(i < nn->len && !( nn->edgelist[i].srcType       == srcFlag       &&
                            nn->edgelist[i].srcIndex      == src           &&
                            nn->edgelist[i].selectorStart == selectorStart &&
                            nn->edgelist[i].selectorEnd   == selectorEnd   &&
                            nn->edgelist[i].dstType       == dstFlag       &&
                            nn->edgelist[i].dstIndex      == dst           ))
      i++;
    if(i < nn->len)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because (%d, %d)-->(%d, %d) exists already.\n", srcFlag, src, dstFlag, dst);
        #endif
        return false;
      }

    len++;                                                          //  Check whether adding the proposed edge
    if((queue = (Node*)malloc(sizeof(Node))) == NULL)               //  creates a cycle. Use DFS.
      {
        printf("ERROR: Unable to allocate DFS queue\n");
        exit(1);
      }
    if((node = (Node*)malloc(sizeof(Node))) == NULL)                //  Allocate space for the popped Node
      {
        printf("ERROR: Unable to allocate single node for DFS\n");
        exit(1);
      }
    queue[len - 1].type = INPUT_ARRAY;                              //  Enqueue the input Node
    queue[len - 1].index = 0;
    while(len > 0)
      {
        node->type = queue[0].type;                                 //  "Pop" the first node from the queue
        node->index = queue[0].index;
        if(len == 1)
          {
            free(queue);
            len = 0;
          }
        else
          {
            if((tmp = (Node*)malloc((len - 1) * sizeof(Node))) == NULL)
              {
                printf("ERROR: Unable to allocate temp queue for copying\n");
                exit(1);
              }
            for(i = 1; i < len; i++)
              {
                tmp[i - 1].type = queue[i].type;
                tmp[i - 1].index = queue[i].index;
              }
            free(queue);
            len--;
            if((queue = (Node*)malloc(len * sizeof(Node))) == NULL)
              {
                printf("ERROR: Unable to allocate reduced queue\n");
                exit(1);
              }
            for(i = 0; i < len; i++)
              {
                queue[i].type = tmp[i].type;
                queue[i].index = tmp[i].index;
              }
            free(tmp);
          }

        i = 0;
        while(i < vlen && !(visited[i].type == node->type && visited[i].index == node->index))
          i++;
        if(i == vlen)                                               //  Node has NOT been visited already
          {
            if(++vlen == 1)
              {
                if((visited = (Node*)malloc(sizeof(Node))) == NULL)
                  {
                    printf("ERROR: Unable to allocate visited array\n");
                    exit(1);
                  }
              }
            else
              {
                if((visited = (Node*)realloc(visited, vlen * sizeof(Node))) == NULL)
                  {
                    printf("ERROR: Unable to allocate visited array\n");
                    exit(1);
                  }
              }
            visited[vlen - 1].type = node->type;                    //  Mark the popped Node as visited
            visited[vlen - 1].index = node->index;
                                                                    //  Does the proposed link depart
            if(srcFlag == node->type && src == node->index)         //  from the popped Node?
              {
                                                                    //  Does the proposed edge lead to a node
                i = 0;                                              //  we've already visited?
                while(i < vlen && !(visited[i].type == dstFlag && visited[i].index == dst))
                  i++;
                if(i < vlen)                                        //  If so, then the proposed edge creates a cycle!
                  {
                    if(len > 0)
                      free(queue);
                    if(vlen > 0)
                      free(visited);
                    free(node);
                    #ifdef __NEURON_DEBUG
                    printf("Edge rejected because it creates a cycle.\n");
                    #endif
                    return false;
                  }

                if(++len == 1)
                  {
                    if((queue = (Node*)malloc(sizeof(Node))) == NULL)
                      {
                        printf("ERROR: Unable to allocate expanded queue\n");
                        exit(1);
                      }
                  }
                else
                  {
                    if((queue = (Node*)realloc(queue, len * sizeof(Node))) == NULL)
                      {
                        printf("ERROR: Unable to re-allocate expanded queue\n");
                        exit(1);
                      }
                  }
                queue[len - 1].type = dstFlag;
                queue[len - 1].index = dst;
              }
                                                                    //  Find all existing connections
            for(i = 0; i < nn->len; i++)                            //  from the popped Node, enqueue them.
              {                                                     //  Enqueue a Node if it's reachable from 'node'
                if(nn->edgelist[i].srcType == node->type && nn->edgelist[i].srcIndex == node->index)
                  {
                    j = 0;                                          //  Do we already have this connection?
                    while(j < len && !(queue[j].type  == nn->edgelist[i].dstType &&
                                       queue[j].index == nn->edgelist[i].dstIndex))
                      j++;
                    if(j == len)
                      {
                        if(++len == 1)
                          {
                            if((queue = (Node*)malloc(sizeof(Node))) == NULL)
                              {
                                printf("ERROR: Unable to allocate expanded queue\n");
                                exit(1);
                              }
                          }
                        else
                          {
                            if((queue = (Node*)realloc(queue, len * sizeof(Node))) == NULL)
                              {
                                printf("ERROR: Unable to re-allocate expanded queue\n");
                                exit(1);
                              }
                          }
                        queue[len - 1].type = nn->edgelist[i].dstType;
                        queue[len - 1].index = nn->edgelist[i].dstIndex;
                      }
                  }
              }
          }
      }

    if(vlen > 0)
      free(visited);
    free(node);

    nn->len++;                                                      //  Add the edge
    if(nn->len == 1)
      {
        if((nn->edgelist = (Edge*)malloc(sizeof(Edge))) == NULL)
          {
            printf("ERROR: Unable to allocate edge list\n");
            exit(1);
          }
      }
    else
      {
        if((nn->edgelist = (Edge*)realloc(nn->edgelist, nn->len * sizeof(Edge))) == NULL)
          {
            printf("ERROR: Unable to re-allocate edge list\n");
            exit(1);
          }
      }
    nn->edgelist[nn->len - 1].srcType = srcFlag;
    nn->edgelist[nn->len - 1].srcIndex = src;

    nn->edgelist[nn->len - 1].selectorStart = selectorStart;
    nn->edgelist[nn->len - 1].selectorEnd = selectorEnd;

    nn->edgelist[nn->len - 1].dstType = dstFlag;
    nn->edgelist[nn->len - 1].dstIndex = dst;

    return true;
  }

/* ONCE the NeuralNet 'nn' has been initialized, load a specific network from the file 'filename'. */
bool load_NN(char* filename, NeuralNet* nn)
  {
    FILE* fp;
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;
    unsigned int i, j;

    #ifdef __NEURON_DEBUG
    printf("load_NN(%s)\n", filename);
    #endif

    fp = fopen(filename, "rb");

    if((ucharBuffer = (unsigned char*)malloc(sizeof(char))) == NULL)//  Allocate 1 uchar
      {
        printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
        exit(1);
      }
                                                                    //  Allocate 10 uints
    if((uintBuffer = (unsigned int*)malloc(10 * sizeof(int))) == NULL)
      {
        printf("ERROR: Unable to allocate unsigned int buffer for reading from file\n");
        exit(1);
      }
    if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)    //  Allocate 1 double
      {
        printf("ERROR: Unable to allocate double buffer for reading from file\n");
        exit(1);
      }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 10
                                                                    //  doubleBuffer --> 1

    if(fread(uintBuffer, sizeof(int), 10, fp) != 10)                //  Read 10 objects of size int into buffer
      {
        printf("ERROR: Unable to read network parameters into buffer\n");
        exit(1);
      }
    nn->i = uintBuffer[0];                                          //  Read NeuralNet input count from buffer
    nn->len = uintBuffer[1];                                        //  Read NeuralNet edge list length from buffer
    nn->denseLen = uintBuffer[2];                                   //  Read NeuralNet DenseLayer list length from buffer
    nn->convLen = uintBuffer[3];                                    //  Read NeuralNet Conv2DLayer list length from buffer
    nn->accumLen = uintBuffer[4];                                   //  Read NeuralNet AccumLayer list length from buffer
    nn->lstmLen = uintBuffer[5];                                    //  Read NeuralNet LSTMLayer list length from buffer
    nn->gruLen = uintBuffer[6];                                     //  Read NeuralNet GRULayer list length from buffer
    nn->poolLen = uintBuffer[7];                                    //  Read NeuralNet Pool2DLayer list length from buffer
    nn->upresLen = uintBuffer[8];                                   //  Read NeuralNet UpresLayer list length from buffer
    nn->normalLen = uintBuffer[9];                                  //  Read NeuralNet NormalLayer list length from buffer
    #ifdef __NEURON_DEBUG
    printf("  nn->i         = %d\n", nn->i);
    printf("  nn->len       = %d\n", nn->len);
    printf("  nn->denseLen  = %d\n", nn->denseLen);
    printf("  nn->convLen   = %d\n", nn->convLen);
    printf("  nn->accumLen  = %d\n", nn->accumLen);
    printf("  nn->lstmLen   = %d\n", nn->lstmLen);
    printf("  nn->gruLen    = %d\n", nn->gruLen);
    printf("  nn->poolLen   = %d\n", nn->poolLen);
    printf("  nn->upresLen  = %d\n", nn->upresLen);
    printf("  nn->normalLen = %d\n", nn->normalLen);
    #endif

    if(fread(ucharBuffer, sizeof(char), 1, fp) != 1)                //  Read 1 object of size char into buffer
      {
        printf("ERROR: Unable to read number of network variables into buffer\n");
        exit(1);
      }
    nn->vars = ucharBuffer[0];                                      //  Read NeuralNet variable count from buffer
    #ifdef __NEURON_DEBUG
    printf("  nn->vars     = %d\n", nn->vars);
    #endif

    if(fread(uintBuffer, sizeof(int), 1, fp) != 1)                  //  Read 1 object of size int into buffer
      {
        printf("ERROR: Unable to read network generation into buffer\n");
        exit(1);
      }
    nn->gen = uintBuffer[0];                                        //  Read NeuralNet generation/epoch from buffer
    #ifdef __NEURON_DEBUG
    printf("  nn->gen      = %d\n", nn->gen);
    #endif

    if(fread(doubleBuffer, sizeof(double), 1, fp) != 1)             //  Read 1 object of size double into buffer
      {
        printf("ERROR: Unable to read network fitness into buffer\n");
        exit(1);
      }
    nn->fit = doubleBuffer[0];                                      //  Read NeuralNet fitness from buffer
    #ifdef __NEURON_DEBUG
    printf("  nn->fit      = %.6f\n", nn->fit);
    #endif

    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, COMMSTR_LEN * sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
        exit(1);
      }
                                                                    //  ucharBuffer  --> COMMSTR_LEN
                                                                    //  uintBuffer   --> 10
                                                                    //  doubleBuffer --> 1

                                                                    //  Read COMMSTR_LEN objects of size char into buffer
    if(fread(ucharBuffer, sizeof(char), COMMSTR_LEN, fp) != COMMSTR_LEN)
      {
        printf("ERROR: Unable to read network comment into buffer\n");
        exit(1);
      }
    for(i = 0; i < COMMSTR_LEN; i++)                                //  Read NeuralNet comment from buffer
      nn->comment[i] = ucharBuffer[i];
    #ifdef __NEURON_DEBUG
    printf("  nn->comment  = %s\n", nn->comment);
    #endif
                                                                    //  Allocate network's edge list
    if((nn->edgelist = (Edge*)malloc(nn->len * sizeof(Edge))) == NULL)
      {
        printf("ERROR: Unable to allocate edge list while reading from file\n");
        exit(1);
      }
    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
        exit(1);
      }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 10
                                                                    //  doubleBuffer --> 1
    #ifdef __NEURON_DEBUG
    printf("  Edge List:\n");
    #endif
    for(i = 0; i < nn->len; i++)                                    //  Read all Edges from file
      {
        if(fread(ucharBuffer, sizeof(char), 1, fp) != 1)            //  Read 1 object of size char into buffer
          {
            printf("ERROR: Unable to read network edge source into buffer\n");
            exit(1);
          }
        nn->edgelist[i].srcType = ucharBuffer[0];                   //  Read edge source type from buffer
        if(fread(uintBuffer, sizeof(int), 3, fp) != 3)              //  Read 3 objects of size int into buffer
          {
            printf("ERROR: Unable to read network edge descriptors into buffer\n");
            exit(1);
          }
        nn->edgelist[i].srcIndex = uintBuffer[0];                   //  Read edge source index from buffer
        nn->edgelist[i].selectorStart = uintBuffer[1];              //  Read edge selector start from buffer
        nn->edgelist[i].selectorEnd = uintBuffer[2];                //  Read edge selector end from buffer
        if(fread(ucharBuffer, sizeof(char), 1, fp) != 1)            //  Read 1 object of size char into buffer
          {
            printf("ERROR: Unable to read network edge destination type into buffer\n");
            exit(1);
          }
        nn->edgelist[i].dstType = ucharBuffer[0];                   //  Read edge destination type from buffer
        if(fread(uintBuffer, sizeof(int), 1, fp) != 1)              //  Read 1 object of size int into buffer
          {
            printf("ERROR: Unable to read network edge destination into buffer\n");
            exit(1);
          }
        nn->edgelist[i].dstIndex = uintBuffer[0];                   //  Read edge destination index from buffer

        #ifdef __NEURON_DEBUG
        printf("    (%d, %d, %d, %d, %d, %d)\n", nn->edgelist[i].srcType,       nn->edgelist[i].srcIndex,
                                                 nn->edgelist[i].selectorStart, nn->edgelist[i].selectorEnd,
                                                 nn->edgelist[i].dstType,       nn->edgelist[i].dstIndex);
        #endif
      }
    #ifdef __NEURON_DEBUG
    printf("\n");
    #endif

    free(ucharBuffer);                                              //  ucharBuffer  --> 0
    free(uintBuffer);                                               //  uintBuffer   --> 0
    free(doubleBuffer);                                             //  doubleBuffer --> 0

    if(nn->denseLen > 0)
      {
                                                                    //  Allocate network's DenseLayer array
        if((nn->denselayers = (DenseLayer*)malloc(nn->denseLen * sizeof(DenseLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate DenseLayer array while reading from file\n");
            exit(1);
          }
        if(!read_Dense(nn->denselayers, nn->denseLen, fp))
          {
            printf("ERROR: Failed to read network dense layers\n");
            exit(1);
          }
      }

    if(nn->convLen > 0)
      {
                                                                    //  Allocate network's Conv2DLayer array
        if((nn->convlayers = (Conv2DLayer*)malloc(nn->convLen * sizeof(Conv2DLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer array while reading from file\n");
            exit(1);
          }
        if(!read_Conv2D(nn->convlayers, nn->convLen, fp))
          {
            printf("ERROR: Failed to read network convolutional(2D) layers\n");
            exit(1);
          }
      }

    if(nn->accumLen > 0)
      {
                                                                    //  Allocate network's AccumLayer array
        if((nn->accumlayers = (AccumLayer*)malloc(nn->accumLen * sizeof(AccumLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate AccumLayer array while reading from file\n");
            exit(1);
          }
        if(!read_Accum(nn->accumlayers, nn->accumLen, fp))
          {
            printf("ERROR: Failed to read network accumulation layers\n");
            exit(1);
          }
      }

    if(nn->lstmLen > 0)
      {
                                                                    //  Allocate network's LSTMLayer array
        if((nn->lstmlayers = (LSTMLayer*)malloc(nn->lstmLen * sizeof(LSTMLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate LSTMLayer array while reading from file\n");
            exit(1);
          }
        if(!read_LSTM(nn->lstmlayers, nn->lstmLen, fp))
          {
            printf("ERROR: Failed to read network LSTM layers\n");
            exit(1);
          }
      }

    if(nn->gruLen > 0)
      {
                                                                    //  Allocate network's GRULayer array
        if((nn->grulayers = (GRULayer*)malloc(nn->gruLen * sizeof(GRULayer))) == NULL)
          {
            printf("ERROR: Unable to allocate GRULayer array while reading from file\n");
            exit(1);
          }
        if(!read_GRU(nn->grulayers, nn->gruLen, fp))
          {
            printf("ERROR: Failed to read network GRU layers\n");
            exit(1);
          }
      }

    if(nn->poolLen > 0)
      {
                                                                    //  Allocate network's Pool2DLayer array
        if((nn->poollayers = (Pool2DLayer*)malloc(nn->poolLen * sizeof(Pool2DLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate Pool2DLayer array while reading from file\n");
            exit(1);
          }
        if(!read_Pool2D(nn->poollayers, nn->poolLen, fp))
          {
            printf("ERROR: Failed to read network 2D pooling layers\n");
            exit(1);
          }
      }

    if(nn->upresLen > 0)
      {
                                                                    //  Allocate network's UpresLayer array
        if((nn->upreslayers = (UpresLayer*)malloc(nn->upresLen * sizeof(UpresLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresLayer array while reading from file\n");
            exit(1);
          }
        if(!read_Upres(nn->upreslayers, nn->upresLen, fp))
          {
            printf("ERROR: Failed to read network up-resolution layers\n");
            exit(1);
          }
      }

    if(nn->normalLen > 0)
      {
                                                                    //  Allocate network's NormalLayer array
        if((nn->normlayers = (NormalLayer*)malloc(nn->normalLen * sizeof(NormalLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate NormalLayer array while reading from file\n");
            exit(1);
          }
        if(!read_Normal(nn->normlayers, nn->normalLen, fp))
          {
            printf("ERROR: Failed to read network normalization layers\n");
            exit(1);
          }
      }

    if(nn->vars > 0)                                                //  Read all Variables
      {
                                                                    //  Allocate network's Variable array
        if((nn->variables = (Variable*)malloc(nn->vars * sizeof(Variable))) == NULL)
          {
            printf("ERROR: Unable to allocate Variable array while reading from file\n");
            exit(1);
          }
        if((ucharBuffer = (unsigned char*)malloc(VARSTR_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            exit(1);
          }
        if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer for reading from file\n");
            exit(1);
          }

        for(i = 0; i < nn->vars; i++)                               //  Write all Variables to file
          {
                                                                    //  Read VARSTR_LEN objects of size char into buffer
            if(fread(ucharBuffer, sizeof(char), VARSTR_LEN, fp) != VARSTR_LEN)
              {
                printf("ERROR: Unable to read variable string into buffer\n");
                exit(1);
              }
            for(j = 0; j < VARSTR_LEN; j++)
              nn->variables[i].key[j] = ucharBuffer[j];
            #ifdef __NEURON_DEBUG
            printf("  nn->variables[%d].key = %s\n", i, nn->variables[i].key);
            #endif

            if(fread(doubleBuffer, sizeof(double), 1, fp) != 1)     //  Read buffer, write 1 object of size double
              {
                printf("ERROR: Unable to read variable value into buffer\n");
                exit(1);
              }
            nn->variables[i].value = doubleBuffer[0];
            #ifdef __NEURON_DEBUG
            printf("  nn->variables[%d].value = %.6f\n", i, nn->variables[i].value);
            #endif
          }

        free(ucharBuffer);
        free(doubleBuffer);
      }

    fclose(fp);

    return true;
  }

/* Write the given Neural Network to a binary file named 'filename'. */
bool write_NN(char* filename, NeuralNet* nn)
  {
    FILE* fp;
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;
    unsigned int i, j;

    #ifdef __NEURON_DEBUG
    printf("write_NN(%s)\n", filename);
    #endif

    fp = fopen(filename, "wb");                                     //  Create file for binary writting
                                                                    //  Write number of network inputs
    if((ucharBuffer = (unsigned char*)malloc(sizeof(char))) == NULL)//  Allocate 1 uchar
      {
        printf("ERROR: Unable to allocate unsigned char buffer for writing to file\n");
        exit(1);
      }
                                                                    //  Allocate 10 uints
    if((uintBuffer = (unsigned int*)malloc(10 * sizeof(int))) == NULL)
      {
        printf("ERROR: Unable to allocate unsigned int buffer for writing to file\n");
        exit(1);
      }
    if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)    //  Allocate 1 double
      {
        printf("ERROR: Unable to allocate double buffer for writing to file\n");
        exit(1);
      }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 10
                                                                    //  doubleBuffer --> 1
    uintBuffer[0] = nn->i;                                          //  Save NeuralNet input count to buffer
    uintBuffer[1] = nn->len;                                        //  Save NeuralNet edge count to buffer
    uintBuffer[2] = nn->denseLen;                                   //  Save number of Dense Layers to buffer
    uintBuffer[3] = nn->convLen;                                    //  Save number of Convolutional Layers to buffer
    uintBuffer[4] = nn->accumLen;                                   //  Save number of Accumulator Layers to buffer
    uintBuffer[5] = nn->lstmLen;                                    //  Save number of LSTM Layers to buffer
    uintBuffer[6] = nn->gruLen;                                     //  Save number of GRU Layers to buffer
    uintBuffer[7] = nn->poolLen;                                    //  Save number of Pool Layers to buffer
    uintBuffer[8] = nn->upresLen;                                   //  Save number of Upres Layers to buffer
    uintBuffer[9] = nn->normalLen;                                  //  Save number of Normal Layers to buffer
    if(fwrite(uintBuffer, sizeof(int), 10, fp) != 10)               //  From buffer, write 10 objects of size int
      {
        printf("ERROR: Unable to write to file from unsigned int buffer.\n");
        exit(1);
      }
    #ifdef __NEURON_DEBUG
    printf("  uintBuffer[0] = nn->i         = %d\n", uintBuffer[0]);
    printf("  uintBuffer[1] = nn->len       = %d\n", uintBuffer[1]);
    printf("  uintBuffer[2] = nn->denseLen  = %d\n", uintBuffer[2]);
    printf("  uintBuffer[3] = nn->convLen   = %d\n", uintBuffer[3]);
    printf("  uintBuffer[4] = nn->accumLen  = %d\n", uintBuffer[4]);
    printf("  uintBuffer[5] = nn->lstmLen   = %d\n", uintBuffer[5]);
    printf("  uintBuffer[6] = nn->gruLen    = %d\n", uintBuffer[6]);
    printf("  uintBuffer[7] = nn->poolLen   = %d\n", uintBuffer[7]);
    printf("  uintBuffer[8] = nn->upresLen  = %d\n", uintBuffer[8]);
    printf("  uintBuffer[9] = nn->normalLen = %d\n", uintBuffer[9]);
    #endif

    ucharBuffer[0] = nn->vars;                                      //  Save number of Variables to buffer
    if(fwrite(ucharBuffer, sizeof(char), 1, fp) != 1)               //  From buffer, write 1 object of size char
      {
        printf("ERROR: Unable to write to file from unsigned char buffer.\n");
        exit(1);
      }
    #ifdef __NEURON_DEBUG
    printf("  ucharBuffer[0] = nn->vars    = %d\n", ucharBuffer[0]);
    #endif

    uintBuffer[0] = nn->gen;                                        //  Save generation/epoch to buffer
    if(fwrite(uintBuffer, sizeof(int), 1, fp) != 1)                 //  From buffer, write 1 object of size int
      {
        printf("ERROR: Unable to write to file from unsigned int buffer.\n");
        exit(1);
      }
    #ifdef __NEURON_DEBUG
    printf("  uintBuffer[0] = nn->gen      = %d\n", uintBuffer[0]);
    #endif

    doubleBuffer[0] = nn->fit;                                      //  Save fitness to buffer
    if(fwrite(doubleBuffer, sizeof(double), 1, fp) != 1)            //  From buffer, write 1 object of size double
      {
        printf("ERROR: Unable to write to file from double buffer.\n");
        exit(1);
      }
    #ifdef __NEURON_DEBUG
    printf("  doubleBuffer[0] = nn->fit    = %.6f\n", doubleBuffer[0]);
    #endif
                                                                    //  Write network comment to file
    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, COMMSTR_LEN * sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to re-allocate unsigned char buffer for writing to file\n");
        exit(1);
      }
                                                                    //  ucharBuffer  --> COMMSTR_LEN
                                                                    //  uintBuffer   --> 10
                                                                    //  doubleBuffer --> 1
    for(i = 0; i < COMMSTR_LEN; i++)
      ucharBuffer[i] = nn->comment[i];
                                                                    //  From buffer, write COMMSTR_LEN objects of size char
    if(fwrite(ucharBuffer, sizeof(char), COMMSTR_LEN, fp) != COMMSTR_LEN)
      {
        printf("ERROR: Unable to write to file from unsigned char buffer.\n");
        exit(1);
      }
    #ifdef __NEURON_DEBUG
    printf("  ucharBuffer = %s\n", ucharBuffer);
    #endif
                                                                    //  Shrink buffer to 1
    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
        exit(1);
      }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 10
                                                                    //  doubleBuffer --> 1
    #ifdef __NEURON_DEBUG
    printf("  Edge List:\n");
    #endif
    for(i = 0; i < nn->len; i++)                                    //  Write all Edges to file
      {
        ucharBuffer[0] = nn->edgelist[i].srcType;                   //  Save edge source type to buffer
        if(fwrite(ucharBuffer, sizeof(char), 1, fp) != 1)           //  From buffer, write 1 object of size char
          {
            printf("ERROR: Unable to write to file from unsigned char buffer.\n");
            exit(1);
          }
        #ifdef __NEURON_DEBUG
        printf("    ucharBuffer[0] = %d\n", ucharBuffer[0]);
        #endif

        uintBuffer[0] = nn->edgelist[i].srcIndex;                   //  Save edge source index to buffer
        uintBuffer[1] = nn->edgelist[i].selectorStart;              //  Save edge selector start to buffer
        uintBuffer[2] = nn->edgelist[i].selectorEnd;                //  Save edge selector end to buffer
        if(fwrite(uintBuffer, sizeof(int), 3, fp) != 3)             //  From buffer, write 3 objects of size int
          {
            printf("ERROR: Unable to write to file from unsigned int buffer.\n");
            exit(1);
          }
        #ifdef __NEURON_DEBUG
        printf("    uintBuffer[0] = %d\n", uintBuffer[0]);
        printf("    uintBuffer[1] = %d\n", uintBuffer[1]);
        printf("    uintBuffer[2] = %d\n", uintBuffer[2]);
        #endif

        ucharBuffer[0] = nn->edgelist[i].dstType;                   //  Save edge destination type to buffer
        if(fwrite(ucharBuffer, sizeof(char), 1, fp) != 1)           //  From buffer, write 1 object of size char
          {
            printf("ERROR: Unable to write to file from unsigned char buffer.\n");
            exit(1);
          }
        #ifdef __NEURON_DEBUG
        printf("    ucharBuffer[0] = %d\n", ucharBuffer[0]);
        #endif

        uintBuffer[0] = nn->edgelist[i].dstIndex;                   //  Save edge destination index to buffer
        if(fwrite(uintBuffer, sizeof(int), 1, fp) != 1)             //  From buffer, write 1 object of size int
          {
            printf("ERROR: Unable to write to file from unsigned int buffer.\n");
            exit(1);
          }
        #ifdef __NEURON_DEBUG
        printf("    uintBuffer[0] = %d\n\n", uintBuffer[0]);
        #endif
      }

    free(ucharBuffer);                                              //  ucharBuffer  --> 0
    free(uintBuffer);                                               //  uintBuffer   --> 0
    free(doubleBuffer);                                             //  doubleBuffer --> 0

    if(nn->denseLen > 0)
      {
        if(!write_Dense(nn->denselayers, nn->denseLen, fp))
          {
            printf("ERROR: Failed to write network dense layers\n");
            exit(1);
          }
      }

    if(nn->convLen > 0)
      {
        if(!write_Conv2D(nn->convlayers, nn->convLen, fp))
          {
            printf("ERROR: Failed to write network convolutional(2D) layers\n");
            exit(1);
          }
      }

    if(nn->accumLen > 0)
      {
        if(!write_Accum(nn->accumlayers, nn->accumLen, fp))
          {
            printf("ERROR: Failed to write network accumulation layers\n");
            exit(1);
          }
      }

    if(nn->lstmLen > 0)
      {
        if(!write_LSTM(nn->lstmlayers, nn->lstmLen, fp))
          {
            printf("ERROR: Failed to write network LSTM layers\n");
            exit(1);
          }
      }

    if(nn->gruLen > 0)
      {
        if(!write_GRU(nn->grulayers, nn->gruLen, fp))
          {
            printf("ERROR: Failed to write network GRU layers\n");
            exit(1);
          }
      }

    if(nn->poolLen > 0)
      {
        if(!write_Pool2D(nn->poollayers, nn->poolLen, fp))
          {
            printf("ERROR: Failed to write network 2D pooling layers\n");
            exit(1);
          }
      }

    if(nn->upresLen > 0)
      {
        if(!write_Upres(nn->upreslayers, nn->upresLen, fp))
          {
            printf("ERROR: Failed to write network up-resolution layers\n");
            exit(1);
          }
      }

    if(nn->normalLen > 0)
      {
        if(!write_Normal(nn->normlayers, nn->normalLen, fp))
          {
            printf("ERROR: Failed to write network normalization layers\n");
            exit(1);
          }
      }

    for(i = 0; i < nn->vars; i++)                                   //  Write all Variables to file
      {
        if((ucharBuffer = (unsigned char*)malloc(VARSTR_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for writing to file\n");
            exit(1);
          }
        if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < VARSTR_LEN; j++)
          ucharBuffer[j] = nn->variables[i].key[j];
        fwrite(ucharBuffer, sizeof(char), VARSTR_LEN, fp);          //  From buffer, write VARSTR_LEN chars
        #ifdef __NEURON_DEBUG
        printf("  ucharBuffer = %s\n", ucharBuffer);
        #endif

        doubleBuffer[0] = nn->variables[i].value;
        fwrite(doubleBuffer, sizeof(double), 1, fp);                //  From buffer, write 1 object of size double
        #ifdef __NEURON_DEBUG
        printf("  doubleBuffer[0] = %.6f\n\n", doubleBuffer[0]);
        #endif

        free(ucharBuffer);
        free(doubleBuffer);
      }

    fclose(fp);

    return true;
  }

/* Put the given network's edge-list in "computing order," from the input layer to the output layer. */
void sortEdges(NeuralNet* nn)
  {
    unsigned int i, j, k, l;
    Node* nodelist;                                                 //  Find which network layers do not have outbounds
    unsigned int listlen;                                           //  Nodes are network layers, plus one input layer

    Edge* newlist;                                                  //  The sorted version we will copy into nn->edgelist
    unsigned int newlistlen;
    unsigned int ptr;

    Edge* addition;                                                 //  Track latest addition to sorted list
    unsigned int additionlen;

    Edge* swap = NULL;                                              //  Hold Edge objects to be moved
    unsigned int swaplen;

    #ifdef __NEURON_DEBUG
    printf("sortEdges()\n");
    #endif

    listlen = nn->denseLen + nn->convLen + nn->accumLen + nn->lstmLen + nn->gruLen + nn->poolLen + nn->upresLen + nn->normalLen + 1;
    if((nodelist = (Node*)malloc(listlen * sizeof(Node))) == NULL)  //  Allocate an array of Nodes (type, index)
      {
        printf("ERROR: Unable to allocate node array\n");
        exit(1);
      }
    nodelist[0].type = INPUT_ARRAY;                                 //  Set the first Node to the input layer
    nodelist[0].index = 0;
    j = 1;
    for(i = 0; i < nn->denseLen; i++)                               //  Add all Dense Layers
      {
        nodelist[j].type = DENSE_ARRAY;
        nodelist[j].index = i;
        j++;
      }
    for(i = 0; i < nn->convLen; i++)                                //  Add all Convolutional Layers
      {
        nodelist[j].type = CONV2D_ARRAY;
        nodelist[j].index = i;
        j++;
      }
    for(i = 0; i < nn->accumLen; i++)                               //  Add all Accumulator Layers
      {
        nodelist[j].type = ACCUM_ARRAY;
        nodelist[j].index = i;
        j++;
      }
    for(i = 0; i < nn->lstmLen; i++)                                //  Add all LSTM Layers
      {
        nodelist[j].type = LSTM_ARRAY;
        nodelist[j].index = i;
        j++;
      }
    for(i = 0; i < nn->gruLen; i++)                                 //  Add all GRU Layers
      {
        nodelist[j].type = GRU_ARRAY;
        nodelist[j].index = i;
        j++;
      }
    for(i = 0; i < nn->poolLen; i++)                                //  Add all Pool Layers
      {
        nodelist[j].type = POOL_ARRAY;
        nodelist[j].index = i;
        j++;
      }
    for(i = 0; i < nn->upresLen; i++)                               //  Add all Upres Layers
      {
        nodelist[j].type = UPRES_ARRAY;
        nodelist[j].index = i;
        j++;
      }
    for(i = 0; i < nn->normalLen; i++)                              //  Add all Normal Layers
      {
        nodelist[j].type = NORMAL_ARRAY;
        nodelist[j].index = i;
        j++;
      }

    //  By now we have an array of Nodes, one for each network layer, including the input layer.
    //  If we go through all network edges and cannot find an edge with a given layer as a source,
    //  then it is a network output

    swaplen = 0;
    for(i = 0; i < listlen; i++)                                    //  Go through all layers/nodes
      {
        j = 0;                                                      //  Try to find an outbound edge from this layer/node
        while(j < nn->len && !(nn->edgelist[j].srcType == nodelist[i].type &&
                               nn->edgelist[j].srcIndex == nodelist[i].index))
          j++;
        if(j == nn->len)                                            //  No outbound edge found:
          {                                                         //  this is a network output layer
                                                                    //  Add all edges that reach
            for(j = 0; j < nn->len; j++)                            //   (nodelist[i].type, nodelist[i].index)
              {
                if(nn->edgelist[j].dstType == nodelist[i].type && nn->edgelist[j].dstIndex == nodelist[i].index)
                  {
                    if(++swaplen == 1)
                      {
                        if((swap = (Edge*)malloc(sizeof(Edge))) == NULL)
                          {
                            printf("ERROR: Unable to allocate edge swap array\n");
                            exit(1);
                          }
                      }
                    else
                      {
                        if((swap = (Edge*)realloc(swap, swaplen * sizeof(Edge))) == NULL)
                          {
                            printf("ERROR: Unable to allocate edge swap array\n");
                            exit(1);
                          }
                      }
                    swap[swaplen - 1].srcType = nn->edgelist[j].srcType;
                    swap[swaplen - 1].srcIndex = nn->edgelist[j].srcIndex;
                    swap[swaplen - 1].selectorStart = nn->edgelist[j].selectorStart;
                    swap[swaplen - 1].selectorEnd = nn->edgelist[j].selectorEnd;
                    swap[swaplen - 1].dstType = nn->edgelist[j].dstType;
                    swap[swaplen - 1].dstIndex = nn->edgelist[j].dstIndex;
                  }
              }
          }
      }

    ptr = nn->len - 1;                                              //  Point to new list's last cell
    newlistlen = nn->len;                                           //  Allocate new edge list
    if((newlist = (Edge*)malloc(newlistlen * sizeof(Edge))) == NULL)
      {
        printf("ERROR: Unable to allocate new edge list array\n");
        exit(1);
      }

    while(swaplen > 0)                                              //  Loop until 'swap' comes up empty
      {
        additionlen = swaplen;                                      //  Allocate 'addition' array
        if((addition = (Edge*)malloc(additionlen * sizeof(Edge))) == NULL)
          {
            printf("ERROR: Unable to allocate array-addition buffer\n");
            exit(1);
          }
                                                                    //  Copy swap --> addition
        for(i = 0; i < swaplen; i++)                                //   and swap --> newlist at ptr
          {
            addition[i].srcType = swap[i].srcType;
            addition[i].srcIndex = swap[i].srcIndex;
            addition[i].selectorStart = swap[i].selectorStart;
            addition[i].selectorEnd = swap[i].selectorEnd;
            addition[i].dstType = swap[i].dstType;
            addition[i].dstIndex = swap[i].dstIndex;

            newlist[ptr - (swaplen - 1) + i].srcType = swap[i].srcType;
            newlist[ptr - (swaplen - 1) + i].srcIndex = swap[i].srcIndex;
            newlist[ptr - (swaplen - 1) + i].selectorStart = swap[i].selectorStart;
            newlist[ptr - (swaplen - 1) + i].selectorEnd = swap[i].selectorEnd;
            newlist[ptr - (swaplen - 1) + i].dstType = swap[i].dstType;
            newlist[ptr - (swaplen - 1) + i].dstIndex = swap[i].dstIndex;
          }

        ptr -= swaplen;                                             //  "Advance" pointer (toward head of array)

        free(swap);                                                 //  Empty 'swap'
        swaplen = 0;

        for(i = 0; i < additionlen; i++)                            //  Scan over 'addition':
          {                                                         //  Find every edge that ends with a Node
            for(j = 0; j < nn->len; j++)                            //  with which any member of 'addition' begins.
              {
                if( nn->edgelist[j].dstType == addition[i].srcType &&
                    nn->edgelist[j].dstIndex == addition[i].srcIndex )
                  {
                                                                    //  Is this Edge already in newlist,
                    k = ptr + 1;                                    //  between ptr + 1 and the end of newlist?
                    while(k < newlistlen && !(nn->edgelist[j].srcType == newlist[k].srcType &&
                                              nn->edgelist[j].srcIndex == newlist[k].srcIndex &&
                                              nn->edgelist[j].dstType == newlist[k].dstType &&
                                              nn->edgelist[j].dstIndex == newlist[k].dstIndex ))
                      k++;
                                                                    //  If so, pull it out of newlist
                    if(k < newlistlen)                              //  and close the gap in newlist
                      {
                        for(l = k; l >= 1; l--)
                          {
                            newlist[l].srcType = newlist[l - 1].srcType;
                            newlist[l].srcIndex = newlist[l - 1].srcIndex;
                            newlist[l].selectorStart = newlist[l - 1].selectorStart;
                            newlist[l].selectorEnd = newlist[l - 1].selectorEnd;
                            newlist[l].dstType = newlist[l - 1].dstType;
                            newlist[l].dstIndex = newlist[l - 1].dstIndex;
                          }
                        ptr++;                                      //  Move toward array tail
                                                                    //  for the single element we took out
                      }
                                                                    //  Add it to swap
                    if(++swaplen == 1)
                      {
                        if((swap = (Edge*)malloc(sizeof(Edge))) == NULL)
                          {
                            printf("ERROR: Unable to allocate emptied edge swap buffer\n");
                            exit(1);
                          }
                      }
                    else
                      {
                        if((swap = (Edge*)realloc(swap, swaplen * sizeof(Edge))) == NULL)
                          {
                            printf("ERROR: Unable to re-allocate edge swap buffer\n");
                            exit(1);
                          }
                      }
                    swap[swaplen - 1].srcType = nn->edgelist[j].srcType;
                    swap[swaplen - 1].srcIndex = nn->edgelist[j].srcIndex;
                    swap[swaplen - 1].selectorStart = nn->edgelist[j].selectorStart;
                    swap[swaplen - 1].selectorEnd = nn->edgelist[j].selectorEnd;
                    swap[swaplen - 1].dstType = nn->edgelist[j].dstType;
                    swap[swaplen - 1].dstIndex = nn->edgelist[j].dstIndex;
                  }
              }
          }

        if(additionlen > 0)
          free(addition);
        additionlen = 0;
      }

    for(i = 0; i < nn->len; i++)                                    //  Sorting's complete:
      {                                                             //  write sorted edges to Neural Net
        nn->edgelist[i].srcType = newlist[i].srcType;
        nn->edgelist[i].srcIndex = newlist[i].srcIndex;
        nn->edgelist[i].selectorStart = newlist[i].selectorStart;
        nn->edgelist[i].selectorEnd = newlist[i].selectorEnd;
        nn->edgelist[i].dstType = newlist[i].dstType;
        nn->edgelist[i].dstIndex = newlist[i].dstIndex;
      }

    if(nn->len > 0)
      free(newlist);

    if(listlen > 0)
      free(nodelist);

    return;
  }

/* Find the network layer with the given 'name'.
   If it's found, we assume that YOU KNOW WHAT TYPE OF LAYER IT IS because all you'll get back
   is the array index.
   If it's not found, return UINT_MAX.
   Results are undefined if there are more than one layer with the same name. */
unsigned int nameIndex(char* name, NeuralNet* nn)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("nameIndex(%s)\n", name);
    #endif

    i = 0;                                                          //  Check Dense layers
    while(i < nn->denseLen && strcmp(nn->denselayers[i].name, name) != 0)
      i++;
    if(i < nn->denseLen)
      return i;

    i = 0;                                                          //  Check Conv2D layers
    while(i < nn->convLen && strcmp(nn->convlayers[i].name, name) != 0)
      i++;
    if(i < nn->convLen)
      return i;

    i = 0;                                                          //  Check Accum layers
    while(i < nn->accumLen && strcmp(nn->accumlayers[i].name, name) != 0)
      i++;
    if(i < nn->accumLen)
      return i;

    i = 0;                                                          //  Check LSTM layers
    while(i < nn->lstmLen && strcmp(nn->lstmlayers[i].name, name) != 0)
      i++;
    if(i < nn->lstmLen)
      return i;

    i = 0;                                                          //  Check GRU layers
    while(i < nn->gruLen && strcmp(nn->grulayers[i].name, name) != 0)
      i++;
    if(i < nn->gruLen)
      return i;

    i = 0;                                                          //  Check Pool layers
    while(i < nn->poolLen && strcmp(nn->poollayers[i].name, name) != 0)
      i++;
    if(i < nn->poolLen)
      return i;

    i = 0;                                                          //  Check Upres layers
    while(i < nn->upresLen && strcmp(nn->upreslayers[i].name, name) != 0)
      i++;
    if(i < nn->upresLen)
      return i;

    i = 0;                                                          //  Check Normalization layers
    while(i < nn->normalLen && strcmp(nn->normlayers[i].name, name) != 0)
      i++;
    if(i < nn->normalLen)
      return i;

    return UINT_MAX;
  }

/* Find the type of layer with the given 'name'.
   If the layer is found, return one of the flags above indicating the array in which it was found.
   If the layer is NOT found, return UCHAR_MAX.
   Results are undefined if there are more than one layer with the same name. */
unsigned char nameType(char* name, NeuralNet* nn)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("nameType(%s)\n", name);
    #endif

    i = 0;                                                          //  Check Dense layers
    while(i < nn->denseLen && strcmp(nn->denselayers[i].name, name) != 0)
      i++;
    if(i < nn->denseLen)
      return DENSE_ARRAY;

    i = 0;                                                          //  Check Conv2D layers
    while(i < nn->convLen && strcmp(nn->convlayers[i].name, name) != 0)
      i++;
    if(i < nn->convLen)
      return CONV2D_ARRAY;

    i = 0;                                                          //  Check Accum layers
    while(i < nn->accumLen && strcmp(nn->accumlayers[i].name, name) != 0)
      i++;
    if(i < nn->accumLen)
      return ACCUM_ARRAY;

    i = 0;                                                          //  Check LSTM layers
    while(i < nn->lstmLen && strcmp(nn->lstmlayers[i].name, name) != 0)
      i++;
    if(i < nn->lstmLen)
      return LSTM_ARRAY;

    i = 0;                                                          //  Check GRU layers
    while(i < nn->gruLen && strcmp(nn->grulayers[i].name, name) != 0)
      i++;
    if(i < nn->gruLen)
      return GRU_ARRAY;

    i = 0;                                                          //  Check Pool layers
    while(i < nn->poolLen && strcmp(nn->poollayers[i].name, name) != 0)
      i++;
    if(i < nn->poolLen)
      return POOL_ARRAY;

    i = 0;                                                          //  Check Upres layers
    while(i < nn->upresLen && strcmp(nn->upreslayers[i].name, name) != 0)
      i++;
    if(i < nn->upresLen)
      return UPRES_ARRAY;

    i = 0;                                                          //  Check Normalization layers
    while(i < nn->normalLen && strcmp(nn->normlayers[i].name, name) != 0)
      i++;
    if(i < nn->normalLen)
      return NORMAL_ARRAY;

    return UCHAR_MAX;
  }

/* Print a table of the network's edge list */
void printEdgeList(NeuralNet* nn)
  {
    unsigned int i;
    printf("Src\t\t\t\tDst\n");
    printf("Type\tIndex\tStart\tEnd\tType\tIndex\n");
    printf("=================================================\n");
    for(i = 0; i < nn->len; i++)
      printf("%d\t%d\t%d\t%d\t%d\t%d\n", nn->edgelist[i].srcType,
                                         nn->edgelist[i].srcIndex,
                                         nn->edgelist[i].selectorStart,
                                         nn->edgelist[i].selectorEnd,
                                         nn->edgelist[i].dstType,
                                         nn->edgelist[i].dstIndex);
    return;
  }

/* Print out a summary of the given network */
void print_NN(NeuralNet* nn)
  {
    unsigned int i;
    unsigned char j;
    unsigned int k;
    unsigned int convparams;
    bool firstInline;
    char buffer[16];
    unsigned char bufflen;

    printf("Layer (type)    Output    Params    IN         OUT\n");
    printf("===========================================================\n");
    for(i = 0; i < nn->denseLen; i++)                               //  Print all Dense layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->denselayers[i].name[j] != '\0')
          {
            printf("%c", nn->denselayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(Dns.) ");

        bufflen = sprintf(buffer, "%d", nn->denselayers[i].n);      //  Print output length
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }
                                                                    //  Print number of parameters
        bufflen = sprintf(buffer, "%d", (nn->denselayers[i].i + 1) * nn->denselayers[i].n);
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == DENSE_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == DENSE_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    for(i = 0; i < nn->convLen; i++)                                //  Print all Convolutional layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->convlayers[i].name[j] != '\0')
          {
            printf("%c", nn->convlayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(C2D.) ");
                                                                    //  Print output length
        bufflen = sprintf(buffer, "%d", outputLen_Conv2D(nn->convlayers + i));
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }
                                                                    //  Print number of parameters
        convparams = 0;
        for(k = 0; k < nn->convlayers[i].n; k++)
          convparams += nn->convlayers[i].filters[k].w * nn->convlayers[i].filters[k].h + 1;
        bufflen = sprintf(buffer, "%d", convparams);
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == CONV2D_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == CONV2D_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    for(i = 0; i < nn->accumLen; i++)                               //  Print all Accumulator layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->accumlayers[i].name[j] != '\0')
          {
            printf("%c", nn->accumlayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(Acc.) ");

        bufflen = sprintf(buffer, "%d", nn->accumlayers[i].i);      //  Print output length
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }
                                                                    //  Print number of parameters
        bufflen = sprintf(buffer, "%d", 0);
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == ACCUM_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == ACCUM_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    for(i = 0; i < nn->lstmLen; i++)                                //  Print all LSTM layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->lstmlayers[i].name[j] != '\0')
          {
            printf("%c", nn->lstmlayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(LSTM) ");

        bufflen = sprintf(buffer, "%d", nn->lstmlayers[i].h);       //  Print output length
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }
                                                                    //  Print number of parameters
        bufflen = sprintf(buffer, "%d", 4 * nn->lstmlayers[i].d * nn->lstmlayers[i].h +
                                        4 * nn->lstmlayers[i].h * nn->lstmlayers[i].h +
                                        4 * nn->lstmlayers[i].h);
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == LSTM_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == LSTM_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    for(i = 0; i < nn->gruLen; i++)                                 //  Print all GRU layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->grulayers[i].name[j] != '\0')
          {
            printf("%c", nn->grulayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(GRU)  ");

        bufflen = sprintf(buffer, "%d", nn->grulayers[i].h);        //  Print output length
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }
                                                                    //  Print number of parameters
        bufflen = sprintf(buffer, "%d", 3 * nn->grulayers[i].d * nn->grulayers[i].h +
                                        3 * nn->grulayers[i].h * nn->grulayers[i].h +
                                        3 * nn->grulayers[i].h);
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == GRU_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == GRU_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    for(i = 0; i < nn->poolLen; i++)                                //  Print all Pool layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->poollayers[i].name[j] != '\0')
          {
            printf("%c", nn->poollayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(Pool) ");

        bufflen = sprintf(buffer, "%d", nn->poollayers[i].outlen);  //  Print output length
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        bufflen = sprintf(buffer, "0");                             //  Print number of parameters
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == POOL_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == POOL_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    for(i = 0; i < nn->upresLen; i++)                               //  Print all Upres layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->upreslayers[i].name[j] != '\0')
          {
            printf("%c", nn->upreslayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(Upres)");
                                                                    //  Print output length
        bufflen = sprintf(buffer, "%d", outputLen_Upres(nn->upreslayers + i));
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        bufflen = sprintf(buffer, "0");                             //  Print number of parameters
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == UPRES_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == UPRES_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    for(i = 0; i < nn->normalLen; i++)                              //  Print all Normalization layers
      {
        firstInline = true;
        j = 0;
        while(j < 9 && nn->normlayers[i].name[j] != '\0')
          {
            printf("%c", nn->normlayers[i].name[j]);
            j++;
          }
        while(j < 9)
          {
            printf(" ");
            j++;
          }
        printf("(Norm)");
                                                                    //  Print output length
        bufflen = sprintf(buffer, "%d", nn->normlayers[i].i);
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        bufflen = sprintf(buffer, "4");                             //  Print number of parameters
        j = 0;
        while(j < 10 && j < bufflen)
          {
            printf("%c", buffer[j]);
            j++;
          }
        while(j < 10)
          {
            printf(" ");
            j++;
          }

        for(k = 0; k < nn->len; k++)                                //  Print inputs to this layer
          {
            if(nn->edgelist[k].dstType == NORMAL_ARRAY && nn->edgelist[k].dstIndex == i)
              {
                if(!firstInline)
                  printf("                                    ");
                printLayerName(nn->edgelist[k].srcType, nn->edgelist[k].srcIndex, nn);
                firstInline = false;
              }
          }
        for(k = 0; k < nn->len; k++)                                //  Print outputs from this layer
          {
            if(nn->edgelist[k].srcType == NORMAL_ARRAY && nn->edgelist[k].srcIndex == i)
              {
                printf("                                               ");
                printLayerName(nn->edgelist[k].dstType, nn->edgelist[k].dstIndex, nn);
              }
          }

        printf("\n");
      }
    printf("===========================================================\n");
    return;
  }

/* Called by print_NN() */
void printLayerName(unsigned char arr, unsigned int index, NeuralNet* nn)
  {
    unsigned char i;
    switch(arr)
      {
        case INPUT_ARRAY:  printf("NETWORK-IN");
                           break;
        case DENSE_ARRAY:  i = 0;
                           while(i < 9 && nn->denselayers[ arr ].name[i] != '\0')
                             {
                               printf("%c", nn->denselayers[ index ].name[i]);
                               i++;
                             }
                           break;
        case CONV2D_ARRAY: i = 0;
                           while(i < 9 && nn->convlayers[ index ].name[i] != '\0')
                             {
                               printf("%c", nn->convlayers[ index ].name[i]);
                               i++;
                             }
                           break;
        case ACCUM_ARRAY:  i = 0;
                           while(i < 9 && nn->accumlayers[ index ].name[i] != '\0')
                             {
                               printf("%c", nn->accumlayers[ index ].name[i]);
                               i++;
                             }
                           break;
        case LSTM_ARRAY:   i = 0;
                           while(i < 9 && nn->lstmlayers[ index ].name[i] != '\0')
                             {
                               printf("%c", nn->lstmlayers[ index ].name[i]);
                               i++;
                             }
                           break;
        case GRU_ARRAY:    i = 0;
                           while(i < 9 && nn->grulayers[ index ].name[i] != '\0')
                             {
                               printf("%c", nn->grulayers[ index ].name[i]);
                               i++;
                             }
                           break;
        case POOL_ARRAY:   i = 0;
                           while(i < 9 && nn->poollayers[ index ].name[i] != '\0')
                             {
                               printf("%c", nn->poollayers[ index ].name[i]);
                               i++;
                             }
                           break;
        case UPRES_ARRAY:  i = 0;
                           while(i < 9 && nn->upreslayers[ index ].name[i] != '\0')
                             {
                               printf("%c", nn->upreslayers[ index ].name[i]);
                               i++;
                             }
                           break;
        case NORMAL_ARRAY: i = 0;
                           while(i < 9 && nn->normlayers[ index ].name[i] != '\0')
                             {
                               printf("%c", nn->normlayers[ index ].name[i]);
                               i++;
                             }
                           break;
      }
    printf("\n");
    return;
  }

/**************************************************************************************************
 Dense-Layers  */

/* Add a layer to a network in progress.
   It shall have 'inputs' inputs and 'nodes' nodes. */
unsigned int add_Dense(unsigned int inputs, unsigned int nodes, NeuralNet* nn)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("add_Dense(%d, %d)\n", inputs, nodes);
    #endif

    nn->denseLen++;
    if(nn->denseLen == 1)                                           //  Expand the DenseLayer array
      {
        if((nn->denselayers = (DenseLayer*)malloc(sizeof(DenseLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate DenseLayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->denselayers = (DenseLayer*)realloc(nn->denselayers, nn->denseLen * sizeof(DenseLayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate DenseLayer array\n");
            exit(1);
          }
      }

    nn->denselayers[nn->denseLen - 1].i = inputs;                   //  Set this newest layer's inputs
    nn->denselayers[nn->denseLen - 1].n = nodes;                    //  Set this newest layer's number of nodes
                                                                    //  Allocate this newest layer's weight matrix
    if((nn->denselayers[nn->denseLen - 1].W = (double*)malloc((inputs + 1) * nodes * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate DenseLayer's weight array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's mask matrix
    if((nn->denselayers[nn->denseLen - 1].M = (double*)malloc((inputs + 1) * nodes * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate DenseLayer's mask array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's function-flag array
    if((nn->denselayers[nn->denseLen - 1].f = (unsigned char*)malloc(nodes * sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to allocate DenseLayer's function-flag array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's function-parameter array
    if((nn->denselayers[nn->denseLen - 1].alpha = (double*)malloc(nodes * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate DenseLayer's function-parameter array\n");
        exit(1);
      }
                                                                    //  Allocate output buffer
    if((nn->denselayers[nn->denseLen - 1].out = (double*)malloc(nodes * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate DenseLayer's internal output array\n");
        exit(1);
      }

    for(i = 0; i < (inputs + 1) * nodes; i++)                       //  Generate random numbers in [ -1.0, 1.0 ]
      {
        nn->denselayers[nn->denseLen - 1].W[i] = -1.0 + ((double)rand() / ((double)RAND_MAX * 0.5));
        nn->denselayers[nn->denseLen - 1].M[i] = 1.0;               //  All are UNmasked
      }
    for(i = 0; i < nodes; i++)                                      //  Default all to ReLU with parameter = 1.0
      {
        nn->denselayers[nn->denseLen - 1].f[i] = RELU;
        nn->denselayers[nn->denseLen - 1].alpha[i] = 1.0;
      }
    for(i = 0; i < nodes; i++)                                      //  Blank out 'out' array
      nn->denselayers[nn->denseLen - 1].out[i] = 0.0;
    for(i = 0; i < LAYER_NAME_LEN; i++)                             //  Blank out layer name
      nn->denselayers[nn->denseLen - 1].name[i] = '\0';

    return nn->denseLen;
  }

bool read_Dense(DenseLayer* denselayers, unsigned int denseLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;
    bool* boolBuffer = NULL;

    unsigned int len;
    unsigned int i, j;

    for(i = 0; i < denseLen; i++)
      {
                                                                    //  Allocate 2 unsigned ints
        if((uintBuffer = (unsigned int*)malloc(2 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer for reading from file\n");
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> 0
                                                                    //  boolBuffer   --> 0
        if(fread(uintBuffer, sizeof(int), 2, fp) != 2)              //  Read 2 objects of size int into buffer
          {
            printf("ERROR: Unable to read Dense Layer parameters into buffer\n");
            free(uintBuffer);
            return false;
          }

        denselayers[i].i = uintBuffer[0];                           //  Read number of inputs for DenseLayer[i] from buffer
        denselayers[i].n = uintBuffer[1];                           //  Read number of units for DenseLayer[i] from buffer
        #ifdef __NEURON_DEBUG
        printf("  denselayers[%d].i = %d\n", i, denselayers[i].i);
        printf("  denselayers[%d].n = %d\n", i, denselayers[i].n);
        #endif

        len = (denselayers[i].i + 1) * denselayers[i].n;
        if((denselayers[i].W = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array while reading DenseLayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((denselayers[i].M = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate mask array while reading DenseLayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((denselayers[i].out = (double*)malloc(denselayers[i].n * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate output array while reading DenseLayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        for(j = 0; j < denselayers[i].n; j++)                       //  Allocate and blank out output array
          denselayers[i].out[j] = 0.0;
        if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> (denselayers[i].i + 1) * denselayers[i].n
                                                                    //  boolBuffer   --> 0
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read Dense Layer weights into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read DenseLayer[i]'s weights from buffer
          denselayers[i].W[j] = doubleBuffer[j];                    //  in the order in which they exist in the file.
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  denselayers[%d].W[%d] = %.6f\n", i, j, denselayers[i].W[j]);
        #endif

        if((boolBuffer = (bool*)malloc(len * sizeof(bool))) == NULL)
          {
            printf("ERROR: Unable to allocate Boolean buffer while reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> (denselayers[i].i + 1) * denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
        if(fread(boolBuffer, sizeof(bool), len, fp) != len)         //  Read 'len' objects of size bool into buffer
          {
            printf("ERROR: Unable to read Dense Layer masks into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read DenseLayer[i]'s weights from buffer
          {
            if(boolBuffer[j])                                       //  True means UNMASKED, means w * 1.0
              denselayers[i].M[j] = 1.0;
            else                                                    //  False means MASKED, means w * 0.0
              denselayers[i].M[j] = 0.0;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  denselayers[%d].M[%d] = %.6f\n", i, j, denselayers[i].M[j]);
        #endif

        len = denselayers[i].n;
        if((denselayers[i].f = (unsigned char*)malloc(len * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate function flag array for DenseLayer[%d] while reading from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        if((ucharBuffer = (unsigned char*)malloc(len * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> denselayers[i].n
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> (denselayers[i].i + 1) * denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
        if(fread(ucharBuffer, sizeof(char), len, fp) != len)        //  Read 'len' objects of size char into buffer
          {
            printf("ERROR: Unable to read Dense Layer function flags into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read function flags
          denselayers[i].f[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  denselayers[%d].f[%d] = %d\n", i, j, denselayers[i].f[j]);
        #endif

        if((denselayers[i].alpha = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate function auxiliaries array for DenseLayer[%d] while reading from file\n", i);
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for reading from file\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(boolBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> denselayers[i].n
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read Dense Layer alphas into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read function auxiliaries
          denselayers[i].alpha[j] = doubleBuffer[j];
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  denselayers[%d].alpha[%d] = %.6f\n", i, j, denselayers[i].alpha[j]);
        #endif

        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
                                                                    //  Read LAYER_NAME_LEN objects of size char to buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read Dense Layer name into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Read layer name
          denselayers[i].name[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        printf("  denselayers[%d].name = %s\n\n", i, denselayers[i].name);
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
        free(boolBuffer);                                           //  boolBuffer   --> 0
      }

    return true;
  }

bool write_Dense(DenseLayer* denselayers, unsigned int denseLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;
    bool* boolBuffer = NULL;

    unsigned int len;
    unsigned int i, j;

    for(i = 0; i < denseLen; i++)
      {
                                                                    //  Allocate 2 unsigned ints
        if((uintBuffer = (unsigned int*)malloc(2 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer for writing to file\n");
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> 0
                                                                    //  boolBuffer   --> 0
        uintBuffer[0] = denselayers[i].i;                           //  Write number of inputs for DenseLayer[i] to buffer
        uintBuffer[1] = denselayers[i].n;                           //  Write number of units for DenseLayer[i] to buffer
        if(fwrite(uintBuffer, sizeof(int), 2, fp) != 2)             //  From buffer, write 2 objects of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  uintBuffer[0] = denselayers[%d].i = %d\n", i, uintBuffer[0]);
        printf("  uintBuffer[1] = denselayers[%d].n = %d\n", i, uintBuffer[1]);
        #endif

        len = (denselayers[i].i + 1) * denselayers[i].n;
        if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer while writing DenseLayer[%d] to file\n", i);
            free(uintBuffer);
            return false;
          }
        if((boolBuffer = (bool*)malloc(len * sizeof(bool))) == NULL)
          {
            printf("ERROR: Unable to allocate boolean buffer while writing DenseLayer[%d] to file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> (denselayers[i].i + 1) * denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
        for(j = 0; j < len; j++)                                    //  Copy weight array to buffer
          {
            doubleBuffer[j] = denselayers[i].W[j];
            if(denselayers[i].M[j] == 1.0)
              boolBuffer[j] = true;
            else
              boolBuffer[j] = false;
          }
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        if(fwrite(boolBuffer, sizeof(bool), len, fp) != len)        //  From buffer, write 'len' objects of size bool
          {
            printf("ERROR: Unable to write bool buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = denselayers[%d].W[%d] = %.6f\n", j, i, j, doubleBuffer[j]);
        for(j = 0; j < len; j++)
          {
            if(boolBuffer[j])
              printf("  boolBuffer[%d] = denselayers[%d].M[%d] = 1\n", j, i, j);
            else
              printf("  boolBuffer[%d] = denselayers[%d].M[%d] = 0\n", j, i, j);
          }
        #endif

        len = denselayers[i].n;
        if((ucharBuffer = (unsigned char*)malloc(len * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for writing to file\n");
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> denselayers[i].n
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> (denselayers[i].i + 1) * denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
        for(j = 0; j < len; j++)                                    //  Copy flag array to buffer
          ucharBuffer[j] = denselayers[i].f[j];
        if(fwrite(ucharBuffer, sizeof(char), len, fp) != len)       //  From buffer, write 'len' objects of size char
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  ucharBuffer[%d] = denselayers[%d].f[%d] = %d\n", j, i, j, ucharBuffer[j]);
        #endif

        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate double buffer for writing to file\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(boolBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> denselayers[i].n
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
        for(j = 0; j < len; j++)                                    //  Copy parameter array to buffer
          doubleBuffer[j] = denselayers[i].alpha[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = denselayers[%d].alpha[%d] = %.6f\n", j, i, j, doubleBuffer[j]);
        #endif

        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to re-allocate unsigned char buffer for writing to file\n");
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 2
                                                                    //  doubleBuffer --> denselayers[i].n
                                                                    //  boolBuffer   --> (denselayers[i].i + 1) * denselayers[i].n
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Copy layer name to buffer
          ucharBuffer[j] = denselayers[i].name[j];
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            free(boolBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  ucharBuffer = denselayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
        free(boolBuffer);                                           //  boolBuffer   --> 0
      }

    return true;
  }

/**************************************************************************************************
 2D-Convolutional-Layers  */

/* Add a Conv2DLayer to a network in progress.
   It shall have an 'inputW' by 'inputH' input matrix.
   Note that this function DOES NOT, itself, allocate any filters! */
unsigned int add_Conv2D(unsigned int inputW, unsigned int inputH, NeuralNet* nn)
  {
    unsigned char i;

    #ifdef __NEURON_DEBUG
    printf("add_Conv2D(%d, %d)\n", inputW, inputH);
    #endif

    nn->convLen++;
    if(nn->convLen == 1)                                            //  Expand the Conv2DLayer array
      {
        if((nn->convlayers = (Conv2DLayer*)malloc(sizeof(Conv2DLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->convlayers = (Conv2DLayer*)realloc(nn->convlayers, nn->convLen * sizeof(Conv2DLayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Conv2DLayer array\n");
            exit(1);
          }
      }

    nn->convlayers[nn->convLen - 1].inputW = inputW;                //  Set this newest layer's input dimensions
    nn->convlayers[nn->convLen - 1].inputH = inputH;
    nn->convlayers[nn->convLen - 1].n = 0;                          //  New layer initially contains zero filters
    nn->convlayers[nn->convLen - 1].outlen = 0;                     //  An empty layer has zero output
    for(i = 0; i < LAYER_NAME_LEN; i++)                             //  Blank out layer name
      nn->convlayers[nn->convLen - 1].name[i] = '\0';

    return nn->convLen;
  }

bool read_Conv2D(Conv2DLayer* convlayers, unsigned int convLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int len;
    unsigned int i, j, k;

    for(i = 0; i < convLen; i++)
      {
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int array while reading from file\n");
            return false;
          }
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> 0
        if(fread(uintBuffer, sizeof(int), 3, fp) != 3)              //  Read 3 objects of size int into buffer
          {
            printf("ERROR: Unable to read Convolution Layer parameters into buffer");
            free(uintBuffer);
            free(ucharBuffer);
            return false;
          }
        convlayers[i].inputW = uintBuffer[0];                       //  Read Conv2DLayer input width from buffer
        convlayers[i].inputH = uintBuffer[1];                       //  Read Conv2DLayer input height from buffer
        convlayers[i].n = uintBuffer[2];                            //  Read number of Conv2DLayer filters from buffer
        #ifdef __NEURON_DEBUG
        printf("  convlayers[%d].inputW = %d\n", i, convlayers[i].inputW);
        printf("  convlayers[%d].inputH = %d\n", i, convlayers[i].inputH);
        printf("  convlayers[%d].n      = %d\n", i, convlayers[i].n);
        #endif
                                                                    //  Read LAYER_NAME_LEN objects of size char into buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read Convolution Layer name into buffer");
            free(uintBuffer);
            free(ucharBuffer);
            return false;
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Read layer name
          convlayers[i].name[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        printf("  convlayers[%d].name = %s\n", i, convlayers[i].name);
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
                                                                    //  doubleBuffer --> 0

                                                                    //  Allocate 'n' filters
        if((convlayers[i].filters = (Filter2D*)malloc(convlayers[i].n * sizeof(Filter2D))) == NULL)
          {
            printf("ERROR: Unable to allocate filter array for Conv2DLayer[%d] while reading from file\n", i);
            return false;
          }
        for(j = 0; j < convlayers[i].n; j++)                        //  Fill in details of each filter in this layer
          {
                                                                    //  Filters need to read 4 ints (w, h, stride_h, stride_v);
                                                                    //                       1 uchar (f);
                                                                    //                       1 double (alpha);
                                                                    //                       w * h + 1 doubles (W)
            if((ucharBuffer = (unsigned char*)malloc(sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned char buffer while reading from file\n");
                return false;
              }
            if((uintBuffer = (unsigned int*)malloc(4 * sizeof(int))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
                free(ucharBuffer);
                return false;
              }
            if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate double buffer while reading from file\n");
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4
                                                                    //  doubleBuffer --> 1
            if(fread(uintBuffer, sizeof(int), 4, fp) != 4)          //  Read 4 objects of size int into buffer
              {
                printf("ERROR: Unable to read dimensions and strides for Convolution Layer filter[%d] into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            convlayers[i].filters[j].w = uintBuffer[0];             //  Read dimensions and strides
            convlayers[i].filters[j].h = uintBuffer[1];
            convlayers[i].filters[j].stride_h = uintBuffer[2];
            convlayers[i].filters[j].stride_v = uintBuffer[3];
            #ifdef __NEURON_DEBUG
            printf("  convlayers[%d].filters[%d].w = %d\n", i, j, convlayers[i].filters[j].w);
            printf("  convlayers[%d].filters[%d].h = %d\n", i, j, convlayers[i].filters[j].h);
            printf("  convlayers[%d].filters[%d].stride_h = %d\n", i, j, convlayers[i].filters[j].stride_h);
            printf("  convlayers[%d].filters[%d].stride_v = %d\n", i, j, convlayers[i].filters[j].stride_v);
            #endif

            if(fread(ucharBuffer, sizeof(char), 1, fp) != 1)        //  Read 1 object of size char into buffer
              {
                printf("ERROR: Unable to read activation function flag for Convolution Layer filter[%d] into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            convlayers[i].filters[j].f = ucharBuffer[0];
            #ifdef __NEURON_DEBUG
            printf("  convlayers[%d].filters[%d].f = %d\n", i, j, convlayers[i].filters[j].f);
            #endif

            if(fread(doubleBuffer, sizeof(double), 1, fp) != 1)     //  Read 1 object of size double into buffer
              {
                printf("ERROR: Unable to read activation function parameter for Convolution Layer filter[%d] into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            convlayers[i].filters[j].alpha = doubleBuffer[0];
            #ifdef __NEURON_DEBUG
            printf("  convlayers[%d].filters[%d].alpha = %.6f\n", i, j, convlayers[i].filters[j].alpha);
            #endif

            free(doubleBuffer);                                     //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4
                                                                    //  doubleBuffer --> 0

            len = convlayers[i].filters[j].w * convlayers[i].filters[j].h + 1;
            if((convlayers[i].filters[j].W = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate filter[%d] of Conv2DLayer[%d] while reading from file\n", j, i);
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate double buffer for reading from file\n");
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4
                                                                    //  doubleBuffer --> convlayers[i].filters[j].w * convlayers[i].filters[j].h + 1
            if(fread(doubleBuffer, sizeof(double), len, fp) != len) //  Read len objects of size double into buffer
              {
                printf("ERROR: Unable to read weights for Convolution Layer filter %d into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            for(k = 0; k < len; k++)                                //  Read Filter2D weights and bias
              convlayers[i].filters[j].W[k] = doubleBuffer[k];
            #ifdef __NEURON_DEBUG
            printf("  ");
            for(k = 0; k < len; k++)
              printf("  %.3f", convlayers[i].filters[j].W[k]);
            printf("\n");
            #endif

            free(ucharBuffer);                                      //  ucharBuffer  --> 0
            free(uintBuffer);                                       //  uintBuffer   --> 0
            free(doubleBuffer);                                     //  doubleBuffer --> 0
          }

        convlayers[i].outlen = outputLen_Conv2D(convlayers + i);    //  Compute layer's output length; allocate and blank its output buffer
        if((convlayers[i].out = (double*)malloc(convlayers[i].outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate output array while reading Conv2DLayer[%d] from file\n", i);
            return false;
          }
        for(j = 0; j < convlayers[i].outlen; j++)                   //  Allocate and blank out output array
          convlayers[i].out[j] = 0.0;
      }

    return true;
  }

bool write_Conv2D(Conv2DLayer* convlayers, unsigned int convLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int len;
    unsigned int i, j, k;

    for(i = 0; i < convLen; i++)
      {
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int array while reading from file\n");
            return false;
          }
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> 0
        uintBuffer[0] = convlayers[i].inputW;                       //  Write the input dimensions for Conv2DLayer[i] to buffer
        uintBuffer[1] = convlayers[i].inputH;
        uintBuffer[2] = convlayers[i].n;
        if(fwrite(uintBuffer, sizeof(int), 3, fp) != 3)             //  From buffer, write 3 objects of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  convlayers[%d].inputW = %d\n", i, uintBuffer[0]);
        printf("  convlayers[%d].inputH = %d\n", i, uintBuffer[1]);
        printf("  convlayers[%d].n = %d\n", i, uintBuffer[2]);
        #endif

        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Write layer name to the buffer
          ucharBuffer[j] = convlayers[i].name[j];
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  convlayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
                                                                    //  doubleBuffer --> 0

        for(j = 0; j < convlayers[i].n; j++)                        //  Fill in details of each filter in this layer
          {
                                                                    //  Filters need to read 4 ints (w, h, stride_h, stride_v);
                                                                    //                       1 uchar (f);
                                                                    //                       1 double (alpha);
                                                                    //                       w * h + 1 doubles (W)
            if((ucharBuffer = (unsigned char*)malloc(sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned char buffer while reading from file\n");
                return false;
              }
            if((uintBuffer = (unsigned int*)malloc(4 * sizeof(int))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
                return false;
              }
            if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate double buffer while reading from file\n");
                return false;
              }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4
                                                                    //  doubleBuffer --> 1
            uintBuffer[0] = convlayers[i].filters[j].w;             //  Write dimensions and strides to buffer
            uintBuffer[1] = convlayers[i].filters[j].h;
            uintBuffer[2] = convlayers[i].filters[j].stride_h;
            uintBuffer[3] = convlayers[i].filters[j].stride_v;
            if(fwrite(uintBuffer, sizeof(int), 4, fp) != 4)         //  From buffer, write 4 objects of size int
              {
                printf("ERROR: Unable to write unsigned int buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            #ifdef __NEURON_DEBUG
            printf("  convlayers[%d].filters[%d].w = %d\n", i, j, uintBuffer[0]);
            printf("  convlayers[%d].filters[%d].h = %d\n", i, j, uintBuffer[1]);
            printf("  convlayers[%d].filters[%d].stride_h = %d\n", i, j, uintBuffer[2]);
            printf("  convlayers[%d].filters[%d].stride_v = %d\n", i, j, uintBuffer[3]);
            #endif

            ucharBuffer[0] = convlayers[i].filters[j].f;            //  Write filter function flag to buffer
            if(fwrite(ucharBuffer, sizeof(char), 1, fp) != 1)       //  From buffer, write 1 object of size char
              {
                printf("ERROR: Unable to write unsigned char buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            #ifdef __NEURON_DEBUG
            printf("  convlayers[%d].filters[%d].f = %d\n", i, j, ucharBuffer[0]);
            #endif

            doubleBuffer[0] = convlayers[i].filters[j].alpha;       //  Write filter function parameter to buffer
            if(fwrite(doubleBuffer, sizeof(double), 1, fp) != 1)    //  From buffer, write 1 object of size double
              {
                printf("ERROR: Unable to write double buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            #ifdef __NEURON_DEBUG
            printf("  convlayers[%d].filters[%d].alpha = %.6f\n", i, j, doubleBuffer[0]);
            #endif

            free(doubleBuffer);                                     //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4
                                                                    //  doubleBuffer --> 0

            len = convlayers[i].filters[j].w * convlayers[i].filters[j].h + 1;
            if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate double buffer for reading from file\n");
                return false;
              }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4
                                                                    //  doubleBuffer --> convlayers[i].filters[j].w * convlayers[i].filters[j].h + 1
            for(k = 0; k < len; k++)                                //  Read Filter2D weights and bias
              doubleBuffer[k] = convlayers[i].filters[j].W[k];
            if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)//  From buffer, write 'len' objects of size double
              {
                printf("ERROR: Unable to write double buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                free(doubleBuffer);
                return false;
              }
            #ifdef __NEURON_DEBUG
            printf("  ");
            for(k = 0; k < len; k++)
              printf("  %.3f", doubleBuffer[k]);
            printf("\n");
            #endif

            free(ucharBuffer);                                      //  ucharBuffer  --> 0
            free(uintBuffer);                                       //  uintBuffer   --> 0
            free(doubleBuffer);                                     //  doubleBuffer --> 0
          }
      }

    return true;
  }

/**************************************************************************************************
 Accumulator-Layers  */

unsigned int add_Accum(unsigned int inputs, NeuralNet* nn)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("add_Accum(%d)\n", inputs);
    #endif

    nn->accumLen++;

    if(nn->accumLen == 1)                                           //  Expand the AccumLayer array
      {
        if((nn->accumlayers = (AccumLayer*)malloc(sizeof(AccumLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate AccumLayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->accumlayers = (AccumLayer*)realloc(nn->accumlayers, nn->accumLen * sizeof(AccumLayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate AccumLayer array\n");
            exit(1);
          }
      }
    nn->accumlayers[nn->accumLen - 1].i = inputs;
                                                                    //  Allocate output buffer
    if((nn->accumlayers[nn->accumLen - 1].out = (double*)malloc(inputs * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate AccumLayer's internal output array\n");
        exit(1);
      }
    for(i = 0; i < inputs; i++)                                     //  Blank out 'out' array
      nn->accumlayers[nn->accumLen - 1].out[i] = 0.0;
    for(i = 0; i < LAYER_NAME_LEN; i++)                             //  Blank out layer name
      nn->accumlayers[nn->accumLen - 1].name[i] = '\0';

    return nn->accumLen;
  }

bool read_Accum(AccumLayer* accumlayers, unsigned int accumLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;

    unsigned int i, j;

    for(i = 0; i < accumLen; i++)
      {
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            exit(1);
          }
        if((uintBuffer = (unsigned int*)malloc(sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer for reading from file\n");
            exit(1);
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 1
                                                                    //  doubleBuffer --> 0
                                                                    //  boolBuffer   --> 0
        if(fread(uintBuffer, sizeof(int), 1, fp) != 1)              //  Read 1 object of size int into buffer
          {
            printf("ERROR: Unable to read Accumulation Layer parameter into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        accumlayers[i].i = uintBuffer[0];                           //  Read number of Accumulator inputs
        #ifdef __NEURON_DEBUG
        printf("  accumlayers[%d].i = %d\n", i, accumlayers[i].i);
        #endif
                                                                    //  Read LAYER_NAME_LEN objects of size char into buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read Accumulation Layer name into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Read Accumulator layer name
          accumlayers[i].name[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        printf("  accumlayers[%d].name = %s\n", i, accumlayers[i].name);
        #endif

        if((accumlayers[i].out = (double*)malloc(accumlayers[i].i * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate output array while reading AccumLayer[%d] from file\n", i);
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        for(j = 0; j < accumlayers[i].i; j++)                       //  Allocate and blank out output array
          accumlayers[i].out[j] = 0.0;

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
      }

    return true;
  }

bool write_Accum(AccumLayer* accumlayers, unsigned int accumLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;

    unsigned int i, j;

    for(i = 0; i < accumLen; i++)
      {
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            return false;
          }
        if((uintBuffer = (unsigned int*)malloc(sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer for reading from file\n");
            free(ucharBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 1
                                                                    //  doubleBuffer --> 0
                                                                    //  boolBuffer   --> 0
        uintBuffer[0] = accumlayers[i].i;                           //  Copy number of Accumulator inputs to buffer
        if(fwrite(uintBuffer, sizeof(int), 1, fp) != 1)             //  From buffer, write 1 object of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  accumlayers[%d].i = %d\n", i, uintBuffer[0]);
        #endif

        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Write layer name to the buffer
          ucharBuffer[j] = accumlayers[i].name[j];
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  accumlayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
      }

    return true;
  }

/**************************************************************************************************
 LSTM-Layers  */

unsigned int add_LSTM(unsigned int dimInput, unsigned int dimState, unsigned int cacheSize, NeuralNet* nn)
  {
    unsigned int i;
    double xavier;

    #ifdef __NEURON_DEBUG
    printf("add_LSTM(%d, %d)\n", dimInput, dimState);
    #endif

    nn->lstmLen++;
    if(nn->lstmLen == 1)                                            //  Expand the LSTMLayer array
      {
        if((nn->lstmlayers = (LSTMLayer*)malloc(sizeof(LSTMLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate LSTMLayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->lstmlayers = (LSTMLayer*)realloc(nn->lstmlayers, nn->lstmLen * sizeof(LSTMLayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate LSTMLayer array\n");
            exit(1);
          }
      }

    nn->lstmlayers[nn->lstmLen - 1].d = dimInput;                   //  Save dimensionality of input
    nn->lstmlayers[nn->lstmLen - 1].h = dimState;                   //  Save dimensionality of states
    nn->lstmlayers[nn->lstmLen - 1].cache = cacheSize;              //  Save the cache size
    nn->lstmlayers[nn->lstmLen - 1].t = 0;                          //  Initial time step = 0

                                                                    //  Allocate this newest layer's W matrices
    if((nn->lstmlayers[nn->lstmLen - 1].Wi = (double*)malloc(dimInput * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Wi weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].Wo = (double*)malloc(dimInput * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Wo weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].Wf = (double*)malloc(dimInput * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Wf weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].Wc = (double*)malloc(dimInput * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Wc weight array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's U matrices
    if((nn->lstmlayers[nn->lstmLen - 1].Ui = (double*)malloc(dimState * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Ui weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].Uo = (double*)malloc(dimState * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Uo weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].Uf = (double*)malloc(dimState * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Uf weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].Uc = (double*)malloc(dimState * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's Uc weight array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's b vectors
    if((nn->lstmlayers[nn->lstmLen - 1].bi = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's bi weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].bo = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's bo weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].bf = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's bf weight array\n");
        exit(1);
      }
    if((nn->lstmlayers[nn->lstmLen - 1].bc = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's bc weight array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's c vector
    if((nn->lstmlayers[nn->lstmLen - 1].c = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's c array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's cache matrix
    if((nn->lstmlayers[nn->lstmLen - 1].H = (double*)malloc(cacheSize * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate LSTMLayer's state cache matrix\n");
        exit(1);
      }

    xavier = sqrt(6 / (dimInput + dimState));
    for(i = 0; i < dimInput * dimState; i++)                        //  Xavier-initialize W matrices
      {
                                                                    //  Generate random numbers in [ -xavier, xavier ]
        nn->lstmlayers[nn->lstmLen - 1].Wi[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->lstmlayers[nn->lstmLen - 1].Wo[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->lstmlayers[nn->lstmLen - 1].Wf[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->lstmlayers[nn->lstmLen - 1].Wc[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
      }
    xavier = sqrt(6 / (dimState + dimState));
    for(i = 0; i < dimState * dimState; i++)                        //  Xavier-initialize U matrices
      {
        nn->lstmlayers[nn->lstmLen - 1].Ui[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->lstmlayers[nn->lstmLen - 1].Uo[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->lstmlayers[nn->lstmLen - 1].Uf[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->lstmlayers[nn->lstmLen - 1].Uc[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
      }
    for(i = 0; i < dimState; i++)                                   //  Initialize b vectors
      {
        nn->lstmlayers[nn->lstmLen - 1].bi[i] = 0.0;
        nn->lstmlayers[nn->lstmLen - 1].bo[i] = 0.0;
        nn->lstmlayers[nn->lstmLen - 1].bf[i] = 1.0;                //  Forget gate set to ones
        nn->lstmlayers[nn->lstmLen - 1].bc[i] = 0.0;
      }
    for(i = 0; i < cacheSize * dimState; i++)                       //  Blank out the cache
      nn->lstmlayers[nn->lstmLen - 1].H[i] = 0.0;
    for(i = 0; i < dimState; i++)                                   //  Blank out the carry
      nn->lstmlayers[nn->lstmLen - 1].c[i] = 0.0;

    return nn->lstmLen;
  }

bool read_LSTM(LSTMLayer* lstmlayers, unsigned int lstmLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int len;
    unsigned int i, j;

    for(i = 0; i < lstmLen; i++)
      {
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> 0
                                                                    //  boolBuffer   --> 0
        if(fread(uintBuffer, sizeof(int), 3, fp) != 3)              //  Read 3 objects of size int into buffer
          {
            printf("ERROR: Unable to read LSTM parameters into buffer\n");
            free(uintBuffer);
            return false;
          }
        lstmlayers[i].d = uintBuffer[0];                            //  Read input dimensionality for LSTMLayer[i] from buffer
        lstmlayers[i].h = uintBuffer[1];                            //  Read state dimensionality for LSTMLayer[i] from buffer
        lstmlayers[i].cache = uintBuffer[2];                        //  Read state cache size for LSTMLayer[i] from buffer
        lstmlayers[i].t = 0;                                        //  Initialize time step to 0
        #ifdef __NEURON_DEBUG
        printf("  lstmlayers[%d].d = %d\n", i, lstmlayers[i].d);
        printf("  lstmlayers[%d].h = %d\n", i, lstmlayers[i].h);
        printf("  lstmlayers[%d].cache = %d\n", i, lstmlayers[i].cache);
        #endif

        len = lstmlayers[i].d * lstmlayers[i].h;                    //  Allocate all things d*h
        if((lstmlayers[i].Wi = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Wi while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((lstmlayers[i].Wo = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Wo while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((lstmlayers[i].Wf = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Wf while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((lstmlayers[i].Wc = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Wc while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> lstmlayers[i].d * lstmlayers[i].h
                                                                    //  boolBuffer   --> 0
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Wi into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Wi weights from buffer
          lstmlayers[i].Wi[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Wo into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Wo weights from buffer
          lstmlayers[i].Wo[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Wf into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Wf weights from buffer
          lstmlayers[i].Wf[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Wc into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Wc weights from buffer
          lstmlayers[i].Wc[j] = doubleBuffer[j];

        len = lstmlayers[i].h * lstmlayers[i].h;                    //  Allocate all things h*h
        if((lstmlayers[i].Ui = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Ui while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((lstmlayers[i].Uo = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Uo while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((lstmlayers[i].Uf = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Uf while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((lstmlayers[i].Uc = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Uc while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> lstmlayers[i].h * lstmlayers[i].h
                                                                    //  boolBuffer   --> 0
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Ui into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Ui weights from buffer
          lstmlayers[i].Ui[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Uo into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Uo weights from buffer
          lstmlayers[i].Uo[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Uf into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Uf weights from buffer
          lstmlayers[i].Uf[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM Uc into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s Uc weights from buffer
          lstmlayers[i].Uc[j] = doubleBuffer[j];

        len = lstmlayers[i].h;                                      //  Allocate all things h
        if((lstmlayers[i].bi = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate bias array bi while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((lstmlayers[i].bo = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate bias array bo while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((lstmlayers[i].bf = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate bias array bf while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((lstmlayers[i].bc = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate bias array bc while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((lstmlayers[i].c = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate cell array while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> lstmlayers[i].h
                                                                    //  boolBuffer   --> 0
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM bi into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s bi bias from buffer
          lstmlayers[i].bi[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM bo into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s bo bias from buffer
          lstmlayers[i].bo[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM bf into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s bf bias from buffer
          lstmlayers[i].bf[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read LSTM bc into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read LSTMLayer[i]'s bc bias from buffer
          lstmlayers[i].bc[j] = doubleBuffer[j];
        for(j = 0; j < len; j++)                                    //  Set vector c to zero-vector
          lstmlayers[i].c[j] = 0.0;

        len = lstmlayers[i].h * lstmlayers[i].cache;                //  Allocate the output/state cache
        if((lstmlayers[i].H = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate state cache while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                //  Blank out output matrix
          lstmlayers[i].H[j] = 0.0;

        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> lstmlayers[i].h
                                                                    //  Read LAYER_NAME_LEN objects of size char into buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read LSTM Layer name into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Read layer name
          lstmlayers[i].name[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        printf("  lstmlayers[%d].name = %s\n", i, lstmlayers[i].name);
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
      }

    return true;
  }

bool write_LSTM(LSTMLayer* lstmlayers, unsigned int lstmLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int len;
    unsigned int i, j;

    for(i = 0; i < lstmLen; i++)
      {
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> 0
        uintBuffer[0] = lstmlayers[i].d;                            //  Copy input dimensionality for LSTMLayer[i] to buffer
        uintBuffer[1] = lstmlayers[i].h;                            //  Copy state dimensionality for LSTMLayer[i] to buffer
        uintBuffer[2] = lstmlayers[i].cache;                        //  Copy state cache size for LSTMLayer[i] to buffer
        if(fwrite(uintBuffer, sizeof(int), 3, fp) != 3)             //  From buffer, write 3 objects of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  lstmlayers[%d].d = %d\n", i, uintBuffer[0]);
        printf("  lstmlayers[%d].h = %d\n", i, uintBuffer[1]);
        printf("  lstmlayers[%d].cache = %d\n", i, uintBuffer[2]);
        #endif

        len = lstmlayers[i].d * lstmlayers[i].h;                    //  Allocate all things d*h
        if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> lstmlayers[i].d * lstmlayers[i].h
        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Wi[j];                    //  Copy Wi to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Wi[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Wo[j];                    //  Copy Wo to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Wo[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Wf[j];                    //  Copy Wf to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Wf[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Wc[j];                    //  Copy Wc to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Wc[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        len = lstmlayers[i].h * lstmlayers[i].h;                    //  Allocate all things h*h
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> lstmlayers[i].h * lstmlayers[i].h
        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Ui[j];                    //  Copy Ui to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Ui[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Uo[j];                    //  Copy Uo to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Uo[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Uf[j];                    //  Copy Uf to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Uf[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].Uc[j];                    //  Copy Uc to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].Uc[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        len = lstmlayers[i].h;                                      //  Allocate all things h
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate double buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> lstmlayers[i].h
                                                                    //  boolBuffer   --> 0
        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].bi[j];                    //  Copy bi to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].bi[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].bo[j];                    //  Copy bo to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].bo[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].bf[j];                    //  Copy bf to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].bf[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = lstmlayers[i].bc[j];                    //  Copy bc to buffer
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  lstmlayers[%d].bc[%d] = %.4f\n", i, j, doubleBuffer[j]);
        #endif

        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> nn->lstmlayers[i].h
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Write layer name to the buffer
          ucharBuffer[j] = lstmlayers[i].name[j];
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  lstmlayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
      }

    return true;
  }

/**************************************************************************************************
 GRU-Layers  */

unsigned int add_GRU(unsigned int dimInput, unsigned int dimState, unsigned int cacheSize, NeuralNet* nn)
  {
    unsigned int i;
    double xavier;

    #ifdef __NEURON_DEBUG
    printf("add_GRU(%d, %d)\n", dimInput, dimState);
    #endif

    nn->gruLen++;
    if(nn->gruLen == 1)                                             //  Expand the GRULayer array
      {
        if((nn->grulayers = (GRULayer*)malloc(sizeof(GRULayer))) == NULL)
          {
            printf("ERROR: Unable to allocate GRULayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->grulayers = (GRULayer*)realloc(nn->grulayers, nn->gruLen * sizeof(GRULayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate GRULayer array\n");
            exit(1);
          }
      }

    nn->grulayers[nn->gruLen - 1].d = dimInput;                     //  Save dimensionality of input
    nn->grulayers[nn->gruLen - 1].h = dimState;                     //  Save dimensionality of states
    nn->grulayers[nn->gruLen - 1].cache = cacheSize;                //  Save the cache size
    nn->grulayers[nn->gruLen - 1].t = 0;                            //  Initial time step = 0

                                                                    //  Allocate this newest layer's W matrices
    if((nn->grulayers[nn->gruLen - 1].Wz = (double*)malloc(dimInput * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's Wz weight array\n");
        exit(1);
      }
    if((nn->grulayers[nn->gruLen - 1].Wr = (double*)malloc(dimInput * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's Wr weight array\n");
        exit(1);
      }
    if((nn->grulayers[nn->gruLen - 1].Wh = (double*)malloc(dimInput * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's Wh weight array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's U matrices
    if((nn->grulayers[nn->gruLen - 1].Uz = (double*)malloc(dimState * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's Uz weight array\n");
        exit(1);
      }
    if((nn->grulayers[nn->gruLen - 1].Ur = (double*)malloc(dimState * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's Ur weight array\n");
        exit(1);
      }
    if((nn->grulayers[nn->gruLen - 1].Uh = (double*)malloc(dimState * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's Uh weight array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's b vectors
    if((nn->grulayers[nn->gruLen - 1].bz = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's bz weight array\n");
        exit(1);
      }
    if((nn->grulayers[nn->gruLen - 1].br = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's br weight array\n");
        exit(1);
      }
    if((nn->grulayers[nn->gruLen - 1].bh = (double*)malloc(dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's bh weight array\n");
        exit(1);
      }
                                                                    //  Allocate this newest layer's cache matrix
    if((nn->grulayers[nn->gruLen - 1].H = (double*)malloc(cacheSize * dimState * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate GRULayer's state cache matrix\n");
        exit(1);
      }

    xavier = sqrt(6 / (dimInput + dimState));
    for(i = 0; i < dimInput * dimState; i++)                        //  Xavier-initialize W matrices
      {
                                                                    //  Generate random numbers in [ -xavier, xavier ]
        nn->grulayers[nn->gruLen - 1].Wz[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->grulayers[nn->gruLen - 1].Wr[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->grulayers[nn->gruLen - 1].Wh[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
      }
    xavier = sqrt(6 / (dimState + dimState));
    for(i = 0; i < dimState * dimState; i++)                        //  Xavier-initialize U matrices
      {
        nn->grulayers[nn->gruLen - 1].Uz[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->grulayers[nn->gruLen - 1].Ur[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
        nn->grulayers[nn->gruLen - 1].Uh[i] = -xavier + ((double)rand() * 2.0 * xavier / ((double)RAND_MAX));
      }
    for(i = 0; i < dimState; i++)                                   //  Initialize b vectors
      {
        nn->grulayers[nn->gruLen - 1].bz[i] = 0.0;
        nn->grulayers[nn->gruLen - 1].br[i] = 0.0;
        nn->grulayers[nn->gruLen - 1].bh[i] = 0.0;
      }
    for(i = 0; i < cacheSize * dimState; i++)                       //  Blank out the cache
      nn->grulayers[nn->gruLen - 1].H[i] = 0.0;

    return nn->gruLen;
  }

bool read_GRU(GRULayer* grulayers, unsigned int gruLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int len;
    unsigned int i, j;

    for(i = 0; i < gruLen; i++)
      {
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> 0
                                                                    //  boolBuffer   --> 0
        if(fread(uintBuffer, sizeof(int), 3, fp) != 3)              //  Read 3 objects of size int into buffer
          {
            printf("ERROR: Unable to read LSTM parameters into buffer\n");
            free(uintBuffer);
            return false;
          }
        grulayers[i].d = uintBuffer[0];                             //  Read input dimensionality for LSTMLayer[i] from buffer
        grulayers[i].h = uintBuffer[1];                             //  Read state dimensionality for LSTMLayer[i] from buffer
        grulayers[i].cache = uintBuffer[2];                         //  Read state cache size for LSTMLayer[i] from buffer
        grulayers[i].t = 0;                                         //  Initialize time step to 0
        #ifdef __NEURON_DEBUG
        printf("  grulayers[%d].d = %d\n", i, grulayers[i].d);
        printf("  grulayers[%d].h = %d\n", i, grulayers[i].h);
        printf("  grulayers[%d].cache = %d\n", i, grulayers[i].cache);
        #endif

        len = grulayers[i].d * grulayers[i].h;                      //  Allocate all things d*h
        if((grulayers[i].Wz = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Wz while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((grulayers[i].Wr = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Wr while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((grulayers[i].Wh = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Wh while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            return false;
          }
        if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].d * grulayers[i].h
                                                                    //  boolBuffer   --> 0
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU Wz into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s Wz weights from buffer
          grulayers[i].Wz[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU Wr into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s Wr weights from buffer
          grulayers[i].Wr[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU Wz into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s Wh weights from buffer
          grulayers[i].Wh[j] = doubleBuffer[j];

        len = grulayers[i].h * grulayers[i].h;                      //  Allocate all things h*h
        if((grulayers[i].Uz = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Uz while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((grulayers[i].Ur = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Ur while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((grulayers[i].Uh = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate weight array Uh while reading LSTMLayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].h * grulayers[i].h
                                                                    //  boolBuffer   --> 0
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU Uz into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s Uz weights from buffer
          grulayers[i].Uz[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU Ur into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s Ur weights from buffer
          grulayers[i].Ur[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU Uh into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s Uh weights from buffer
          grulayers[i].Uh[j] = doubleBuffer[j];

        len = grulayers[i].h;                                       //  Allocate all things h
        if((grulayers[i].bz = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate bias array bz while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((grulayers[i].br = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate bias array br while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((grulayers[i].bh = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate bias array bh while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].h
                                                                    //  boolBuffer   --> 0
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU bz into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s bz bias from buffer
          grulayers[i].bz[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU br into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s br bias from buffer
          grulayers[i].br[j] = doubleBuffer[j];
        if(fread(doubleBuffer, sizeof(double), len, fp) != len)     //  Read 'len' objects of size double into buffer
          {
            printf("ERROR: Unable to read GRU bh into buffer\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Read GRULayer[i]'s bh bias from buffer
          grulayers[i].bh[j] = doubleBuffer[j];

        len = grulayers[i].h * grulayers[i].cache;                  //  Allocate the output/state cache
        if((grulayers[i].H = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate state cache while reading GRULayer[%d] from file\n", i);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < len; j++)                                    //  Blank out output matrix
          grulayers[i].H[j] = 0.0;

        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].h
                                                                    //  boolBuffer   --> 0
                                                                    //  Read LAYER_NAME_LEN objects of size char into buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read GRU Layer name into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Read Accumulator layer name
          grulayers[i].name[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        printf("  grulayers[%d].name = %s\n", i, grulayers[i].name);
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
      }

    return true;
  }

bool write_GRU(GRULayer* grulayers, unsigned int gruLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int len;
    unsigned int i, j;

    for(i = 0; i < gruLen; i++)
      {
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while weiting to file\n");
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> 0
        uintBuffer[0] = grulayers[i].d;                             //  Copy input dimensionality for LSTMLayer[i] to buffer
        uintBuffer[1] = grulayers[i].h;                             //  Copy state dimensionality for LSTMLayer[i] to buffer
        uintBuffer[2] = grulayers[i].cache;                         //  Copy state cache size for LSTMLayer[i] to buffer
        if(fwrite(uintBuffer, sizeof(int), 3, fp) != 3)             //  From buffer, write 3 objects of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  grulayers[%d].d = %d\n", i, uintBuffer[0]);
        printf("  grulayers[%d].h = %d\n", i, uintBuffer[1]);
        printf("  grulayers[%d].cache = %d\n", i, uintBuffer[2]);
        #endif

        len = grulayers[i].d * grulayers[i].h;                      //  Allocate all things d*h
        if((doubleBuffer = (double*)malloc(len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].d * grulayers[i].h
        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].Wz[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].Wz[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].Wr[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].Wr[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].Wh[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].Wh[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        len = grulayers[i].h * grulayers[i].h;                      //  Allocate all things h*h
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].h * grulayers[i].h
        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].Uz[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].Uz[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].Ur[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].Ur[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].Uh[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].Uh[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        len = grulayers[i].h;                                       //  Allocate all things h
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> 0
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].h
        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].bz[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].bz[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].br[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].br[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        for(j = 0; j < len; j++)
          doubleBuffer[j] = grulayers[i].bh[j];
        if(fwrite(doubleBuffer, sizeof(double), len, fp) != len)    //  From buffer, write 'len' objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  grulayers[%d].bh[%d] = %.6f\n", i, j, doubleBuffer[j]);
        #endif

        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
                                                                    //  doubleBuffer --> grulayers[i].h
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Write layer name to the buffer
          ucharBuffer[j] = grulayers[i].name[j];
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  grulayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
      }

    return true;
  }

/**************************************************************************************************
 Pooling-Layers  */

/* Add a PoolingLayer to a network in progress. */
unsigned int add_Pool(unsigned int inputW, unsigned int inputH, NeuralNet* nn)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("add_Pool(%d, %d)\n", inputW, inputH);
    #endif

    nn->poolLen++;
    if(nn->poolLen == 1)                                            //  Expand the Pool2DLayer array
      {
        if((nn->poollayers = (Pool2DLayer*)malloc(sizeof(Pool2DLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate Pool2DLayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->poollayers = (Pool2DLayer*)realloc(nn->poollayers, nn->poolLen * sizeof(Pool2DLayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Pool2DLayer array\n");
            exit(1);
          }
      }

    nn->poollayers[nn->poolLen - 1].inputW = inputW;                //  Set this newest layer's input dimensions
    nn->poollayers[nn->poolLen - 1].inputH = inputH;
    nn->poollayers[nn->poolLen - 1].n = 0;                          //  New layer initially contains zero pools
    nn->poollayers[nn->poolLen - 1].outlen = 0;                     //  An empty layer has zero output

    for(i = 0; i < LAYER_NAME_LEN; i++)                             //  Blank out layer name
      nn->poollayers[nn->poolLen - 1].name[i] = '\0';

    return nn->poolLen;
  }

bool read_Pool2D(Pool2DLayer* poollayers, unsigned int poolLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;

    unsigned int i, j;

    for(i = 0; i < poolLen; i++)
      {
                                                                    //  Allocate buffers for reading this type of layer
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
            return false;
          }
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3

        if(fread(uintBuffer, sizeof(int), 3, fp) != 3)              //  Read 3 objects of size int into buffer
          {
            printf("ERROR: Unable to read Pooling Layer int parameters into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        poollayers[i].inputW = uintBuffer[0];                       //  Read input width for Pool2DLayer[i] from buffer
        poollayers[i].inputH = uintBuffer[1];                       //  Read input height for Pool2DLayer[i] from buffer
        poollayers[i].n = uintBuffer[2];                            //  Read number of pools for Pool2DLayer[i] from buffer
                                                                    //  Read LAYER_NAME_LEN objects of size char into buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read Pooling Layer name into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Read layer name
          poollayers[i].name[j] = ucharBuffer[j];

        #ifdef __NEURON_DEBUG
        printf("  poollayers[%d].inputW = %d\n", i, poollayers[i].inputW);
        printf("  poollayers[%d].inputH = %d\n", i, poollayers[i].inputH);
        printf("  poollayers[%d].n      = %d\n", i, poollayers[i].n);
        printf("  poollayers[%d].name = %s\n", i, poollayers[i].name);
        #endif
        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0

                                                                    //  Allocate 'n' pools
        if((poollayers[i].pools = (Pool2D*)malloc(poollayers[i].n * sizeof(Pool2D))) == NULL)
          {
            printf("ERROR: Unable to allocate pool array for Pool2DLayer[%d] while reading from file\n", i);
            exit(1);
          }
        for(j = 0; j < poollayers[i].n; j++)                        //  Fill in details of each pool in this layer
          {
                                                                    //  Pools need to read 4 ints (w, h, stride_h, stride_v);
                                                                    //                     1 uchar (f);
            if((ucharBuffer = (unsigned char*)malloc(sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned char buffer while reading from file\n");
                return false;
              }
            if((uintBuffer = (unsigned int*)malloc(4 * sizeof(int))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
                return false;
              }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4

            if(fread(uintBuffer, sizeof(int), 4, fp) != 4)          //  Read 4 objects of size int into buffer
              {
                printf("ERROR: Unable to read dimensions and strides for Pooling Layer pool[%d] into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            poollayers[i].pools[j].w = uintBuffer[0];               //  Read dimensions and strides
            poollayers[i].pools[j].h = uintBuffer[1];
            poollayers[i].pools[j].stride_h = uintBuffer[2];
            poollayers[i].pools[j].stride_v = uintBuffer[3];
            if(fread(ucharBuffer, sizeof(char), 1, fp) != 1)        //  Read 1 object of size char into buffer
              {
                printf("ERROR: Unable to read pool type flag for Pooling Layer pool[%d] into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            poollayers[i].pools[j].f = ucharBuffer[0];              //  Read pool-type flag
            #ifdef __NEURON_DEBUG
            printf("  poollayers[%d].pools[%d].w = %d\n", i, j, poollayers[i].pools[j].w);
            printf("  poollayers[%d].pools[%d].h = %d\n", i, j, poollayers[i].pools[j].h);
            printf("  poollayers[%d].pools[%d].stride_h = %d\n", i, j, poollayers[i].pools[j].stride_h);
            printf("  poollayers[%d].pools[%d].stride_v = %d\n", i, j, poollayers[i].pools[j].stride_v);
            switch(poollayers[i].pools[j].f)
              {
                case MAX_POOL:     printf("  poollayers[%d].pools[%d].f = MAX\n", i, j);  break;
                case MIN_POOL:     printf("  poollayers[%d].pools[%d].f = MIN\n", i, j);  break;
                case AVG_POOL:     printf("  poollayers[%d].pools[%d].f = AVG\n", i, j);  break;
                case MEDIAN_POOL:  printf("  poollayers[%d].pools[%d].f = MEDIAN\n", i, j);  break;
              }
            #endif
            free(ucharBuffer);                                      //  ucharBuffer  --> 0
            free(uintBuffer);                                       //  uintBuffer   --> 0
          }
                                                                    //  Compute layer's output length; allocate and blank its output buffer
        poollayers[i].outlen = outputLen_Pool2D(poollayers + i);
        if((poollayers[i].out = (double*)malloc(poollayers[i].outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate output array while reading Pool2DLayer[%d] from file\n", i);
            return false;
          }                                                         //  Allocate and blank out output array
        for(j = 0; j < poollayers[i].outlen; j++)
          poollayers[i].out[j] = 0.0;
      }

    return true;
  }

bool write_Pool2D(Pool2DLayer* poollayers, unsigned int poolLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;

    unsigned int i, j;

    for(i = 0; i < poolLen; i++)
      {
                                                                    //  Allocate buffers for reading this type of layer
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while writing to file\n");
            return false;
          }
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for writing to file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
        uintBuffer[0] = poollayers[i].inputW;                       //  Copy input width for Pool2DLayer[i] to buffer
        uintBuffer[1] = poollayers[i].inputH;                       //  Copy input height for Pool2DLayer[i] to buffer
        uintBuffer[2] = poollayers[i].n;                            //  Copy number of pools for Pool2DLayer[i] to buffer
        if(fwrite(uintBuffer, sizeof(int), 3, fp) != 3)             //  From buffer, write 3 objects of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  poollayers[%d].inputW = %d\n", i, uintBuffer[0]);
        printf("  poollayers[%d].inputH = %d\n", i, uintBuffer[1]);
        printf("  poollayers[%d].n = %d\n", i, uintBuffer[2]);
        #endif

        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Write layer name to the buffer
          ucharBuffer[j] = poollayers[i].name[j];
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  poollayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0

        for(j = 0; j < poollayers[i].n; j++)                        //  Fill in details of each pool in this layer
          {
                                                                    //  Pools need to read 4 ints (w, h, stride_h, stride_v);
                                                                    //                     1 uchar (f);
            if((ucharBuffer = (unsigned char*)malloc(sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned char buffer while reading from file\n");
                return false;
              }
            if((uintBuffer = (unsigned int*)malloc(4 * sizeof(int))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
                return false;
              }
                                                                    //  ucharBuffer  --> 1
                                                                    //  uintBuffer   --> 4
            uintBuffer[0] = poollayers[i].pools[j].w;               //  Copy dimensions and strides
            uintBuffer[1] = poollayers[i].pools[j].h;
            uintBuffer[2] = poollayers[i].pools[j].stride_h;
            uintBuffer[3] = poollayers[i].pools[j].stride_v;
            ucharBuffer[0] = poollayers[i].pools[j].f;              //  Copy pool-type flag
            if(fwrite(uintBuffer, sizeof(int), 4, fp) != 4)         //  From buffer, write 4 objects of size int
              {
                printf("ERROR: Unable to write unsigned int buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            if(fwrite(ucharBuffer, sizeof(char), 1, fp) != 1)       //  From buffer, write 1 object of size char
              {
                printf("ERROR: Unable to write unsigned char buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            #ifdef __NEURON_DEBUG
            printf("  poollayers[%d].pools[%d].w = %d\n", i, j, uintBuffer[0]);
            printf("  poollayers[%d].pools[%d].h = %d\n", i, j, uintBuffer[1]);
            printf("  poollayers[%d].pools[%d].stride_h = %d\n", i, j, uintBuffer[2]);
            printf("  poollayers[%d].pools[%d].stride_v = %d\n", i, j, uintBuffer[3]);
            switch(ucharBuffer[0])
              {
                case MAX_POOL:     printf("  poollayers[%d].pools[%d].f = MAX\n", i, j);  break;
                case MIN_POOL:     printf("  poollayers[%d].pools[%d].f = MIN\n", i, j);  break;
                case AVG_POOL:     printf("  poollayers[%d].pools[%d].f = AVG\n", i, j);  break;
                case MEDIAN_POOL:  printf("  poollayers[%d].pools[%d].f = MEDIAN\n", i, j);  break;
              }
            #endif

            free(ucharBuffer);                                      //  ucharBuffer  --> 0
            free(uintBuffer);                                       //  uintBuffer   --> 0
          }
      }

    return true;
  }

/**************************************************************************************************
 Upres-Layers  */

/* Add a 2D upres layer to a network in progress. */
unsigned int add_Upres(unsigned int inputW, unsigned int inputH, NeuralNet* nn)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("add_Upres(%d, %d)\n", inputW, inputH);
    #endif

    nn->upresLen++;
    if(nn->upresLen == 1)                                           //  Expand the UpresLayer array
      {
        if((nn->upreslayers = (UpresLayer*)malloc(sizeof(UpresLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate UpresLayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->upreslayers = (UpresLayer*)realloc(nn->upreslayers, nn->upresLen * sizeof(UpresLayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate UpresLayer array\n");
            exit(1);
          }
      }

    nn->upreslayers[nn->upresLen - 1].inputW = inputW;              //  Set this newest layer's input dimensions
    nn->upreslayers[nn->upresLen - 1].inputH = inputH;
    nn->upreslayers[nn->upresLen - 1].n = 0;                        //  Initially, no up-ressings
    nn->upreslayers[nn->upresLen - 1].outlen = 0;                   //  And therefore, no output

    for(i = 0; i < LAYER_NAME_LEN; i++)                             //  Blank out layer name
      nn->upreslayers[nn->upresLen - 1].name[i] = '\0';

    return nn->upresLen;
  }

bool read_Upres(UpresLayer* upreslayers, unsigned int upresLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;

    unsigned int i, j;

    for(i = 0; i < upresLen; i++)
      {
                                                                    //  Reallocate buffers for reading this type of layer
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
            return false;
          }
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3

        if(fread(uintBuffer, sizeof(int), 3, fp) != 3)              //  Read 3 objects of size int into buffer
          {
            printf("ERROR: Unable to read Up-Res Layer int parameters into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        upreslayers[i].inputW = uintBuffer[0];                      //  Read input width for UpresLayer[i] from buffer
        upreslayers[i].inputH = uintBuffer[1];                      //  Read input height for UpresLayer[i] from buffer
        upreslayers[i].n = uintBuffer[2];                           //  Read number of up-ressings for UpresLayer[i] from buffer
                                                                    //  Read Upres-Layer's name into buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read Up-Res Layer name into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }

        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Copy layer name
          upreslayers[i].name[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        printf("  upreslayers[%d].inputW = %d\n", i, upreslayers[i].inputW);
        printf("  upreslayers[%d].inputH = %d\n", i, upreslayers[i].inputH);
        printf("  upreslayers[%d].n      = %d\n", i, upreslayers[i].n);
        printf("  upreslayers[%d].name   = %s\n", i, upreslayers[i].name);
        #endif
        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0

                                                                    //  Allocate 'n' upressings
        if((upreslayers[i].params = (UpresParams*)malloc(upreslayers[i].n * sizeof(UpresParams))) == NULL)
          {
            printf("ERROR: Unable to allocate upressing array for UpresLayer[%d] while reading from file\n", i);
            return false;
          }
        for(j = 0; j < upreslayers[i].n; j++)                       //  Fill in details of each upressing in this layer
          {
                                                                    //  Upressings need to read 4 ints (stride_h, stride_v, padding_h, padding_v);
                                                                    //                          2 uchars (sMethod, pMethod);
            if((ucharBuffer = (unsigned char*)malloc(2 * sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned char buffer while reading from file\n");
                return false;
              }
            if((uintBuffer = (unsigned int*)malloc(4 * sizeof(int))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned int buffer while reading from file\n");
                free(ucharBuffer);
                return false;
              }
                                                                    //  ucharBuffer  --> 2
                                                                    //  uintBuffer   --> 4
                                                                    //  doubleBuffer --> 0
                                                                    //  boolBuffer   --> 0
            if(fread(uintBuffer, sizeof(int), 4, fp) != 4)          //  Read 4 objects of size int into buffer
              {
                printf("ERROR: Unable to read strides and padding for Upres Layer filter[%d] into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            if(fread(ucharBuffer, sizeof(char), 2, fp) != 2)        //  Read 2 objects of size char into buffer
              {
                printf("ERROR: Unable to read stride and padding method flags for Upres Layer filter[%d] into buffer", j);
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
                                                                    //  Read strides and paddings
            upreslayers[i].params[j].stride_h = uintBuffer[0];
            upreslayers[i].params[j].stride_v = uintBuffer[1];
            upreslayers[i].params[j].padding_h = uintBuffer[2];
            upreslayers[i].params[j].padding_v = uintBuffer[3];
            upreslayers[i].params[j].sMethod = ucharBuffer[0];
            upreslayers[i].params[j].pMethod = ucharBuffer[1];
            #ifdef __NEURON_DEBUG
            printf("  upreslayers[%d].params[%d].stride_h = %d\n", i, j, upreslayers[i].params[j].stride_h);
            printf("  upreslayers[%d].params[%d].stride_v = %d\n", i, j, upreslayers[i].params[j].stride_v);
            printf("  upreslayers[%d].params[%d].padding_h = %d\n", i, j, upreslayers[i].params[j].padding_h);
            printf("  upreslayers[%d].params[%d].padding_v = %d\n", i, j, upreslayers[i].params[j].padding_v);
            printf("  upreslayers[%d].params[%d].sMethod = %d\n", i, j, upreslayers[i].params[j].sMethod);
            printf("  upreslayers[%d].params[%d].pMethod = %d\n", i, j, upreslayers[i].params[j].pMethod);
            #endif

            free(ucharBuffer);                                      //  ucharBuffer  --> 0
            free(uintBuffer);                                       //  uintBuffer   --> 0
          }
                                                                    //  Compute layer's output length; allocate and blank its output buffer
        upreslayers[i].outlen = outputLen_Upres(upreslayers + i);
        if((upreslayers[i].out = (double*)malloc(upreslayers[i].outlen * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate output array while reading UpresLayer[%d] from file\n", i);
            return false;
          }                                                         //  Allocate and blank out output array
        for(j = 0; j < upreslayers[i].outlen; j++)
          upreslayers[i].out[j] = 0.0;
      }

    return true;
  }

bool write_Upres(UpresLayer* upreslayers, unsigned int upresLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;

    unsigned int i, j;

    for(i = 0; i < upresLen; i++)
      {
                                                                    //  Reallocate buffers for reading this type of layer
        if((uintBuffer = (unsigned int*)malloc(3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer while writing to file\n");
            return false;
          }
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for writing to file\n");
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 3
        uintBuffer[0] = upreslayers[i].inputW;                      //  Copy input width for UpresLayer[i] to buffer
        uintBuffer[1] = upreslayers[i].inputH;                      //  Copy input height for UpresLayer[i] to buffer
        uintBuffer[2] = upreslayers[i].n;                           //  Copy number of up-ressings for UpresLayer[i] to buffer
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Copy layer name
          ucharBuffer[j] = upreslayers[i].name[j];
        if(fwrite(uintBuffer, sizeof(int), 3, fp) != 3)             //  From buffer, write 3 objects of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  upreslayers[%d].inputW = %d\n", i, uintBuffer[0]);
        printf("  upreslayers[%d].inputH = %d\n", i, uintBuffer[1]);
        printf("  upreslayers[%d].n = %d\n", i, uintBuffer[2]);
        printf("  upreslayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif
        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0

        for(j = 0; j < upreslayers[i].n; j++)                       //  Fill in details of each upressing in this layer
          {
                                                                    //  Upressings need to read 4 ints (stride_h, stride_v, padding_h, padding_v);
                                                                    //                          2 uchars (sMethod, pMethod);
            if((ucharBuffer = (unsigned char*)malloc(2 * sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned char buffer while writing to file\n");
                return false;
              }
            if((uintBuffer = (unsigned int*)malloc(4 * sizeof(int))) == NULL)
              {
                printf("ERROR: Unable to allocate unsigned int buffer while writing to file\n");
                free(ucharBuffer);
                return false;
              }
                                                                    //  ucharBuffer  --> 2
                                                                    //  uintBuffer   --> 4
            uintBuffer[0] = upreslayers[i].params[j].stride_h;      //  Copy strides and paddings to buffer
            uintBuffer[1] = upreslayers[i].params[j].stride_v;
            uintBuffer[2] = upreslayers[i].params[j].padding_h;
            uintBuffer[3] = upreslayers[i].params[j].padding_v;
            ucharBuffer[0] = upreslayers[i].params[j].sMethod;
            ucharBuffer[1] = upreslayers[i].params[j].pMethod;
            if(fwrite(uintBuffer, sizeof(int), 4, fp) != 4)         //  From buffer, write 4 objects of size int
              {
                printf("ERROR: Unable to write unsigned int buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            if(fwrite(ucharBuffer, sizeof(char), 2, fp) != 2)       //  From buffer, write 2 objects of size char
              {
                printf("ERROR: Unable to write unsigned char buffer to file.\n");
                free(ucharBuffer);
                free(uintBuffer);
                return false;
              }
            #ifdef __NEURON_DEBUG
            printf("  upreslayers[%d].params[%d].stride_h = %d\n", i, j, uintBuffer[0]);
            printf("  upreslayers[%d].params[%d].stride_v = %d\n", i, j, uintBuffer[1]);
            printf("  upreslayers[%d].params[%d].padding_h = %d\n", i, j, uintBuffer[2]);
            printf("  upreslayers[%d].params[%d].padding_v = %d\n", i, j, uintBuffer[3]);
            printf("  upreslayers[%d].params[%d].sMethod = %d\n", i, j, ucharBuffer[0]);
            printf("  upreslayers[%d].params[%d].pMethod = %d\n", i, j, ucharBuffer[1]);
            #endif

            free(ucharBuffer);                                      //  ucharBuffer  --> 0
            free(uintBuffer);                                       //  uintBuffer   --> 0
          }
      }

    return true;
  }

/**************************************************************************************************
 Normalization-Layers  */

unsigned int add_Normal(unsigned int inputs, NeuralNet* nn)
  {
    unsigned int i;

    #ifdef __NEURON_DEBUG
    printf("add_Normal(%d)\n", inputs);
    #endif

    nn->normalLen++;

    if(nn->normalLen == 1)                                          //  Expand the NormalLayer array
      {
        if((nn->normlayers = (NormalLayer*)malloc(sizeof(NormalLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate NormalLayer array\n");
            exit(1);
          }
      }
    else
      {
        if((nn->normlayers = (NormalLayer*)realloc(nn->normlayers, nn->normalLen * sizeof(NormalLayer))) == NULL)
          {
            printf("ERROR: Unable to re-allocate NormalLayer array\n");
            exit(1);
          }
      }

    nn->normlayers[nn->normalLen - 1].i = inputs;
    nn->normlayers[nn->normalLen - 1].m = 0.0;                      //  Initialize to no effect:
    nn->normlayers[nn->normalLen - 1].s = 1.0;                      //  y = g * ((x - m) / s) + b
    nn->normlayers[nn->normalLen - 1].g = 1.0;
    nn->normlayers[nn->normalLen - 1].b = 0.0;
                                                                    //  Allocate output buffer
    if((nn->normlayers[nn->normalLen - 1].out = (double*)malloc(inputs * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate NormalLayer's internal output array\n");
        exit(1);
      }
    for(i = 0; i < inputs; i++)                                     //  Blank out 'out' array
      nn->normlayers[nn->normalLen - 1].out[i] = 0.0;
    for(i = 0; i < LAYER_NAME_LEN; i++)                             //  Blank out layer name
      nn->normlayers[nn->normalLen - 1].name[i] = '\0';

    return nn->normalLen;
  }

bool read_Normal(NormalLayer* normlayers, unsigned int normalLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int i, j;

    for(i = 0; i < normalLen; i++)
      {
                                                                    //  Allocate buffers for reading this type of layer
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            return false;
          }
        if((uintBuffer = (unsigned int*)malloc(sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer for reading from file\n");
            free(ucharBuffer);
            return false;
          }
        if((doubleBuffer = (double*)malloc(4 * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer while reading from file\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 1
                                                                    //  doubleBuffer --> 4
        if(fread(uintBuffer, sizeof(int), 1, fp) != 1)              //  Read 1 object of size int into buffer
          {
            printf("ERROR: Unable to read Normal Layer int parameter into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if(fread(doubleBuffer, sizeof(double), 4, fp) != 4)         //  Read 4 objects of size double into buffer
          {
            printf("ERROR: Unable to read Normal Layer double parameters into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }

        normlayers[i].i = uintBuffer[0];                            //  Read number of inputs for NormalLayer[i] from buffer

        normlayers[i].m = doubleBuffer[0];                          //  Read mean for NormalLayer[i] from buffer
        normlayers[i].s = doubleBuffer[1];                          //  Read standard deviation height for NormalLayer[i] from buffer
        normlayers[i].g = doubleBuffer[2];                          //  Read coefficient for NormalLayer[i] from buffer
        normlayers[i].b = doubleBuffer[3];                          //  Read constant for NormalLayer[i] from buffer

        #ifdef __NEURON_DEBUG
        printf("  normlayers[%d].i = %d\n", i, normlayers[i].i);
        printf("  normlayers[%d].m = %f\n", i, normlayers[i].m);
        printf("  normlayers[%d].s = %f\n", i, normlayers[i].s);
        printf("  normlayers[%d].g = %f\n", i, normlayers[i].g);
        printf("  normlayers[%d].b = %f\n", i, normlayers[i].b);
        #endif

        if((normlayers[i].out = (double*)malloc(normlayers[i].i * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate output array while reading NormalLayer[%d] from file\n", i);
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < normlayers[i].i; j++)                        //  Allocate and blank out output array
          normlayers[i].out[j] = 0.0;
                                                                    //  Read LAYER_NAME_LEN objects of size char to buffer
        if(fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to read Normal Layer name into buffer\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Read layer name
          normlayers[i].name[j] = ucharBuffer[j];
        #ifdef __NEURON_DEBUG
        printf("  normlayers[%d].name = %s\n\n", i, normlayers[i].name);
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
      }

    return true;
  }

bool write_Normal(NormalLayer* normlayers, unsigned int normalLen, FILE* fp)
  {
    unsigned char* ucharBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;

    unsigned int i, j;

    for(i = 0; i < normalLen; i++)
      {
                                                                    //  Allocate buffers for reading this type of layer
        if((ucharBuffer = (unsigned char*)malloc(LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
            return false;
          }
        if((uintBuffer = (unsigned int*)malloc(sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to allocate unsigned int buffer for writing to file\n");
            free(ucharBuffer);
            return false;
          }
        if((doubleBuffer = (double*)malloc(4 * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate double buffer while writing to file\n");
            free(ucharBuffer);
            free(uintBuffer);
            return false;
          }
                                                                    //  ucharBuffer  --> LAYER_NAME_LEN
                                                                    //  uintBuffer   --> 1
                                                                    //  doubleBuffer --> 4
        uintBuffer[0] = normlayers[i].i;                            //  Copy number of inputs for NormalLayer[i] to buffer

        doubleBuffer[0] = normlayers[i].m;                          //  Copy mean for NormalLayer[i] to buffer
        doubleBuffer[1] = normlayers[i].s;                          //  Copy standard deviation height for NormalLayer[i] to buffer
        doubleBuffer[2] = normlayers[i].g;                          //  Copy coefficient for NormalLayer[i] to buffer
        doubleBuffer[3] = normlayers[i].b;                          //  Copy constant for NormalLayer[i] to buffer
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Copy layer name
          ucharBuffer[j] = normlayers[i].name[j];

        if(fwrite(uintBuffer, sizeof(int), 1, fp) != 1)             //  From buffer, write 1 object of size int
          {
            printf("ERROR: Unable to write unsigned int buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        if(fwrite(doubleBuffer, sizeof(double), 4, fp) != 4)        //  From buffer, write 4 objects of size double
          {
            printf("ERROR: Unable to write double buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
                                                                    //  From buffer, write LAYER_NAME_LEN objects of size char
        if(fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp) != LAYER_NAME_LEN)
          {
            printf("ERROR: Unable to write unsigned char buffer to file.\n");
            free(ucharBuffer);
            free(uintBuffer);
            free(doubleBuffer);
            return false;
          }
        #ifdef __NEURON_DEBUG
        printf("  normlayers[%d].i = %d\n", i, uintBuffer[0]);
        printf("  normlayers[%d].m = %.6f\n", i, doubleBuffer[0]);
        printf("  normlayers[%d].s = %.6f\n", i, doubleBuffer[1]);
        printf("  normlayers[%d].g = %.6f\n", i, doubleBuffer[2]);
        printf("  normlayers[%d].b = %.6f\n", i, doubleBuffer[3]);
        printf("  normlayers[%d].name = ", i);
        for(j = 0; j < LAYER_NAME_LEN; j++)
          {
            if(ucharBuffer[j] > 0)
              printf("%c", ucharBuffer[j]);
          }
        printf("\n");
        #endif

        free(ucharBuffer);                                          //  ucharBuffer  --> 0
        free(uintBuffer);                                           //  uintBuffer   --> 0
        free(doubleBuffer);                                         //  doubleBuffer --> 0
      }

    return true;
  }

#endif
