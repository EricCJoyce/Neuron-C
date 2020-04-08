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

#ifndef __NEURON_H
#define __NEURON_H

#define INPUT_ARRAY   0                                             /* Flag refers to network input */
#define DENSE_ARRAY   1                                             /* Flag refers to 'denselayers' */
#define CONV2D_ARRAY  2                                             /* Flag refers to 'convlayers' */
#define ACCUM_ARRAY   3                                             /* Flag refers to 'accumlayers' */
#define LSTM_ARRAY    4                                             /* Flag refers to 'lstmlayers' */
#define GRU_ARRAY     5                                             /* Flag refers to 'grulayers' */

#define RELU                 0                                      /* [ 0.0, inf) */
#define LEAKY_RELU           1                                      /* (-inf, inf) */
#define SIGMOID              2                                      /* ( 0.0, 1.0) */
#define HYPERBOLIC_TANGENT   3                                      /* [-1.0, 1.0] */
#define SOFTMAX              4                                      /* [ 0.0, 1.0] */
#define SYMMETRICAL_SIGMOID  5                                      /* (-1.0, 1.0) */
#define THRESHOLD            6                                      /* { 0.0, 1.0} */
#define LINEAR               7                                      /* (-inf, inf) */

#define VARSTR_LEN      16                                          /* Length of a Variable key string */
#define LAYER_NAME_LEN  32                                          /* Length of a Layer 'name' string */
#define COMMSTR_LEN     64                                          /* Length of a Network Comment string */

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

typedef struct Filter2DType
  {
    unsigned int w;                                                 //  Width of the filter
    unsigned int h;                                                 //  Height of the filter
    double* W;                                                      //  Array of (w * h) weights, arranged row-major, +1 for the bias
  } Filter2D;

typedef struct Conv2DLayerType
  {
    unsigned int inputW, inputH;                                    //  Dimensions of the input
    unsigned int n;                                                 //  Number of processing units in this layer =
                                                                    //  number of filters in this layer
    Filter2D* filters;                                              //  Array of 2D filter structs

    unsigned char* f;                                               //  Array of function flags, length equal to the number of filters
    double* alpha;                                                  //  Array of function parameters, length equal to the number of filters

    char name[LAYER_NAME_LEN];
    double* out;
  } Conv2DLayer;

typedef struct AccumLayerType
  {
    unsigned int i;                                                 //  Number of inputs--ACCUMULATORS GET NO bias-1
    char name[LAYER_NAME_LEN];
    double* out;
  } AccumLayer;

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

unsigned int add_LSTM(unsigned int, unsigned int, unsigned int, NeuralNet*);
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

unsigned int add_GRU(unsigned int, unsigned int, unsigned int, NeuralNet*);
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

unsigned int add_Conv2D(unsigned int, unsigned int, NeuralNet*);
unsigned int add_Conv2DFilter(unsigned int, unsigned int, Conv2DLayer*);
void setW_i_Conv2D(double*, unsigned int, Conv2DLayer*);             //  Set entirety of i-th filter; w is length width * height + 1
void setW_ij_Conv2D(double, unsigned int, unsigned int, Conv2DLayer*);
void setF_i_Conv2D(unsigned char, unsigned int, Conv2DLayer*);       //  Set activation function of i-th filter
void setA_i_Conv2D(double, unsigned int, Conv2DLayer*);              //  Set activation function auxiliary parameter of i-th filter
void setName_Conv2D(char*, Conv2DLayer*);
void print_Conv2D(Conv2DLayer*);
unsigned int outputLen_Conv2D(Conv2DLayer*);
unsigned int run_Conv2D(double*, Conv2DLayer*);

unsigned int add_Accum(unsigned int, NeuralNet*);
void setName_Accum(char* n, AccumLayer*);

unsigned int add_Dense(unsigned int, unsigned int, NeuralNet*);
void setW_Dense(double*, DenseLayer*);                              //  Set entirety of layer's weight matrix
void setW_i_Dense(double*, unsigned int, DenseLayer*);              //  Set entirety of weights for i-th column/neuron/unit
void setW_ij_Dense(double, unsigned int, unsigned int, DenseLayer*);//  Set element [i, j] of layer's weight matrix
void setM_Dense(bool*, DenseLayer*);                                //  Set entirety of layer's mask matrix
void setM_i_Dense(bool*, unsigned int, DenseLayer*);                //  Set entirety of masks for i-th column/neuron/unit
void setM_ij_Dense(bool, unsigned int, unsigned int, DenseLayer*);  //  Set element [i, j] of layer's mask matrix
void setF_i_Dense(unsigned char, unsigned int, DenseLayer*);        //  Set activation function of i-th neuron/unit
void setA_i_Dense(double, unsigned int, DenseLayer*);               //  Set activation function auxiliary parameter of i-th neuron/unit
void setName_Dense(char* n, DenseLayer*);
void print_Dense(DenseLayer*);
unsigned int outputLen_Dense(DenseLayer*);
unsigned int run_Dense(double*, DenseLayer*);

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

        if(nn->denseLen > 0)
          {
            for(i = 0; i < nn->denseLen; i++)
              {
                free(nn->denselayers[i].W);
                free(nn->denselayers[i].M);
                free(nn->denselayers[i].f);
                free(nn->denselayers[i].alpha);
                free(nn->denselayers[i].out);
              }
            free(nn->denselayers);
          }

        if(nn->convLen > 0)
          {
            for(i = 0; i < nn->convLen; i++)
              {
                if(nn->convlayers[i].n > 0)
                  {
                    for(j = 0; j < nn->convlayers[i].n; j++)
                      free(nn->convlayers[i].filters[j].W);
                    free(nn->convlayers[i].filters);
                  }
              }
            free(nn->convlayers);
          }

        if(nn->accumLen > 0)
          {
            for(i = 0; i < nn->accumLen; i++)
              free(nn->accumlayers[i].out);
            free(nn->accumlayers);
          }

        if(nn->lstmLen > 0)
          {
            for(i = 0; i < nn->lstmLen; i++)
              {
                free(nn->lstmlayers[i].Wi);
                free(nn->lstmlayers[i].Wo);
                free(nn->lstmlayers[i].Wf);
                free(nn->lstmlayers[i].Wc);

                free(nn->lstmlayers[i].Ui);
                free(nn->lstmlayers[i].Uo);
                free(nn->lstmlayers[i].Uf);
                free(nn->lstmlayers[i].Uc);

                free(nn->lstmlayers[i].bi);
                free(nn->lstmlayers[i].bo);
                free(nn->lstmlayers[i].bf);
                free(nn->lstmlayers[i].bc);

                free(nn->lstmlayers[i].c);
                free(nn->lstmlayers[i].H);
              }
            free(nn->lstmlayers);
          }

        if(nn->gruLen > 0)
          {
            for(i = 0; i < nn->gruLen; i++)
              {
                free(nn->grulayers[i].Wz);
                free(nn->grulayers[i].Wr);
                free(nn->grulayers[i].Wh);

                free(nn->grulayers[i].Uz);
                free(nn->grulayers[i].Ur);
                free(nn->grulayers[i].Uh);

                free(nn->grulayers[i].bz);
                free(nn->grulayers[i].br);
                free(nn->grulayers[i].bh);

                free(nn->grulayers[i].H);
              }
            free(nn->grulayers);
          }

        if(nn->vars > 0)
          free(nn->variables);

        free(nn);
      }

    return;
  }

/* Input vector 'x' has length = 'nn'->'i'.
   Output vector 'z' will have length = # of units/outputs in last layer */
unsigned int run_NN(double* x, NeuralNet* nn, double** z)
  {
    double* in;
    unsigned int inLen;
    unsigned int outLen;
    unsigned int i, j, k, l;
    unsigned int last;
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
            case DENSE_ARRAY:  inLen = nn->denselayers[nn->edgelist[i].dstIndex].i;
                               break;
            case CONV2D_ARRAY: inLen = nn->convlayers[nn->edgelist[i].dstIndex].inputW *
                                       nn->convlayers[nn->edgelist[i].dstIndex].inputH;
                               break;
            case ACCUM_ARRAY:  inLen = nn->accumlayers[nn->edgelist[i].dstIndex].i;
                               break;
            case LSTM_ARRAY:   inLen = nn->lstmlayers[nn->edgelist[i].dstIndex].d;
                               break;
            case GRU_ARRAY:    inLen = nn->grulayers[nn->edgelist[i].dstIndex].d;
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
                                           in[k] = nn->lstmlayers[nn->edgelist[j].srcIndex].H[ t * nn->lstmlayers[nn->edgelist[j].srcIndex].h + l];
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
                                           in[k] = nn->grulayers[nn->edgelist[j].srcIndex].H[ t * nn->grulayers[nn->edgelist[j].srcIndex].h + l];
                                           k++;
                                         }
                                       break;
              }
            j++;
          }

        switch(nn->edgelist[i].dstType)                             //  Which array contains the destination layer?
          {
            case DENSE_ARRAY:  outLen = run_Dense(in, nn->denselayers + nn->edgelist[i].dstIndex);
                               break;
            case CONV2D_ARRAY: outLen = run_Conv2D(in, nn->convlayers + nn->edgelist[i].dstIndex);
                               break;
            case ACCUM_ARRAY:  outLen = inLen;
                               #ifdef __NEURON_DEBUG
                               printf("run_Accum()\n");
                               #endif
                               for(k = 0; k < inLen; k++)
                                 nn->accumlayers[nn->edgelist[i].dstIndex].out[k] = in[k];
                               break;
            case LSTM_ARRAY:   outLen = run_LSTM(in, nn->lstmlayers + nn->edgelist[i].dstIndex);
                               break;
            case GRU_ARRAY:    outLen = run_GRU(in, nn->grulayers + nn->edgelist[i].dstIndex);
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
        case DENSE_ARRAY:  for(i = 0; i < outLen; i++)
                             (*z)[i] = nn->denselayers[nn->edgelist[last].dstIndex].out[i];
                           break;
        case CONV2D_ARRAY: for(i = 0; i < outLen; i++)
                             (*z)[i] = nn->convlayers[nn->edgelist[last].dstIndex].out[i];
                           break;
        case ACCUM_ARRAY:  for(i = 0; i < outLen; i++)
                             (*z)[i] = nn->accumlayers[nn->edgelist[last].dstIndex].out[i];
                           break;
        case LSTM_ARRAY:   if(nn->lstmlayers[nn->edgelist[last].dstIndex].t >= nn->lstmlayers[nn->edgelist[last].dstIndex].cache)
                             t = nn->lstmlayers[nn->edgelist[last].dstIndex].cache - 1;
                           else
                             t = nn->lstmlayers[nn->edgelist[last].dstIndex].t - 1;
                           for(i = 0; i < outLen; i++)
                             (*z)[i] = nn->lstmlayers[nn->edgelist[last].dstIndex].H[t * nn->lstmlayers[nn->edgelist[last].dstIndex].h + i];
                           break;
        case GRU_ARRAY:    if(nn->grulayers[nn->edgelist[last].dstIndex].t >= nn->grulayers[nn->edgelist[last].dstIndex].cache)
                             t = nn->grulayers[nn->edgelist[last].dstIndex].cache - 1;
                           else
                             t = nn->grulayers[nn->edgelist[last].dstIndex].t - 1;
                           for(i = 0; i < outLen; i++)
                             (*z)[i] = nn->grulayers[nn->edgelist[last].dstIndex].H[t * nn->grulayers[nn->edgelist[last].dstIndex].h + i];
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
    Node* visited;
    unsigned int vlen = 0;

    #ifdef __NEURON_DEBUG
    printf("linkLayers(");
    switch(srcFlag)
      {
        case INPUT_ARRAY:  printf("Input, ");  break;
        case DENSE_ARRAY:  printf("Dense, ");  break;
        case CONV2D_ARRAY: printf("Conv2D, "); break;
        case ACCUM_ARRAY:  printf("Accum, ");  break;
        case LSTM_ARRAY:   printf("LSTM, ");  break;
        case GRU_ARRAY:    printf("GRU, ");  break;
      }
    printf("%d, %d, %d, ", src, selectorStart, selectorEnd);
    switch(dstFlag)
      {
        case DENSE_ARRAY:  printf("Dense, ");  break;
        case CONV2D_ARRAY: printf("Conv2D, "); break;
        case ACCUM_ARRAY:  printf("Accum, ");  break;
        case LSTM_ARRAY:   printf("LSTM, ");  break;
        case GRU_ARRAY:    printf("GRU, ");  break;
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
        printf("Edge rejected because output from dense layer %d does not match input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == LSTM_ARRAY &&           //  Dense-->LSTM
       outputLen_Dense(nn->denselayers + src) > nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == DENSE_ARRAY && dstFlag == GRU_ARRAY &&            //  Dense-->GRU
       outputLen_Dense(nn->denselayers + src) > nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from dense layer %d does not match input for GRU layer %d.\n", src, dst);
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
        printf("Edge rejected because output from convolutional layer %d does not match input for accumulator layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == LSTM_ARRAY &&          //  Conv2D-->LSTM
       outputLen_Conv2D(nn->convlayers + src) > nn->lstmlayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for LSTM layer %d.\n", src, dst);
        #endif
        return false;
      }
    if(srcFlag == CONV2D_ARRAY && dstFlag == GRU_ARRAY &&           //  Conv2D-->GRU
       outputLen_Conv2D(nn->convlayers + src) > nn->grulayers[dst].d)
      {
        #ifdef __NEURON_DEBUG
        printf("Edge rejected because output from convolutional layer %d does not match input for GRU layer %d.\n", src, dst);
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
        printf("Edge rejected because output from accumulator layer %d does not match input for accumulator layer %d.\n", src, dst);
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
        printf("Edge rejected because output from LSTM layer %d does not match input for accumulator layer %d.\n", src, dst);
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
        printf("Edge rejected because output from GRU layer %d does not match input for accumulator layer %d.\n", src, dst);
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
    bool* boolBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;
    unsigned int len;
    unsigned int i, j, k;

    #ifdef __NEURON_DEBUG
    printf("load_NN(%s)\n", filename);
    #endif

    fp = fopen(filename, "rb");

    if((ucharBuffer = (unsigned char*)malloc(sizeof(char))) == NULL)//  Allocate 1 uchar
      {
        printf("ERROR: Unable to allocate unsigned char buffer for reading from file\n");
        exit(1);
      }
                                                                    //  Allocate 7 uints
    if((uintBuffer = (unsigned int*)malloc(7 * sizeof(int))) == NULL)
      {
        printf("ERROR: Unable to allocate unsigned int buffer for reading from file\n");
        exit(1);
      }
    if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)    //  Allocate 1 double
      {
        printf("ERROR: Unable to allocate double buffer for reading from file\n");
        exit(1);
      }

    fread(uintBuffer, sizeof(int), 7, fp);                          //  Read 6 objects of size int into buffer
    nn->i = uintBuffer[0];                                          //  Read NeuralNet input count from buffer
    nn->len = uintBuffer[1];                                        //  Read NeuralNet edge list length from buffer
    nn->denseLen = uintBuffer[2];                                   //  Read NeuralNet DenseLayer list length from buffer
    nn->convLen = uintBuffer[3];                                    //  Read NeuralNet Conv2DLayer list length from buffer
    nn->accumLen = uintBuffer[4];                                   //  Read NeuralNet AccumLayer list length from buffer
    nn->lstmLen = uintBuffer[5];                                    //  Read NeuralNet LSTMLayer list length from buffer
    nn->gruLen = uintBuffer[6];                                     //  Read NeuralNet GRULayer list length from buffer

    #ifdef __NEURON_DEBUG
    printf("  nn->i        = %d\n", nn->i);
    printf("  nn->len      = %d\n", nn->len);
    printf("  nn->denseLen = %d\n", nn->denseLen);
    printf("  nn->convLen  = %d\n", nn->convLen);
    printf("  nn->accumLen = %d\n", nn->accumLen);
    printf("  nn->lstmLen  = %d\n", nn->lstmLen);
    printf("  nn->gruLen   = %d\n", nn->gruLen);
    #endif

    fread(ucharBuffer, sizeof(char), 1, fp);                        //  Read 1 object of size char into buffer
    nn->vars = ucharBuffer[0];                                      //  Read NeuralNet variable count from buffer

    #ifdef __NEURON_DEBUG
    printf("  nn->vars     = %d\n", nn->vars);
    #endif

    fread(uintBuffer, sizeof(int), 1, fp);                          //  Read 1 object of size int into buffer
    nn->gen = uintBuffer[0];                                        //  Read NeuralNet generation/epoch from buffer

    #ifdef __NEURON_DEBUG
    printf("  nn->gen      = %d\n", nn->gen);
    #endif

    fread(doubleBuffer, sizeof(double), 1, fp);                     //  Read 1 object of size double into buffer
    nn->fit = doubleBuffer[0];                                      //  Read NeuralNet fitness from buffer

    #ifdef __NEURON_DEBUG
    printf("  nn->fit      = %.6f\n", nn->fit);
    #endif

    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, COMMSTR_LEN * sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
        exit(1);
      }
    fread(ucharBuffer, sizeof(char), COMMSTR_LEN, fp);              //  Read COMMSTR_LEN objects of size char into buffer
    for(i = 0; i < COMMSTR_LEN; i++)                                //  Read NeuralNet comment from buffer
      nn->comment[i] = ucharBuffer[i];

    #ifdef __NEURON_DEBUG
    printf("  nn->comment  = %s\n  Edge List:\n", nn->comment);
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
    for(i = 0; i < nn->len; i++)                                    //  Read all Edges from file
      {
        fread(ucharBuffer, sizeof(char), 1, fp);                    //  Read 1 object of size char into buffer
        nn->edgelist[i].srcType = ucharBuffer[0];                   //  Read edge source type from buffer
        fread(uintBuffer, sizeof(int), 3, fp);                      //  Read 3 objects of size int into buffer
        nn->edgelist[i].srcIndex = uintBuffer[0];                   //  Read edge source index from buffer
        nn->edgelist[i].selectorStart = uintBuffer[1];              //  Read edge selector start from buffer
        nn->edgelist[i].selectorEnd = uintBuffer[2];                //  Read edge selector end from buffer
        fread(ucharBuffer, sizeof(char), 1, fp);                    //  Read 1 object of size char into buffer
        nn->edgelist[i].dstType = ucharBuffer[0];                   //  Read edge destination type from buffer
        fread(uintBuffer, sizeof(int), 1, fp);                      //  Read 1 object of size int into buffer
        nn->edgelist[i].dstIndex = uintBuffer[0];                   //  Read edge destination index from buffer

        #ifdef __NEURON_DEBUG
        printf("    (%d, %d, %d, %d, %d, %d)\n", nn->edgelist[i].srcType,
                                                 nn->edgelist[i].srcIndex,
                                                 nn->edgelist[i].selectorStart,
                                                 nn->edgelist[i].selectorEnd,
                                                 nn->edgelist[i].dstType,
                                                 nn->edgelist[i].dstIndex);
        #endif
      }
    #ifdef __NEURON_DEBUG
    printf("\n");
    #endif

    if(nn->denseLen > 0)
      {
                                                                    //  Allocate network's DenseLayer array
        if((nn->denselayers = (DenseLayer*)malloc(nn->denseLen * sizeof(DenseLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate DenseLayer array while reading from file\n");
            exit(1);
          }
        for(i = 0; i < nn->denseLen; i++)
          {
            fread(uintBuffer, sizeof(int), 2, fp);                  //  Read 2 objects of size int into buffer
            nn->denselayers[i].i = uintBuffer[0];                   //  Read number of inputs for DenseLayer[i] from buffer
            nn->denselayers[i].n = uintBuffer[1];                   //  Read number of units for DenseLayer[i] from buffer
            #ifdef __NEURON_DEBUG
            printf("  nn->denselayers[%d].i = %d\n", i, nn->denselayers[i].i);
            printf("  nn->denselayers[%d].n = %d\n", i, nn->denselayers[i].n);
            #endif

            len = (nn->denselayers[i].i + 1) * nn->denselayers[i].n;
            if((nn->denselayers[i].W = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array while reading DenseLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->denselayers[i].M = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate mask array while reading DenseLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->denselayers[i].out = (double*)malloc(nn->denselayers[i].n * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate output array while reading DenseLayer[%d] from file\n", i);
                exit(1);
              }
            for(j = 0; j < nn->denselayers[i].n; j++)               //  Allocate and blank out output array
              nn->denselayers[i].out[j] = 0.0;
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read DenseLayer[i]'s weights from buffer
              nn->denselayers[i].W[j] = doubleBuffer[j];            //  in the order in which they exist in the file.
            #ifdef __NEURON_DEBUG
            for(j = 0; j < len; j++)
              printf("  nn->denselayers[%d].W[%d] = %.6f\n", i, j, nn->denselayers[i].W[j]);
            #endif

            if(i == 0)                                              //  If this is our first use of the Bool buffer, malloc
              {
                if((boolBuffer = (bool*)malloc(len * sizeof(bool))) == NULL)
                  {
                    printf("ERROR: Unable to allocate Boolean buffer while reading from file\n");
                    exit(1);
                  }
              }
            else                                                    //  Otherwise, realloc
              {
                if((boolBuffer = (bool*)realloc(boolBuffer, len * sizeof(bool))) == NULL)
                  {
                    printf("ERROR: Unable to reallocate Boolean buffer while reading from file\n");
                    exit(1);
                  }
              }
            fread(boolBuffer, sizeof(bool), len, fp);               //  Read 'len' objects of size bool into buffer
            for(j = 0; j < len; j++)                                //  Read DenseLayer[i]'s weights from buffer
              {
                if(boolBuffer[j])                                   //  True means UNMASKED, means w * 1.0
                  nn->denselayers[i].M[j] = 1.0;
                else                                                //  False means MASKED, means w * 0.0
                  nn->denselayers[i].M[j] = 0.0;
              }
            #ifdef __NEURON_DEBUG
            for(j = 0; j < len; j++)
              printf("  nn->denselayers[%d].M[%d] = %.6f\n", i, j, nn->denselayers[i].M[j]);
            #endif

            len = nn->denselayers[i].n;
            if((nn->denselayers[i].f = (unsigned char*)malloc(len * sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate function flag array for DenseLayer[%d] while reading from file\n", i);
                exit(1);
              }
            if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, len * sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
                exit(1);
              }
            fread(ucharBuffer, sizeof(char), len, fp);              //  Read 'len' objects of size char into buffer
            for(j = 0; j < len; j++)                                //  Read function flags
              nn->denselayers[i].f[j] = ucharBuffer[j];
            #ifdef __NEURON_DEBUG
            for(j = 0; j < len; j++)
              printf("  nn->denselayers[%d].f[%d] = %d\n", i, j, nn->denselayers[i].f[j]);
            #endif

            if((nn->denselayers[i].alpha = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate function auxiliaries array for DenseLayer[%d] while reading from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read function auxiliaries
              nn->denselayers[i].alpha[j] = doubleBuffer[j];
            #ifdef __NEURON_DEBUG
            for(j = 0; j < len; j++)
              printf("  nn->denselayers[%d].alpha[%d] = %.6f\n", i, j, nn->denselayers[i].alpha[j]);
            #endif

            if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
                exit(1);
              }
            fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);   //  Read LAYER_NAME_LEN objects of size char to buffer
            for(j = 0; j < LAYER_NAME_LEN; j++)                     //  Read layer name
              nn->denselayers[i].name[j] = ucharBuffer[j];
            #ifdef __NEURON_DEBUG
            printf("  nn->denselayers[%d].name = %s\n\n", i, nn->denselayers[i].name);
            #endif
          }
        free(boolBuffer);                                           //  We won't need the Boolean buffer again
      }

    if(nn->convLen > 0)
      {
                                                                    //  Allocate network's Conv2DLayer array
        if((nn->convlayers = (Conv2DLayer*)malloc(nn->convLen * sizeof(Conv2DLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer array while reading from file\n");
            exit(1);
          }
        for(i = 0; i < nn->convLen; i++)                            //  Read in all convolutional layers
          {
            fread(uintBuffer, sizeof(int), 3, fp);                  //  Read 3 objects of size int into buffer
            nn->convlayers[i].inputW = uintBuffer[0];               //  Read Conv2DLayer input width from buffer
            nn->convlayers[i].inputH = uintBuffer[1];               //  Read Conv2DLayer input height from buffer
            nn->convlayers[i].n = uintBuffer[2];                    //  Read number of Conv2DLayer filters from buffer
            #ifdef __NEURON_DEBUG
            printf("  nn->convlayers[%d].inputW = %d\n", i, nn->convlayers[i].inputW);
            printf("  nn->convlayers[%d].inputH = %d\n", i, nn->convlayers[i].inputH);
            printf("  nn->convlayers[%d].n      = %d\n", i, nn->convlayers[i].n);
            #endif
                                                                    //  Allocate 'n' filters
            if((nn->convlayers[i].filters = (Filter2D*)malloc(nn->convlayers[i].n * sizeof(Filter2D))) == NULL)
              {
                printf("ERROR: Unable to allocate filter array for Conv2DLayer[%d] while reading from file\n", i);
                exit(1);
              }
            for(j = 0; j < nn->convlayers[i].n; j++)                //  Fill in details of each filter in this layer
              {
                fread(uintBuffer, sizeof(int), 2, fp);              //  Read 2 objects of size int into buffer
                nn->convlayers[i].filters[j].w = uintBuffer[0];     //  Read dimensions
                nn->convlayers[i].filters[j].h = uintBuffer[1];
                #ifdef __NEURON_DEBUG
                printf("  nn->convlayers[%d].filters[%d].w = %d\n", i, j, nn->convlayers[i].filters[j].w);
                printf("  nn->convlayers[%d].filters[%d].h = %d\n", i, j, nn->convlayers[i].filters[j].h);
                #endif

                len = nn->convlayers[i].filters[j].w * nn->convlayers[i].filters[j].h + 1;
                if((nn->convlayers[i].filters[j].W = (double*)malloc(len * sizeof(double))) == NULL)
                  {
                    printf("ERROR: Unable to allocate filter[%d] of Conv2DLayer[%d] while reading from file\n", j, i);
                    exit(1);
                  }
                if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
                  {
                    printf("ERROR: Unable to reallocate double buffer for writing to file\n");
                    exit(1);
                  }
                fread(doubleBuffer, sizeof(double), len, fp);       //  Read len objects of size double into buffer
                for(k = 0; k < len; k++)                            //  Read Filter2D weights and bias
                  nn->convlayers[i].filters[j].W[k] = doubleBuffer[k];
                #ifdef __NEURON_DEBUG
                printf("  ");
                for(k = 0; k < len; k++)
                  printf("  %.3f", nn->convlayers[i].filters[j].W[k]);
                printf("\n");
                #endif
              }
            if((nn->convlayers[i].out = (double*)malloc(outputLen_Conv2D(nn->convlayers + i) * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate output array while reading Conv2DLayer[%d] from file\n", i);
                exit(1);
              }                                                     //  Allocate and blank out output array
            for(j = 0; j < outputLen_Conv2D(nn->convlayers + i); j++)
              nn->convlayers[i].out[j] = 0.0;

            len = nn->convlayers[i].n;
            if((nn->convlayers[i].f = (unsigned char*)malloc(len * sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to allocate function flag array for Conv2DLayer[%d] while reading from file\n", i);
                exit(1);
              }
            if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
                exit(1);
              }
            fread(ucharBuffer, sizeof(char), len, fp);              //  Read 'len' objects of size char into buffer
            for(j = 0; j < len; j++)                                //  Read function flag array for layer[i]
              nn->convlayers[i].f[j] = ucharBuffer[j];
            #ifdef __NEURON_DEBUG
            for(j = 0; j < len; j++)
              printf("  nn->convlayers[%d].f[%d] = %d\n", i, j, nn->convlayers[i].f[j]);
            #endif

            if((nn->convlayers[i].alpha = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate function auxiliaries array for Conv2DLayer[%d] while reading from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read function auxiliaries for layer[i]
              nn->convlayers[i].alpha[j] = doubleBuffer[j];
            #ifdef __NEURON_DEBUG
            for(j = 0; j < len; j++)
              printf("  nn->convlayers[%d].alpha[%d] = %.6f\n", i, j, nn->convlayers[i].alpha[j]);
            #endif

            if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
              {
                printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
                exit(1);
              }
            fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);   //  Read LAYER_NAME_LEN objects of size char into buffer
            for(j = 0; j < LAYER_NAME_LEN; j++)                     //  Read layer name
              nn->convlayers[i].name[j] = ucharBuffer[j];
            #ifdef __NEURON_DEBUG
            printf("  nn->convlayers[%d].name = %s\n", i, nn->convlayers[i].name);
            #endif
          }
      }

    if(nn->accumLen > 0)
      {
        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
            exit(1);
          }
                                                                    //  Allocate network's AccumLayer array
        if((nn->accumlayers = (AccumLayer*)malloc(nn->accumLen * sizeof(AccumLayer))) == NULL)
          {
            printf("ERROR: Unable to allocate AccumLayer array while reading from file\n");
            exit(1);
          }

        for(i = 0; i < nn->accumLen; i++)                           //  Read all Accumulator Layers from file
          {
            fread(uintBuffer, sizeof(int), 1, fp);                  //  Read 1 object of size int into buffer
            nn->accumlayers[i].i = uintBuffer[0];                   //  Read number of Accumulator inputs
            #ifdef __NEURON_DEBUG
            printf("  nn->accumlayers[%d].i = %d\n", i, nn->accumlayers[i].i);
            #endif

            if((nn->accumlayers[i].out = (double*)malloc(nn->accumlayers[i].i * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate output array while reading AccumLayer[%d] from file\n", i);
                exit(1);
              }
            for(j = 0; j < nn->accumlayers[i].i; j++)               //  Allocate and blank out output array
              nn->accumlayers[i].out[j] = 0.0;

            fread(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);   //  Read LAYER_NAME_LEN objects of size char into buffer
            for(j = 0; j < LAYER_NAME_LEN; j++)                     //  Read Accumulator layer name
              nn->accumlayers[i].name[j] = ucharBuffer[j];
            #ifdef __NEURON_DEBUG
            printf("  nn->accumlayers[%d].name = %s\n", i, nn->accumlayers[i].name);
            #endif
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
        if((uintBuffer = (unsigned int*)realloc(uintBuffer, 3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned int buffer while reading from file\n");
            exit(1);
          }
        for(i = 0; i < nn->lstmLen; i++)
          {
            fread(uintBuffer, sizeof(int), 3, fp);                  //  Read 3 objects of size int into buffer
            nn->lstmlayers[i].d = uintBuffer[0];                    //  Read input dimensionality for LSTMLayer[i] from buffer
            nn->lstmlayers[i].h = uintBuffer[1];                    //  Read state dimensionality for LSTMLayer[i] from buffer
            nn->lstmlayers[i].cache = uintBuffer[2];                //  Read state cache size for LSTMLayer[i] from buffer
            nn->lstmlayers[i].t = 0;                                //  Initialize time step to 0
            #ifdef __NEURON_DEBUG
            printf("  nn->lstmlayers[%d].d = %d\n", i, nn->lstmlayers[i].d);
            printf("  nn->lstmlayers[%d].h = %d\n", i, nn->lstmlayers[i].h);
            printf("  nn->lstmlayers[%d].cache = %d\n", i, nn->lstmlayers[i].cache);
            #endif

            len = nn->lstmlayers[i].d * nn->lstmlayers[i].h;        //  Allocate all things d*h
            if((nn->lstmlayers[i].Wi = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Wi while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].Wo = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Wo while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].Wf = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Wf while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].Wc = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Wc while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Wi weights from buffer
              nn->lstmlayers[i].Wi[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Wo weights from buffer
              nn->lstmlayers[i].Wo[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Wf weights from buffer
              nn->lstmlayers[i].Wf[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Wc weights from buffer
              nn->lstmlayers[i].Wc[j] = doubleBuffer[j];

            len = nn->lstmlayers[i].h * nn->lstmlayers[i].h;        //  Allocate all things h*h
            if((nn->lstmlayers[i].Ui = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Ui while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].Uo = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Uo while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].Uf = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Uf while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].Uc = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Uc while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Ui weights from buffer
              nn->lstmlayers[i].Ui[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Uo weights from buffer
              nn->lstmlayers[i].Uo[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Uf weights from buffer
              nn->lstmlayers[i].Uf[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s Uc weights from buffer
              nn->lstmlayers[i].Uc[j] = doubleBuffer[j];

            len = nn->lstmlayers[i].h;                              //  Allocate all things h
            if((nn->lstmlayers[i].bi = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate bias array bi while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].bo = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate bias array bo while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].bf = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate bias array bf while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].bc = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate bias array bc while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->lstmlayers[i].c = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate cell array while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s bi bias from buffer
              nn->lstmlayers[i].bi[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s bo bias from buffer
              nn->lstmlayers[i].bo[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s bf bias from buffer
              nn->lstmlayers[i].bf[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read LSTMLayer[i]'s bc bias from buffer
              nn->lstmlayers[i].bc[j] = doubleBuffer[j];
            for(j = 0; j < len; j++)                                //  Set vector c to zero-vector
              nn->lstmlayers[i].c[j] = 0.0;

            len = nn->lstmlayers[i].h * nn->lstmlayers[i].cache;    //  Allocate the output/state cache
            if((nn->lstmlayers[i].H = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate state cache while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            for(j = 0; j < len; j++)                                //  Blank out output matrix
              nn->lstmlayers[i].H[j] = 0.0;
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
        if((uintBuffer = (unsigned int*)realloc(uintBuffer, 3 * sizeof(int))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned int buffer while reading from file\n");
            exit(1);
          }
        for(i = 0; i < nn->gruLen; i++)
          {
            fread(uintBuffer, sizeof(int), 3, fp);                  //  Read 3 objects of size int into buffer
            nn->grulayers[i].d = uintBuffer[0];                     //  Read input dimensionality for GRULayer[i] from buffer
            nn->grulayers[i].h = uintBuffer[1];                     //  Read state dimensionality for GRULayer[i] from buffer
            nn->grulayers[i].cache = uintBuffer[2];                 //  Read state cache size for GRULayer[i] from buffer
            nn->grulayers[i].t = 0;                                 //  Initialize time step to 0
            #ifdef __NEURON_DEBUG
            printf("  nn->grulayers[%d].d = %d\n", i, nn->grulayers[i].d);
            printf("  nn->grulayers[%d].h = %d\n", i, nn->grulayers[i].h);
            printf("  nn->grulayers[%d].cache = %d\n", i, nn->grulayers[i].cache);
            #endif

            len = nn->grulayers[i].d * nn->grulayers[i].h;          //  Allocate all things d*h
            if((nn->grulayers[i].Wz = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Wz while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->grulayers[i].Wr = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Wr while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->grulayers[i].Wh = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Wh while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s Wz weights from buffer
              nn->grulayers[i].Wz[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s Wr weights from buffer
              nn->grulayers[i].Wr[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s Wh weights from buffer
              nn->grulayers[i].Wh[j] = doubleBuffer[j];

            len = nn->grulayers[i].h * nn->grulayers[i].h;          //  Allocate all things h*h
            if((nn->grulayers[i].Uz = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Uz while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->grulayers[i].Ur = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Ur while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->grulayers[i].Uh = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate weight array Uh while reading LSTMLayer[%d] from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s Uz weights from buffer
              nn->grulayers[i].Uz[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s Ur weights from buffer
              nn->grulayers[i].Ur[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s Uh weights from buffer
              nn->grulayers[i].Uh[j] = doubleBuffer[j];

            len = nn->grulayers[i].h;                               //  Allocate all things h
            if((nn->grulayers[i].bz = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate bias array bz while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->grulayers[i].br = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate bias array br while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((nn->grulayers[i].bh = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate bias array bh while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for reading from file\n");
                exit(1);
              }
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s bz bias from buffer
              nn->grulayers[i].bz[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s br bias from buffer
              nn->grulayers[i].br[j] = doubleBuffer[j];
            fread(doubleBuffer, sizeof(double), len, fp);           //  Read 'len' objects of size double into buffer
            for(j = 0; j < len; j++)                                //  Read GRULayer[i]'s bh bias from buffer
              nn->grulayers[i].bh[j] = doubleBuffer[j];

            len = nn->grulayers[i].h * nn->grulayers[i].cache;      //  Allocate the output/state cache
            if((nn->grulayers[i].H = (double*)malloc(len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to allocate state cache while reading GRULayer[%d] from file\n", i);
                exit(1);
              }
            for(j = 0; j < len; j++)                                //  Blank out output matrix
              nn->grulayers[i].H[j] = 0.0;
          }
      }

    if(nn->vars > 0)                                                //  Read all Variables
      {
        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, VARSTR_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for reading from file\n");
            exit(1);
          }
        for(i = 0; i < nn->vars; i++)                               //  Write all Variables to file
          {
            fread(ucharBuffer, sizeof(char), VARSTR_LEN, fp);       //  Read VARSTR_LEN objects of size char into buffer
            for(j = 0; j < VARSTR_LEN; j++)
              nn->variables[i].key[j] = ucharBuffer[j];
            #ifdef __NEURON_DEBUG
            printf("  nn->variables[%d].key = %s\n", i, nn->variables[i].key);
            #endif

            fread(doubleBuffer, sizeof(double), 1, fp);             //  Read buffer, write 1 object of size double
            nn->variables[i].value = doubleBuffer[0];
            #ifdef __NEURON_DEBUG
            printf("  nn->variables[%d].value = %.6f\n", i, nn->variables[i].value);
            #endif
          }
      }

    free(ucharBuffer);
    free(uintBuffer);
    free(doubleBuffer);
    fclose(fp);

    return true;
  }

/* Write the given Neural Network to a binary file named 'filename'. */
bool write_NN(char* filename, NeuralNet* nn)
  {
    FILE* fp;
    unsigned char* ucharBuffer;
    bool* boolBuffer;
    unsigned int* uintBuffer;
    double* doubleBuffer;
    unsigned int len;
    unsigned int i, j, k;

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
                                                                    //  Allocate 7 uints
    if((uintBuffer = (unsigned int*)malloc(7 * sizeof(int))) == NULL)
      {
        printf("ERROR: Unable to allocate unsigned int buffer for writing to file\n");
        exit(1);
      }
    if((doubleBuffer = (double*)malloc(sizeof(double))) == NULL)    //  Allocate 1 double
      {
        printf("ERROR: Unable to allocate double buffer for writing to file\n");
        exit(1);
      }

    uintBuffer[0] = nn->i;                                          //  Save NeuralNet input count to buffer
    uintBuffer[1] = nn->len;                                        //  Save NeuralNet edge count to buffer
    uintBuffer[2] = nn->denseLen;                                   //  Save number of Dense Layers to buffer
    uintBuffer[3] = nn->convLen;                                    //  Save number of Convolutional Layers to buffer
    uintBuffer[4] = nn->accumLen;                                   //  Save number of Accumulator Layers to buffer
    uintBuffer[5] = nn->lstmLen;                                    //  Save number of LSTM Layers to buffer
    uintBuffer[6] = nn->gruLen;                                     //  Save number of GRU Layers to buffer
    fwrite(uintBuffer, sizeof(int), 7, fp);                         //  From buffer, write 6 objects of size int
    #ifdef __NEURON_DEBUG
    printf("  uintBuffer[0] = nn->i        = %d\n", uintBuffer[0]);
    printf("  uintBuffer[1] = nn->len      = %d\n", uintBuffer[1]);
    printf("  uintBuffer[2] = nn->denseLen = %d\n", uintBuffer[2]);
    printf("  uintBuffer[3] = nn->convLen  = %d\n", uintBuffer[3]);
    printf("  uintBuffer[4] = nn->accumLen = %d\n", uintBuffer[4]);
    printf("  uintBuffer[5] = nn->lstmLen  = %d\n", uintBuffer[5]);
    printf("  uintBuffer[6] = nn->gruLen   = %d\n", uintBuffer[6]);
    #endif

    ucharBuffer[0] = nn->vars;                                      //  Save number of Variables to buffer
    fwrite(ucharBuffer, sizeof(char), 1, fp);                       //  From buffer, write 1 object of size char
    #ifdef __NEURON_DEBUG
    printf("  ucharBuffer[0] = nn->vars    = %d\n", ucharBuffer[0]);
    #endif

    uintBuffer[0] = nn->gen;                                        //  Save generation/epoch to buffer
    fwrite(uintBuffer, sizeof(int), 1, fp);                         //  From buffer, write 1 object of size int
    #ifdef __NEURON_DEBUG
    printf("  uintBuffer[0] = nn->gen      = %d\n", uintBuffer[0]);
    #endif

    doubleBuffer[0] = nn->fit;                                      //  Save fitness to buffer
    fwrite(doubleBuffer, sizeof(double), 1, fp);                    //  From buffer, write 1 object of size double
    #ifdef __NEURON_DEBUG
    printf("  doubleBuffer[0] = nn->fit    = %.6f\n", doubleBuffer[0]);
    #endif
                                                                    //  Write network comment to file
    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, COMMSTR_LEN * sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
        exit(1);
      }
    for(i = 0; i < COMMSTR_LEN; i++)
      ucharBuffer[i] = nn->comment[i];
    fwrite(ucharBuffer, COMMSTR_LEN, 1, fp);                        //  From buffer, write COMMSTR_LEN objects of size char
    #ifdef __NEURON_DEBUG
    printf("  ucharBuffer = %s\n  Edge List:\n", ucharBuffer);
    #endif
                                                                    //  Shrink buffer to 1
    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
        exit(1);
      }
    for(i = 0; i < nn->len; i++)                                    //  Write all Edges to file
      {
        ucharBuffer[0] = nn->edgelist[i].srcType;                   //  Save edge source type to buffer
        fwrite(ucharBuffer, sizeof(char), 1, fp);                   //  From buffer, write 1 object of size char
        #ifdef __NEURON_DEBUG
        printf("    ucharBuffer[0] = %d\n", ucharBuffer[0]);
        #endif

        uintBuffer[0] = nn->edgelist[i].srcIndex;                   //  Save edge source index to buffer
        uintBuffer[1] = nn->edgelist[i].selectorStart;              //  Save edge selector start to buffer
        uintBuffer[2] = nn->edgelist[i].selectorEnd;                //  Save edge selector end to buffer
        fwrite(uintBuffer, sizeof(int), 3, fp);                     //  From buffer, write 3 objects of size int
        #ifdef __NEURON_DEBUG
        printf("    uintBuffer[0] = %d\n", uintBuffer[0]);
        printf("    uintBuffer[1] = %d\n", uintBuffer[1]);
        printf("    uintBuffer[2] = %d\n", uintBuffer[2]);
        #endif

        ucharBuffer[0] = nn->edgelist[i].dstType;                   //  Save edge destination type to buffer
        fwrite(ucharBuffer, sizeof(char), 1, fp);                   //  From buffer, write 1 object of size char
        #ifdef __NEURON_DEBUG
        printf("    ucharBuffer[0] = %d\n", ucharBuffer[0]);
        #endif

        uintBuffer[0] = nn->edgelist[i].dstIndex;                   //  Save edge destination index to buffer
        fwrite(uintBuffer, sizeof(int), 1, fp);                     //  From buffer, write 1 object of size int
        #ifdef __NEURON_DEBUG
        printf("    uintBuffer[0] = %d\n\n", uintBuffer[0]);
        #endif
      }

    for(i = 0; i < nn->denseLen; i++)                               //  Write all Dense Layers to file
      {
        uintBuffer[0] = nn->denselayers[i].i;                       //  Save number of DenseLayer inputs to buffer
        uintBuffer[1] = nn->denselayers[i].n;                       //  Save number of DenseLayer units to buffer
        fwrite(uintBuffer, sizeof(int), 2, fp);                     //  From buffer, write 2 objects of size int
        #ifdef __NEURON_DEBUG
        printf("  uintBuffer[0] = %d\n", uintBuffer[0]);
        printf("  uintBuffer[1] = %d\n", uintBuffer[1]);
        #endif

        len = (nn->denselayers[i].i + 1) * nn->denselayers[i].n;
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Write weights as they exist in the buffer
          doubleBuffer[j] = nn->denselayers[i].W[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        if(i == 0)                                                  //  We haven't used the Bool buffer before
          {                                                         //  so consider the first case
            if((boolBuffer = (bool*)malloc(len * sizeof(bool))) == NULL)
              {
                printf("ERROR: Unable to allocate Boolean buffer for writing to file\n");
                exit(1);
              }
          }
        else
          {
            if((boolBuffer = (bool*)realloc(boolBuffer, len * sizeof(bool))) == NULL)
              {
                printf("ERROR: Unable to reallocate Boolean buffer for writing to file\n");
                exit(1);
              }
          }
        for(j = 0; j < len; j++)                                    //  When live, M matrix is 0.0 or 1.0.
          {                                                         //  When stored, M matrix is false or true
            if(nn->denselayers[i].M[j] == 0.0)                      //                           (masked) (unmasked)
              boolBuffer[j] = false;
            else
              boolBuffer[j] = true;
          }
        fwrite(boolBuffer, sizeof(bool), len, fp);                  //  From buffer, write 'len' objects of size bool
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          {
            if(nn->denselayers[i].M[j] == 0.0)
              printf("  boolBuffer[%d] = 0\n", j);
            else
              printf("  boolBuffer[%d] = 1\n", j);
          }
        #endif

        len = nn->denselayers[i].n;
        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, len * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Write function flags
          ucharBuffer[j] = nn->denselayers[i].f[j];
        fwrite(ucharBuffer, sizeof(char), len, fp);                 //  From buffer, write 'len' objects of size char
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  ucharBuffer[%d] = %d\n", j, ucharBuffer[j]);
        #endif

        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Write function auxiliaries
          doubleBuffer[j] = nn->denselayers[i].alpha[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)                         //  Write layer name
          ucharBuffer[j] = nn->denselayers[i].name[j];
        fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);      //  From buffer, write LAYER_NAME_LEN chars
        #ifdef __NEURON_DEBUG
        printf("  ucharBuffer = %s\n\n", ucharBuffer);
        #endif
      }
    if(nn->denseLen > 0)                                            //  We won't need the Boolean buffer again
      free(boolBuffer);

    if((uintBuffer = (unsigned int*)realloc(uintBuffer, 3 * sizeof(int))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned int buffer for writing to file\n");
        exit(1);
      }
    for(i = 0; i < nn->convLen; i++)                                //  Write all Convolutional Layers to file
      {
        uintBuffer[0] = nn->convlayers[i].inputW;                   //  Save Conv2DLayer input width to buffer
        uintBuffer[1] = nn->convlayers[i].inputH;                   //  Save Conv2DLayer input height to buffer
        uintBuffer[2] = nn->convlayers[i].n;                        //  Save number of Conv2DLayer filters to buffer
        fwrite(uintBuffer, sizeof(int), 3, fp);                     //  From buffer, write 3 objects of size int
        #ifdef __NEURON_DEBUG
        printf("  uintBuffer[0] = %d\n", uintBuffer[0]);
        printf("  uintBuffer[1] = %d\n", uintBuffer[1]);
        printf("  uintBuffer[2] = %d\n", uintBuffer[2]);
        #endif

        for(j = 0; j < nn->convlayers[i].n; j++)                    //  Write all convlayer[i]'s filters
          {
            uintBuffer[0] = nn->convlayers[i].filters[j].w;         //  Write dimensions
            uintBuffer[1] = nn->convlayers[i].filters[j].h;
            fwrite(uintBuffer, sizeof(int), 2, fp);                 //  From buffer, write 2 objects of size int
            #ifdef __NEURON_DEBUG
            printf("  uintBuffer[0] = %d\n", uintBuffer[0]);
            printf("  uintBuffer[1] = %d\n", uintBuffer[1]);
            #endif

            len = nn->convlayers[i].filters[j].w * nn->convlayers[i].filters[j].h + 1;
            if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
              {
                printf("ERROR: Unable to reallocate double buffer for writing to file\n");
                exit(1);
              }
            for(k = 0; k < len; k++)                                //  Write all filter weights and bias
              doubleBuffer[k] = nn->convlayers[i].filters[j].W[k];
            fwrite(doubleBuffer, sizeof(double), len, fp);          //  From buffer, write 'len' objects of size double
            #ifdef __NEURON_DEBUG
            for(k = 0; k < len; k++)
              printf("  doubleBuffer[%d] = %.6f\n", k, doubleBuffer[k]);
            #endif
          }

        len = nn->convlayers[i].n;
        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, len * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)
          ucharBuffer[j] = nn->convlayers[i].f[j];
        fwrite(ucharBuffer, sizeof(char), len, fp);                 //  From buffer, write 'len' objects of size char
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  ucharBuffer[%d] = %d\n", j, ucharBuffer[j]);
        #endif

        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)
          doubleBuffer[j] = nn->convlayers[i].alpha[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < LAYER_NAME_LEN; j++)
          ucharBuffer[j] = nn->convlayers[i].name[j];
        fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);      //  From buffer, write LAYER_NAME_LEN chars
        #ifdef __NEURON_DEBUG
        printf("  ucharBuffer = %s\n\n", ucharBuffer);
        #endif
      }

    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, LAYER_NAME_LEN * sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
        exit(1);
      }
    for(i = 0; i < nn->accumLen; i++)                               //  Write all Accumulator Layers to file
      {
        uintBuffer[0] = nn->accumlayers[i].i;                       //  Save number of AccumLayer inputs to buffer
        fwrite(uintBuffer, sizeof(int), 1, fp);                     //  From buffer, write 1 object of size int
        #ifdef __NEURON_DEBUG
        printf("  uintBuffer[0] = %d\n", uintBuffer[0]);
        #endif

        for(j = 0; j < LAYER_NAME_LEN; j++)
          ucharBuffer[j] = nn->accumlayers[i].name[j];
        fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);      //  From buffer, write LAYER_NAME_LEN chars
        #ifdef __NEURON_DEBUG
        printf("  ucharBuffer = %s\n\n", ucharBuffer);
        #endif
      }

    for(i = 0; i < nn->lstmLen; i++)                                //  Write all LSTM Layers to file
      {
        uintBuffer[0] = nn->lstmlayers[i].d;                        //  Save LSTMLayer input dimension to buffer
        uintBuffer[1] = nn->lstmlayers[i].h;                        //  Save LSTMLayer state dimension to buffer
        uintBuffer[2] = nn->lstmlayers[i].cache;                    //  Save LSTMLayer state cache size to buffer
        fwrite(uintBuffer, sizeof(int), 3, fp);                     //  From buffer, write 3 objects of size int
        #ifdef __NEURON_DEBUG
        printf("  uintBuffer[0] = %d\n", uintBuffer[0]);
        printf("  uintBuffer[1] = %d\n", uintBuffer[1]);
        printf("  uintBuffer[2] = %d\n", uintBuffer[2]);
        #endif

        len = nn->lstmlayers[i].d * nn->lstmlayers[i].h;            //  Write all d-by-h W matrices
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Wi weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Wi[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Wo weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Wo[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Wf weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Wf[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Wc weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Wc[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        len = nn->lstmlayers[i].h * nn->lstmlayers[i].h;            //  Write all h-by-h U matrices
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Ui weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Ui[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Uo weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Uo[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Uf weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Uf[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer Uc weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].Uc[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        len = nn->lstmlayers[i].h;                                  //  Write all h-length b vectors
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer bi weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].bi[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer bo weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].bo[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer bf weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].bf[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save LSTMLayer bc weights to buffer
          doubleBuffer[j] = nn->lstmlayers[i].bc[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        for(j = 0; j < LAYER_NAME_LEN; j++)
          ucharBuffer[j] = nn->lstmlayers[i].name[j];
        fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);      //  From buffer, write LAYER_NAME_LEN chars
        #ifdef __NEURON_DEBUG
        printf("  ucharBuffer = %s\n\n", ucharBuffer);
        #endif
      }

    for(i = 0; i < nn->gruLen; i++)                                 //  Write all GRU Layers to file
      {
        uintBuffer[0] = nn->grulayers[i].d;                         //  Save GRULayer input dimension to buffer
        uintBuffer[1] = nn->grulayers[i].h;                         //  Save GRULayer state dimension to buffer
        uintBuffer[2] = nn->grulayers[i].cache;                     //  Save GRULayer state cache size to buffer
        fwrite(uintBuffer, sizeof(int), 3, fp);                     //  From buffer, write 3 objects of size int
        #ifdef __NEURON_DEBUG
        printf("  uintBuffer[0] = %d\n", uintBuffer[0]);
        printf("  uintBuffer[1] = %d\n", uintBuffer[1]);
        printf("  uintBuffer[2] = %d\n", uintBuffer[2]);
        #endif

        len = nn->grulayers[i].d * nn->grulayers[i].h;              //  Write all d-by-h W matrices
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Save GRULayer Wz weights to buffer
          doubleBuffer[j] = nn->grulayers[i].Wz[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save GRULayer Wr weights to buffer
          doubleBuffer[j] = nn->grulayers[i].Wr[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save GRULayer Wh weights to buffer
          doubleBuffer[j] = nn->grulayers[i].Wh[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        len = nn->grulayers[i].h * nn->grulayers[i].h;              //  Write all h-by-h U matrices
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Save GRULayer Uz weights to buffer
          doubleBuffer[j] = nn->grulayers[i].Uz[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save GRULayer Ur weights to buffer
          doubleBuffer[j] = nn->grulayers[i].Ur[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save GRULayer Uh weights to buffer
          doubleBuffer[j] = nn->grulayers[i].Uh[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        len = nn->grulayers[i].h;                                   //  Write all h-length b vectors
        if((doubleBuffer = (double*)realloc(doubleBuffer, len * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to reallocate double buffer for writing to file\n");
            exit(1);
          }
        for(j = 0; j < len; j++)                                    //  Save GRULayer bz weights to buffer
          doubleBuffer[j] = nn->grulayers[i].bz[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save GRULayer br weights to buffer
          doubleBuffer[j] = nn->grulayers[i].br[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif
        for(j = 0; j < len; j++)                                    //  Save GRULayer bh weights to buffer
          doubleBuffer[j] = nn->grulayers[i].bh[j];
        fwrite(doubleBuffer, sizeof(double), len, fp);              //  From buffer, write 'len' objects of size double
        #ifdef __NEURON_DEBUG
        for(j = 0; j < len; j++)
          printf("  doubleBuffer[%d] = %.6f\n", j, doubleBuffer[j]);
        #endif

        for(j = 0; j < LAYER_NAME_LEN; j++)
          ucharBuffer[j] = nn->grulayers[i].name[j];
        fwrite(ucharBuffer, sizeof(char), LAYER_NAME_LEN, fp);      //  From buffer, write LAYER_NAME_LEN chars
        #ifdef __NEURON_DEBUG
        printf("  ucharBuffer = %s\n\n", ucharBuffer);
        #endif
      }

    if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, VARSTR_LEN * sizeof(char))) == NULL)
      {
        printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
        exit(1);
      }
    for(i = 0; i < nn->vars; i++)                                   //  Write all Variables to file
      {
        if((ucharBuffer = (unsigned char*)realloc(ucharBuffer, VARSTR_LEN * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to reallocate unsigned char buffer for writing to file\n");
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
      }

    free(ucharBuffer);
    free(uintBuffer);
    free(doubleBuffer);
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

    Edge* swap;                                                     //  Hold Edge objects to be moved
    unsigned int swaplen;

    #ifdef __NEURON_DEBUG
    printf("sortEdges()\n");
    #endif

    listlen = nn->denseLen + nn->convLen + nn->accumLen + nn->lstmLen + nn->gruLen + 1;
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
      }
    printf("\n");
    return;
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

    if(i * layer->h + j < layer->h * layer->d)
      layer->Wz[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Wr matrix */
void setWr_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWr_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->d)
      layer->Wr[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Wh matrix */
void setWh_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setWh_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->d)
      layer->Wh[i * layer->h + j] = w;
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

    if(i * layer->h + j < layer->h * layer->h)
      layer->Uz[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Ur matrix */
void setUr_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUr_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->h)
      layer->Ur[i * layer->h + j] = w;
    return;
  }

/* Set column[i], row[j] of the given layer, Uh matrix */
void setUh_ij_GRU(double w, unsigned int i, unsigned int j, GRULayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setUh_ij_GRU(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * layer->h + j < layer->h * layer->h)
      layer->Uh[i * layer->h + j] = w;
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
    for(i = 0; i < layer->d; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Wz[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Wr (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->d; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Wr[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Wh (%d x %d)\n", layer->h, layer->d);
    for(i = 0; i < layer->d; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Wh[i * layer->h + j]);
        printf(" ]\n");
      }

    printf("Uz (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Uz[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Ur (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Ur[i * layer->h + j]);
        printf(" ]\n");
      }
    printf("Uh (%d x %d)\n", layer->h, layer->h);
    for(i = 0; i < layer->h; i++)
      {
        printf("[");
        for(j = 0; j < layer->h; j++)
          printf(" %.5f", layer->Uh[i * layer->h + j]);
        printf(" ]\n");
      }

    printf("bz (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->bz[i]);
    printf("br (%d x 1)\n", layer->h);
    for(i = 0; i < layer->h; i++)
      printf("[ %.5f ]\n", layer->br[i]);
    printf("bh (%d x 1)\n", layer->h);
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

    order = CblasColMajor;
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
          ht_1[n] = layer->H[ t_1 * layer->h + n ];
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
              layer->H[(m - 1) * layer->h + n] = layer->H[m * layer->h + n];
          }
      }

    for(n = 0; n < layer->h; n++)
      {
                                                                    //  h = z*ht_1 + (1-z)*tanh(h)
        layer->H[ t * layer->h + n ] = z[n] * ht_1[n] + (1.0 - z[n]) * ((2.0 / (1.0 + pow(M_E, -2.0 * h[n]))) - 1.0);
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

    nn->convlayers[nn->convLen - 1].inputW = inputW;                //  Set this newest layer's input dimentions
    nn->convlayers[nn->convLen - 1].inputH = inputH;
    nn->convlayers[nn->convLen - 1].n = 0;                          //  New layer initially contains zero filters
    for(i = 0; i < LAYER_NAME_LEN; i++)                             //  Blank out layer name
      nn->convlayers[nn->convLen - 1].name[i] = '\0';

    return nn->convLen;
  }

/* Add a Filter2D to an existing Conv2DLayer.
   The new filter shall have dimensions 'filterW' by 'filterH'. */
unsigned int add_Conv2DFilter(unsigned int filterW, unsigned int filterH, Conv2DLayer* convlayer)
  {
    unsigned int i;
    unsigned int ctr;

    #ifdef __NEURON_DEBUG
    printf("add_Conv2DFilter(%d, %d)\n", filterW, filterH);
    #endif

    convlayer->n++;                                                 //  Increment the number of filters/units
    if(convlayer->n == 1)
      {
                                                                    //  Allocate filter in 'filters' array
        if((convlayer->filters = (Filter2D*)malloc(sizeof(Filter2D))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer's filters array\n");
            exit(1);
          }
                                                                    //  Allocate this layer's function-flag array
        if((convlayer->f = (unsigned char*)malloc(sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer's function-flag array\n");
            exit(1);
          }
                                                                    //  Allocate this layer's function-parameter array
        if((convlayer->alpha = (double*)malloc(sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer's function-parameter array\n");
            exit(1);
          }
                                                                    //  Allocate this layer's output array
        if((convlayer->out = (double*)malloc((convlayer->inputW - filterW + 1) *
                                             (convlayer->inputH - filterH + 1) * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate Conv2DLayer's internal output array\n");
            exit(1);
          }
      }
    else
      {
                                                                    //  Allocate another filter in 'filters' array
        if((convlayer->filters = (Filter2D*)realloc(convlayer->filters, convlayer->n * sizeof(Filter2D))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Conv2DLayer's filters array\n");
            exit(1);
          }
        if((convlayer->f = (unsigned char*)realloc(convlayer->f, convlayer->n * sizeof(char))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Conv2DLayer's function-flag array\n");
            exit(1);
          }
                                                                    //  Allocate this newest layer's function-parameter array
        if((convlayer->alpha = (double*)realloc(convlayer->alpha, convlayer->n * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Conv2DLayer's function-parameter array\n");
            exit(1);
          }
        ctr = 0;                                                    //  Count output length attributable to previous filters
        for(i = 0; i < convlayer->n - 1; i++)
          ctr += (convlayer->inputW - convlayer->filters[i].w + 1) * (convlayer->inputH - convlayer->filters[i].h + 1);
        if((convlayer->out = (double*)realloc(convlayer->out,       //  Allocate this layer's output array
                                              (ctr + (convlayer->inputW - filterW + 1)  *
                                                     (convlayer->inputH - filterH + 1)) * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to re-allocate Conv2DLayer's internal output array\n");
            exit(1);
          }
      }

    convlayer->filters[convlayer->n - 1].w = filterW;               //  Set newest filter's dimensions
    convlayer->filters[convlayer->n - 1].h = filterH;
                                                                    //  Allocate the filter matrix plus bias
    if((convlayer->filters[convlayer->n - 1].W = (double*)malloc((filterW * filterH + 1) * sizeof(double))) == NULL)
      {
        printf("ERROR: Unable to allocate Conv2DLayer's filter\n");
        exit(1);
      }
    for(i = 0; i < filterW * filterH; i++)                          //  Generate random numbers in [ -1.0, 1.0 ]
      convlayer->filters[convlayer->n - 1].W[i] = -1.0 + ((double)rand() / ((double)RAND_MAX * 0.5));
    convlayer->filters[convlayer->n - 1].W[i] = 0.0;                //  Defaut bias = 0.0

    convlayer->f[convlayer->n - 1] = RELU;                          //  Default to ReLU
    convlayer->alpha[convlayer->n - 1] = 1.0;                       //  Default to 1.0

    return convlayer->n;
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

/* Set the activation function for unit[i] of the given layer */
void setF_i_Conv2D(unsigned char func, unsigned int i, Conv2DLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setF_i_Conv2D(%d, %d)\n", func, i);
    #endif

    if(i < layer->n)
      layer->f[i] = func;
    return;
  }

/* Set the activation function parameter for unit[i] of the given layer */
void setA_i_Conv2D(double a, unsigned int i, Conv2DLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setA_i_Conv2D(%f, %d)\n", a, i);
    #endif

    if(i < layer->n)
      layer->alpha[i] = a;
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
        printf("\n");
        printf("  Param: %.5f\n", layer->alpha[i]);
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
      ctr += (layer->inputW - layer->filters[i].w + 1) * (layer->inputH - layer->filters[i].h + 1);

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
    unsigned int outlen = 0;                                        //  Length of the output vector
    double* cache;                                                  //  Output array for a single filter
    double softmaxdenom;
    double val;

    #ifdef __NEURON_DEBUG
    printf("run_Conv2D(%d, %d)\n", layer->inputW, layer->inputH);
    #endif

    for(i = 0; i < layer->n; i++)                                   //  Add up the outputs for each filter, given the input size
      outlen += (layer->inputW - layer->filters[i].w + 1) * (layer->inputH - layer->filters[i].h + 1);

    for(i = 0; i < layer->n; i++)                                   //  For each filter
      {
        c = 0;
        softmaxdenom = 0.0;
        if((cache = (double*)malloc((layer->inputW - layer->filters[i].w + 1) * (layer->inputH - layer->filters[i].h + 1) * sizeof(double))) == NULL)
          {
            printf("ERROR: Unable to allocate filter output buffer\n");
            exit(1);
          }

        for(y = 0; y <= layer->inputH - layer->filters[i].h; y++)
          {
            for(x = 0; x <= layer->inputW - layer->filters[i].w; x++)
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
            switch(layer->f[i])
              {
                case RELU:                 layer->out[o] = (cache[s] > 0.0) ? cache[s] : 0.0;
                                           break;
                case LEAKY_RELU:           layer->out[o] = (cache[s] > cache[s] * layer->alpha[i]) ? cache[s] : layer->alpha[i];
                                           break;
                case SIGMOID:              layer->out[o] = 1.0 / (1.0 + pow(M_E, -cache[s] * layer->alpha[i]));
                                           break;
                case HYPERBOLIC_TANGENT:   layer->out[o] = (2.0 / (1.0 + pow(M_E, -2.0 * cache[s] * layer->alpha[i]))) - 1.0;
                                           break;
                case SOFTMAX:              layer->out[o] = pow(M_E, cache[s]) / softmaxdenom;
                                           break;
                case SYMMETRICAL_SIGMOID:  layer->out[o] = (1.0 - pow(M_E, -cache[s] * layer->alpha[i])) / (1.0 + pow(M_E, -cache[s] * layer->alpha[i]));
                                           break;
                case THRESHOLD:            layer->out[o] = (cache[s] > layer->alpha[i]) ? 1.0 : 0.0;
                                           break;
                                                                    //  (Includes LINEAR)
                default:                   layer->out[o] = cache[s] * layer->alpha[i];
              }
            o++;
          }

        free(cache);                                                //  Release the cache for this filter
      }

    return outlen;
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

/* Set entirety of layer's weight matrix.
   Input buffer 'w' is expected to be ROW-MAJOR
   (because it gets transposed by BLAS)
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
      layer->W[i * (layer->i + 1) + j] = w[j];
    return;
  }

/* Set unit[i], weight[j] of the given layer */
void setW_ij_Dense(double w, unsigned int i, unsigned int j, DenseLayer* layer)
  {
    #ifdef __NEURON_DEBUG
    printf("setW_ij_Dense(%f, %d, %d)\n", w, i, j);
    #endif

    if(i * (layer->i + 1) + j < (layer->i + 1) * layer->n)
      layer->W[i * (layer->i + 1) + j] = w;
    return;
  }

/* Set entirety of layer's mask matrix */
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
          layer->M[i * (layer->i + 1) + j] = 1.0;
        else
          layer->M[i * (layer->i + 1) + j] = 0.0;
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

    if(i * (layer->i + 1) + j < (layer->i + 1) * layer->n)
      {
        if(m)
          layer->M[i * (layer->i + 1) + j] = 1.0;
        else
          layer->M[i * (layer->i + 1) + j] = 0.0;
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
            case LEAKY_RELU:           layer->out[i] = (layer->out[i] > layer->out[i] * layer->alpha[i]) ? layer->out[i] : layer->alpha[i];
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