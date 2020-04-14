#include "neuron.h"

/*
  Eric C. Joyce, Stevens Institute of Technology, 2020

  Your C code will necessarily exhibit knowledge of the Keras model you exported to weights.
  The command-line argument argv[1] is assumed to be the directory containing these weights.

  gcc -c -Wall build_neuron_model.c
  gfortran -o build_neuron_model build_neuron_model.o cblas_LINUX.a libblas.a
  ./build_neuron_model 01
*/

unsigned int readWeights(char*, double**);

int main(int argc, char* argv[])
  {
    NeuralNet* nn;
    double* w = NULL;
    char buffer[256];
    unsigned char len;
    unsigned char i;

    init_NN(&nn, 784);                                              //  Initialize for input 784-vec

    /******************************************************************************/
    /***************************************************************    C O N V 2 */
    add_Conv2D(28, 28, nn);                                         //  Add a Conv2D layer that receives the input: 28 x 28
    setName_Conv2D("Conv2D-1", nn->convlayers);                     //  Name the Conv2D layer
    //////////////////////////////////////////////////////////////////  Add 8 (3 x 3) kernels, each = 10 weights
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][0]
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][1]
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][2]
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][3]
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][4]
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][5]
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][6]
    add_Conv2DFilter(3, 3, nn->convlayers);                         //  filter[0][7]

    for(i = 0; i < 8; i++)
      {
        len = sprintf(buffer, "%s/Conv2D-%d.weights", argv[1], i);
        buffer[len] = '\0';
        readWeights(buffer, &w);
        setW_i_Conv2D(w, i, nn->convlayers);                        //  Set weights for filter[0][i]
        free(w);
      }

    //////////////////////////////////////////////////////////////////  Add 16 (5 x 5) kernels, each = 26 weights
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][8]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][9]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][10]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][11]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][12]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][13]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][14]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][15]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][16]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][17]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][18]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][19]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][20]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][21]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][22]
    add_Conv2DFilter(5, 5, nn->convlayers);                         //  filter[0][23]

    for(i = 8; i < 24; i++)
      {
        len = sprintf(buffer, "%s/Conv2D-%d.weights", argv[1], i);
        buffer[len] = '\0';
        readWeights(buffer, &w);
        setW_i_Conv2D(w, i, nn->convlayers);                        //  Set weights for filter[0][i]
        free(w);
      }

    /******************************************************************************/
    /***************************************************************    A C C U M */
    add_Accum(14624, nn);                                           //  Add accumulator layer (ACCUM_ARRAY, 0)
    setName_Accum("Accum-1", nn->accumlayers);                      //  Name the first accumulator layer

    /******************************************************************************/
    /***************************************************************    D E N S E */
    add_Dense(14624, 100, nn);                                      //  Add dense layer (DENSE_ARRAY, 0)
    setName_Dense("Dense-0", nn->denselayers);                      //  Name the first dense layer
    len = sprintf(buffer, "%s/Dense-0.weights", argv[1]);
    buffer[len] = '\0';
    readWeights(buffer, &w);
    setW_Dense(w, nn->denselayers);
    free(w);

    add_Dense(100, 10, nn);                                         //  Add dense layer (DENSE_ARRAY, 1)
    setName_Dense("Dense-1", nn->denselayers + 1);                  //  Name the third dense layer
    len = sprintf(buffer, "%s/Dense-1.weights", argv[1]);
    buffer[len] = '\0';
    readWeights(buffer, &w);
    setW_Dense(w, nn->denselayers + 1);
    free(w);

    setF_i_Dense(SOFTMAX, 0, nn->denselayers + 1);                  //  Set output layer's activation function to softmax
    setF_i_Dense(SOFTMAX, 1, nn->denselayers + 1);                  //  (Because we allow units to have different activation functions
    setF_i_Dense(SOFTMAX, 2, nn->denselayers + 1);                  //  you have to set SOFTMAX for all output nodes.)
    setF_i_Dense(SOFTMAX, 3, nn->denselayers + 1);
    setF_i_Dense(SOFTMAX, 4, nn->denselayers + 1);
    setF_i_Dense(SOFTMAX, 5, nn->denselayers + 1);
    setF_i_Dense(SOFTMAX, 6, nn->denselayers + 1);
    setF_i_Dense(SOFTMAX, 7, nn->denselayers + 1);
    setF_i_Dense(SOFTMAX, 8, nn->denselayers + 1);
    setF_i_Dense(SOFTMAX, 9, nn->denselayers + 1);

    /******************************************************************************/
    /******************************************************************************/
    /******************************************************************************/

    if(!linkLayers(INPUT_ARRAY, 0, 0, 784, CONV2D_ARRAY, 0, nn))    //  Connect input to conv2d[0]
      printf(">>>                Link[0] failed\n");
    if(!linkLayers(CONV2D_ARRAY, 0, 0, 14624, ACCUM_ARRAY, 0, nn))  //  Connect conv2d[0] to accum[0]
      printf(">>>                Link[1] failed\n");
    if(!linkLayers(ACCUM_ARRAY, 0, 0, 14624, DENSE_ARRAY, 0, nn))   //  Connect accum[0] to dense[0]
      printf(">>>                Link[2] failed\n");
    if(!linkLayers(DENSE_ARRAY, 0, 0, 100, DENSE_ARRAY, 1, nn))     //  Connect dense[0] to dense[1]
      printf(">>>                Link[3] failed\n");

    sortEdges(nn);
    printEdgeList(nn);
    printf("\n\n");
    print_NN(nn);

    len = sprintf(buffer, "mnist.nn");
    buffer[len] = '\0';

    write_NN(buffer, nn);
    free_NN(nn);

    return 0;
  }

/* Open the given file, read its weights into the given 'buffer,' and return the length of 'buffer.' */
unsigned int readWeights(char* filename, double** buffer)
  {
    FILE* fp;
    unsigned int len = 0;
    double x;

    printf("Reading %s:", filename);

    if((fp = fopen(filename, "rb")) == NULL)
      {
        printf("ERROR: Unable to open file\n");
        exit(1);
      }
    fseek(fp, 0, SEEK_SET);                                         //  Rewind
    while(!feof(fp))
      {
        if(fread(&x, sizeof(double), 1, fp) == 1)
          {
            if(++len == 1)
              {
                if(((*buffer) = (double*)malloc(sizeof(double))) == NULL)
                  {
                    printf("ERROR: Unable to malloc buffer\n");
                    exit(1);
                  }
              }
            else
              {
                if(((*buffer) = (double*)realloc((*buffer), len * sizeof(double))) == NULL)
                  {
                    printf("ERROR: Unable to realloc buffer\n");
                    exit(1);
                  }
              }
            (*buffer)[len - 1] = x;
          }
      }
    printf(" %d weights\n", len);
    fclose(fp);                                                     //  Close the file

    return len;
  }
