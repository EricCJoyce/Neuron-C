#include "neuron.h"

/*
cblas_LINUX.a, libblas.a, and cblas.h need to be in the same directory as neuron.h and xor.c.
This program creates a network file named "xor.nn".
Issue the following commands to compile and run. 
*/

int main(int argc, char* argv[])
  {
    NeuralNet* nn;
    double in[2];
    double* out;

    init_NN(&nn, 2);                                                //  Initialize for two inputs

    add_Dense(2, 2, nn);                                            //  Add dense layer (DENSE_ARRAY, 0)
    setName_Dense("Dense-1", nn->denselayers);                      //  Name the first dense layer
    add_Dense(2, 1, nn);                                            //  Add dense layer (DENSE_ARRAY, 1)
    setName_Dense("Dense-2", nn->denselayers + 1);                  //  Name the second dense layer
    linkLayers(INPUT_ARRAY, 0, 0, 2, DENSE_ARRAY, 0, nn);           //  Connect input to dense[0]
    linkLayers(DENSE_ARRAY, 0, 0, 2, DENSE_ARRAY, 1, nn);           //  Connect input to dense[1]

    setW_ij_Dense( 4.169506539262890, 0, 0, nn->denselayers);       //  Set unit 0, weight 0 of layer 0
    setW_ij_Dense( 4.175620772246105, 0, 1, nn->denselayers);       //  Set unit 0, weight 1 of layer 0
    setW_ij_Dense(-6.399885541033798, 0, 2, nn->denselayers);       //  Set unit 0, weight 2 (bias) of layer 0
    setF_i_Dense(SIGMOID, 0, nn->denselayers);                      //  Set activation function of unit 0, layer 0

    setW_ij_Dense( 6.166518349083749, 1, 0, nn->denselayers);       //  Set unit 1, weight 0 of layer 0
    setW_ij_Dense( 6.187965760394095, 1, 1, nn->denselayers);       //  Set unit 1, weight 1 of layer 0
    setW_ij_Dense(-2.678140646720913, 1, 2, nn->denselayers);       //  Set unit 1, weight 2 (bias) of layer 0
    setF_i_Dense(SIGMOID, 1, nn->denselayers);                      //  Set activation function of unit 1, layer 0

    print_Dense(nn->denselayers);                                   //  Show me the layer we just built

    setW_ij_Dense(-9.175274710095412, 0, 0, nn->denselayers + 1);   //  Set unit 0, weight 0 of layer 1
    setW_ij_Dense( 8.486130185157748, 0, 1, nn->denselayers + 1);   //  Set unit 0, weight 1 of layer 1
    setW_ij_Dense(-3.875273098510313, 0, 2, nn->denselayers + 1);   //  Set unit 0, weight 2 (bias) of layer 1
    setF_i_Dense(SIGMOID, 0, nn->denselayers + 1);                  //  Set activation function of unit 0, layer 1

    print_Dense(nn->denselayers + 1);                               //  Show me the layer we just built
    printf("\n");
    sortEdges(nn);
    printEdgeList(nn);                                              //  Show me all the connections
    print_NN(nn);                                                   //  Summarize the network

    write_NN("xor.nn", nn);                                         //  Write the network to file

    in[0] = 1.0;                                                    //  Now let's try the network out.
    in[1] = 0.0;                                                    //  This input should produce a signal close to 1.0
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 1.0;                                                    //  This input should produce a signal close to 0.0
    in[1] = 1.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 0.0;                                                    //  This input should produce a signal close to 0.0
    in[1] = 0.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 0.0;                                                    //  This input should produce a signal close to 1.0
    in[1] = 1.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    free_NN(nn);                                                    //  Destroy the network

    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

    init_NN(&nn, 2);                                                //  Always initialize before you load
    load_NN("xor.nn", nn);                                          //  Load the network we just wrote to file

    in[0] = 1.0;                                                    //  Check that these produce the same outputs
    in[1] = 0.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 1.0;
    in[1] = 1.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 0.0;
    in[1] = 0.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 0.0;
    in[1] = 1.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    free_NN(nn);                                                    //  Clean up
    return 0;                                                       //  Go home
  }
