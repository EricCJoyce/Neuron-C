#include "neuron.h"

/*
Issue the following commands to compile and run

gcc -c -Wall xor.c
gfortran -o xor xor.o cblas_LINUX.a libblas.a
./xor
*/

int main(int argc, char* argv[])
  {
    NeuralNet* nn;
    double in[2];
    double* out;

    init_NN(&nn, 2);                                                //  Initialize a network that receives two inputs

    add_Dense(2, 2, nn);                                            //  Add dense layer (DENSE_ARRAY, 0)
    setName_Dense("Dense-1", nn->denselayers);                      //  Name the first dense layer
    add_Dense(2, 1, nn);                                            //  Add dense layer (DENSE_ARRAY, 1)
    setName_Dense("Dense-2", nn->denselayers + 1);                  //  Name the second dense layer
    linkLayers(INPUT_ARRAY, 0, 0, 2, DENSE_ARRAY, 0, nn);           //  Connect input to dense[0]
    linkLayers(DENSE_ARRAY, 0, 0, 2, DENSE_ARRAY, 1, nn);           //  Connect input to dense[1]

    setW_Dense( 4.169506539262890, 0, 0, nn->denselayers);          //  Set layer 0, unit 0, weight 0
    setW_Dense( 4.175620772246105, 0, 1, nn->denselayers);          //  Set layer 0, unit 0, weight 1
    setW_Dense(-6.399885541033798, 0, 2, nn->denselayers);          //  Set layer 0, unit 0, weight 2 (bias)
    setF_Dense(SIGMOID, 0, nn->denselayers);                        //  Set sigmoid function, layer 0, unit 0

    setW_Dense( 6.166518349083749, 1, 0, nn->denselayers);          //  Set layer 0, unit 1, weight 0
    setW_Dense( 6.187965760394095, 1, 1, nn->denselayers);          //  Set layer 0, unit 1, weight 1
    setW_Dense(-2.678140646720913, 1, 2, nn->denselayers);          //  Set layer 0, unit 1, weight 2 (bias)
    setF_Dense(SIGMOID, 1, nn->denselayers);                        //  Set sigmoid function, layer 0, unit 1

    print_Dense(nn->denselayers);                                   //  Show me the layer

    setW_Dense(-9.175274710095412, 0, 0, nn->denselayers + 1);      //  Set layer 1, unit 0, weight 0
    setW_Dense( 8.486130185157748, 0, 1, nn->denselayers + 1);      //  Set layer 1, unit 0, weight 1
    setW_Dense(-3.875273098510313, 0, 2, nn->denselayers + 1);      //  Set layer 1, unit 0, weight 2 (bias)
    setF_Dense(SIGMOID, 0, nn->denselayers + 1);                    //  Set sigmoid function, layer 1, unit 0

    print_Dense(nn->denselayers + 1);                               //  Show me the layer
    printf("\n");
    sortEdges(nn);                                                  //  Put edges in order of operation
    printEdgeList(nn);                                              //  Show me the results
    print_NN(nn);                                                   //  Show me the network

    write_NN("xor.nn", nn);                                         //  Write the network to file

    in[0] = 1.0;                                                    //  Now let's test it:
    in[1] = 0.0;                                                    //  This case should output a signal close to 1.0
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 1.0;                                                    //  This case should output a signal close to 0.0
    in[1] = 1.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 0.0;                                                    //  Should be close to 0.0
    in[1] = 0.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    in[0] = 0.0;                                                    //  Should be close to 1.0
    in[1] = 1.0;
    run_NN(in, nn, &out);
    printf("%f %f\n", in[0], in[1]);
    printf("%f\n\n", out[0]);

    free_NN(nn);                                                    //  Destroy the current copy of the network

    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

    init_NN(&nn, 2);                                                //  Always initialize before you load from file
    load_NN("xor.nn", nn);                                          //  Load the file we just wrote

    in[0] = 1.0;                                                    //  Test it again: outputs should be identical
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
