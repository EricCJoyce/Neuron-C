#include "neuron.h"

/*

gcc -c -Wall xortest.c
gfortran -o xortest xortest.o cblas_LINUX.a libblas.a
./xortest

*/

int main(int argc, char* argv[])
  {
    NeuralNet* nn;
    double in[2];
    double* out;

    srand(time(NULL));                                              //  Seed randomizer

    init_NN(&nn, 2);                                                //  Initialize for two inputs

    add_Dense(2, 2, nn);                                            //  Add dense layer (DENSE_ARRAY, 0)
    setName_Dense("Dense-1", nn->denselayers);                      //  Name the first dense layer
    add_Dense(2, 1, nn);                                            //  Add dense layer (DENSE_ARRAY, 1)
    setName_Dense("Dense-2", nn->denselayers + 1);                  //  Name the second dense layer
    linkLayers(INPUT_ARRAY, 0, 0, 2, DENSE_ARRAY, 0, nn);           //  Connect input to dense[0]
    linkLayers(DENSE_ARRAY, 0, 0, 2, DENSE_ARRAY, 1, nn);           //  Connect input to dense[1]

    setW_ij_Dense( 4.169506539262890, 0, 0, nn->denselayers);       //  Set layer 0, unit 0, weight 0
    setW_ij_Dense( 4.175620772246105, 0, 1, nn->denselayers);       //  Set layer 0, unit 0, weight 1
    setW_ij_Dense(-6.399885541033798, 0, 2, nn->denselayers);       //  Set layer 0, unit 0, weight 2 (bias)
    setF_i_Dense(SIGMOID, 0, nn->denselayers);                      //  Set sigmoid function, layer 0, unit 0

    setW_ij_Dense( 6.166518349083749, 1, 0, nn->denselayers);       //  Set layer 0, unit 1, weight 0
    setW_ij_Dense( 6.187965760394095, 1, 1, nn->denselayers);       //  Set layer 0, unit 1, weight 1
    setW_ij_Dense(-2.678140646720913, 1, 2, nn->denselayers);       //  Set layer 0, unit 1, weight 2 (bias)
    setF_i_Dense(SIGMOID, 1, nn->denselayers);                      //  Set sigmoid function, layer 0, unit 1

    print_Dense(nn->denselayers);

    setW_ij_Dense(-9.175274710095412, 0, 0, nn->denselayers + 1);   //  Set layer 1, unit 0, weight 0
    setW_ij_Dense( 8.486130185157748, 0, 1, nn->denselayers + 1);   //  Set layer 1, unit 0, weight 1
    setW_ij_Dense(-3.875273098510313, 0, 2, nn->denselayers + 1);   //  Set layer 1, unit 0, weight 2 (bias)
    setF_i_Dense(SIGMOID, 0, nn->denselayers + 1);                  //  Set sigmoid function, layer 1, unit 0

    print_Dense(nn->denselayers + 1);
    printf("\n");
    sortEdges(nn);
    printEdgeList(nn);
    print_NN(nn);

    write_NN("xor.nn", nn);

    in[0] = 1.0;
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

    free_NN(nn);

    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

    init_NN(&nn, 2);
    load_NN("xor.nn", nn);

    in[0] = 1.0;
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

    free_NN(nn);

    return 0;
  }
