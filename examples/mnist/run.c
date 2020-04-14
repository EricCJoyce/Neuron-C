#include "neuron.h"

/*
  Eric C. Joyce, Stevens Institute of Technology, 2020

  Read one of the 28-by-28 MNIST PGM files to be identified.
  Convert its unsigned char values in [0, 255] to a double in [0.0, 1.0].
  Input this floating-point buffer to the neural network.
  Print each of the 10 output values.

  gcc -c -Wall run.c
  gfortran -o run run.o cblas_LINUX.a libblas.a
  ./run mnist.nn samples/sample_1.pgm
*/

/**************************************************************************************************
 Typedefs  */

typedef struct P5Type
  {
    unsigned int w;                                                 //  Width of source image
    unsigned int h;                                                 //  Height of source image
    unsigned char maxGray;                                          //  Maximum grayscale value
    unsigned char* buffer;                                          //  The pixel data
    unsigned int buflen;                                            //  Length of pixel data array
  } P5;

/**************************************************************************************************
 Prototypes  */

bool readImg(char*, P5*);

int main(int argc, char* argv[])
  {
    NeuralNet* nn;
    P5* img = NULL;                                                 //  The source image, read from file
    double* x;
    double* out;
    unsigned int i;

    init_NN(&nn, 784);                                              //  Initialize
    load_NN(argv[1], nn);                                           //  Load the network

    if((img = (P5*)malloc(sizeof(P5))) == NULL)                     //  Allocate source image structure
      {
        printf("Unable to allocate source P5 image struct.\n");
        return 1;
      }
    img->buflen = 0;                                                //  Initialize image buffer length to zero

    if(!readImg(argv[2], img))                                      //  Read source file into structure
      {
        printf("ERROR: Unable to load PGM file.\n");
        return 1;
      }
    if((x = (double*)malloc(784 * sizeof(double))) == NULL)
      {
        if(img->buflen > 0)
          free(img->buffer);
        free(img);
        free_NN(nn);
      }
    for(i = 0; i < 784; i++)                                        //  Convert to floating point
      x[i] = (double)img->buffer[i] / 255.0;

    run_NN(x, nn, &out);                                            //  Run the network

    for(i = 0; i < 10; i++)
      printf("%d\t%f\n", i, out[i]);

    if(img->buflen > 0)                                             //  Clean up
      free(img->buffer);
    free(img);
    free_NN(nn);
    free(out);
    free(x);

    return 0;
  }

/* Read data from the given file into the given image structure */
bool readImg(char* filename, P5* img)
  {
    FILE* fp;                                                       //  File pointer
    char* line;                                                     //  File-reading buffer
    char* parse;                                                    //  Split strings
    bool readingW = true;                                           //  Width and height are on same line in file:
                                                                    //  track which one we're reading
    unsigned char linectr = 0;                                      //  Track which line we're reading (there may be more than
                                                                    //  255 lines, but we only care about the first few.
    size_t len = 0;                                                 //  Size of each line from file
    size_t read;                                                    //  Size of bytes read from file
    unsigned int i;                                                 //  Loop iterator

    if((fp = fopen(filename, "r")) == NULL)                         //  Open file for reading
      return false;                                                 //  Quit
                                                                    //  Read file line by line
    while((read = getline(&line, &len, fp)) != -1)
      {
        //  The P5 format requires that the header have three lines before pixel data begin:
        //  The "Magic Number," the identifier "P5" must be first;
        //  The image width and height, on a single line, separated by a space;
        //  The maximum grayscale value, on its own line.
        //  Comments may or may not appear in the header, beginning with '#'.
        //  If we're already counted off the three pieces of info we need, then assume that
        //  anything beginning with '#' is pixel data.
        if(strcmp(line, "P5\n") == 0)                               //  Count off "Magic Number" P5 (plus carriage return)
          linectr++;                                                //  Set up the next meaningful piece of header info
        else if(linectr == 1 && line[0] != '#')                     //  Retrieve width and height
          {
            parse = strtok(line, " ");                              //  Split line across space
            while(parse != NULL)                                    //  Retrieve segments until none remain
              {
                if(readingW)                                        //  Are we reading width? Store accordingly
                  {
                    img->w = (unsigned int)atoi(parse);
                    readingW = false;                               //  Disable this case
                  }
                else                                                //  If not width, we're reading height
                  img->h = (unsigned int)atoi(parse);
                parse = strtok(NULL, " ");                          //  Get the next slice
              }
            linectr++;                                              //  Field retrieved: set up the next.
          }
        else if(linectr == 2 && line[0] != '#')                     //  Retrieve max gray
          {
            img->maxGray = (unsigned char)atoi(line);
            linectr++;                                              //  Field retrieved: set up the next.
          }
        else if(linectr >= 3)                                       //  Read into buffer
          {
            if(img->buflen == 0)                                    //  First read: malloc
              {                                                     //  Attempt to allocate
                if((img->buffer = (unsigned char*)malloc(read)) == NULL)
                  return false;
              }
            else                                                    //  Subsequent reads: realloc
              {                                                     //  Attempt to reallocate
                if((img->buffer = (unsigned char*)realloc(img->buffer,
                                                          read + img->buflen * sizeof(char))) == NULL)
                  return false;
              }

            for(i = 0; i < read; i++)                               //  Write in bytes
              img->buffer[img->buflen + i] = line[i];

            img->buflen += (unsigned int)read;                      //  Update buffer size
          }
      }
    fclose(fp);                                                     //  Close file

    if(line)                                                        //  Clean up
      free(line);
    if(parse)
      free(parse);

    if(img->buflen != img->w * img->h)                              //  Do image dimensions match buffer size?
      {
        printf("Image data does not match stated image size.");
        return false;
      }

    return true;
  }
