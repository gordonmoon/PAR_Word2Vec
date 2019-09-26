// Copyright 2017 Trevor Simonton

#include "src/console.h"
#include "src/pht_model.h"
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <iostream>
// #include <papi.h>

// #define PAPI_ERROR_CHECK(X) if((X)!=PAPI_OK) std::cerr<<"Error \n";

// int event_set = PAPI_NULL;
// long_long values[2];
// //Initialize PAPI
// PAPI_library_init(PAPI_VER_CURRENT);
// //Create event set
// PAPI_ERROR_CHECK(PAPI_create_eventset(&event_set));
// //Add events
// // PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_TOT_CYC));
// PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_SP_OPS));
// PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_DP_OPS));

// // //Start the counters
// PAPI_ERROR_CHECK(PAPI_start(event_set));

// PAPI_ERROR_CHECK(PAPI_stop(event_set, values));
// std::cout << "SP ops: " << values[0] << " " << "DP ops: " << values[1] << "\n";

double rtclock(void) {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int main(int argc, char **argv) {
  if (readConsoleArgs(argc, argv)) {
    PHTModel t;
    t.init();

    double start_training = rtclock();

    t.train();

    double end_training = rtclock();
    // double training_time = end_training - start_training;
    FILE *fpOut = fopen("wombatSGNS_cpu_time", "w");
    fprintf(fpOut, "Elapsed %.2lf",end_training - start_training);
    fclose(fpOut);

    FILE *fo = fopen("wombatSGNS_cpu_vectors.txt", "wb");
    // Save the word vectors
    fprintf(fo, "%d %d\n", vocab_size, hidden_size);
    for (int a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary)
        for (int b = 0; b < hidden_size; b++)
            fwrite(&Wih[a * hidden_size + b], sizeof(real), 1, fo);
        else
        for (int b = 0; b < hidden_size; b++)
            fprintf(fo, "%f ", Wih[a * hidden_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);



  }
  return 0;
}
