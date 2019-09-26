// Copyright 2017 Trevor Simonton

#include "src/consumer.h"

#ifdef USE_MKL
#include "src/sgd_trainers/sgd_mkl_trainer.h"
Consumer::Consumer() {
  trainer = new SGDMKLTrainer();
  local_item = reinterpret_cast<int *>(
      calloc(tc_buffer_item_size, sizeof(int)));
}
#else
Consumer::Consumer() {
  trainer = new SGDTrainer();
  local_item = reinterpret_cast<int *>(
      calloc(tc_buffer_item_size, sizeof(int)));
}
#endif

Consumer::~Consumer() {
  delete trainer;
  free(local_item);
}

int Consumer::acquire() {
  TCBufferReader tc_reader;
  int got_item = tc_buffer->getReadyItem(&tc_reader);
  if (got_item) {
    memcpy(local_item, tc_reader.getData(), tc_buffer_item_size * sizeof(int));
    has_item = 1;
    return 1;
  }
  return 0;
}

int Consumer::consume() {
  if (has_item) {
    trainer->clear();
    TCBufferReader tc_reader(local_item);
    trainer->loadIndices(&tc_reader);
    trainer->loadTWords();
    trainer->loadCWords();
    trainer->train();
    word_count += 1 + tc_reader.droppedWords();
    if (word_count - last_word_count > 10000) {
      #pragma omp atomic
      word_count_actual += word_count - last_word_count;

      last_word_count = word_count;

      if (debug_mode > 1) {
        double now = omp_get_wtime();
        printf("\rAlpha_consumer: %f  Progress: %.2f%%  Words/sec: %.2fk",  alpha,
            word_count_actual / (real) (iter * train_words + 1) * 100,
            word_count_actual / ((now - start) * 1000));
        fflush(stdout);
      }

      alpha = starting_alpha *
        (1 - word_count_actual / (real) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001f) {
        alpha = starting_alpha * 0.0001f;
      }
      // printf("alpha = %f\n",alpha);
      
      // if (id == 0) {
      // 0.022500
        
        /*if (alpha >= 0.022499 && alpha < 0.022501) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 1 done!!\n\n");
            FILE *fo = fopen("wombat_cpu_vectors_epoch_1.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.020000
        if (alpha >= 0.019999 && alpha < 0.020001) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 2 done!!\n\n");
            
            FILE *fo = fopen("wombat_cpu_vectors_epoch_2.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.017500
        if (alpha >= 0.017499 && alpha < 0.017501) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 3 done!!\n\n");

            FILE *fo = fopen("wombat_cpu_vectors_epoch_3.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.015000
        if (alpha >= 0.014999 && alpha < 0.015001) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 4 done!!\n\n");

            FILE *fo = fopen("wombat_cpu_vectors_epoch_4.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.012500
        if (alpha >= 0.012499 && alpha < 0.012501) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 5 done!!\n\n");

            FILE *fo = fopen("wombat_cpu_vectors_epoch_5.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.010000
        if (alpha >= 0.009999 && alpha < 0.010001) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 6 done!!\n\n");

            FILE *fo = fopen("wombat_cpu_vectors_epoch_6.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.007500
        if (alpha >= 0.007499 && alpha < 0.007501) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 7 done!!\n\n");

            FILE *fo = fopen("wombat_cpu_vectors_epoch_7.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.005000
        if (alpha >= 0.004999 && alpha < 0.005001) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 8 done!!\n\n");

            FILE *fo = fopen("wombat_cpu_vectors_epoch_8.txt", "wb");
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
          // }
          
        }
      // }
      // if (id == 0) {
      // 0.002500
        if (alpha >= 0.002499 && alpha < 0.002501) {
          // #pragma omp barrier
          // if (id == 0) {
            printf("\n\nepoch = 9 done!!\n\n");

            FILE *fo = fopen("wombat_cpu_vectors_epoch_9.txt", "wb");
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
            
          // }
        }*/
      // }

      
    }

    return 1;
  }

  return 0;
}

void Consumer::setTCBuffer(TCBuffer *tcb) {
  tc_buffer = tcb;
}
TCBuffer* Consumer::getTCBuffer() {
  return tc_buffer;
}
