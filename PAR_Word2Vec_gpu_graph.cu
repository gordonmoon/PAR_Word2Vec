/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The data pre-processing and loading parts are based on the pWord2Vec implementation from Intel:
 * https://github.com/IntelLabs/pWord2Vec
 */

#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cuda_util.h"

using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

#define NUM_SMS 56

#define checkCUDAerr(err) {\
  cudaError_t cet = err;\
  if (cudaSuccess != cet) {\
    printf("%s %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(cet));\
    exit(0);\
  }\
}

struct vocab_word {
    unsigned int cn;
    char *word;
};

double rtclock(void) {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int MAX_NUM_SENTENCES = 9400;
int GAMMA = 32;
int binary = 0, debug_mode = 2;
bool disk = false;
int negative = 5, min_count = 5, min_reduce = 1, iter = 5, window = 5, batch_size = 16; // batch_size = 11;
int vocab_max_size = 1000, vocab_size = 0, hidden_size = 100;
unsigned long long train_words = 0, file_size = 0;
float alpha = 0.025f, sample = 1e-3f;
// float initial_alpha = 0.02f;
const float EXP_RESOLUTION = EXP_TABLE_SIZE / (MAX_EXP * 2.0f);

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int table_size = 1e8;

struct vocab_word *vocab = NULL;
int *vocab_hash = NULL;
int *table = NULL;
float *W_in = NULL, *W_out = NULL, *expTable = NULL;

int *d_table;
float *d_expTable;
float *d_W_in, *d_W_out;


void InitUnigramTable() {
    // table = (int *) _mm_malloc(table_size * sizeof(int), 64);
    table = (int *)malloc(table_size * sizeof(int));

    const float power = 0.75f;
    double train_words_pow = 0.;
    // #pragma omp parallel for num_threads(num_threads) reduction(+: train_words_pow)
    for (int i = 0; i < vocab_size; i++) {
        train_words_pow += pow(vocab[i].cn, power);
    }

    int i = 0;
    float d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (int a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (float) table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size)
            i = vocab_size - 1;
    }

    checkCUDAerr(cudaMalloc((void **)&d_table, table_size*sizeof(int)));
    checkCUDAerr(cudaMemcpy(d_table, table, table_size*sizeof(int), cudaMemcpyHostToDevice));
} // SAME as word2vec-gpu

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13)
            continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n')
                    ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *) "</s>");
                return;
            } else
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1)
            a--;   // Truncate too long words
    }
    word[a] = 0;
} // SAME as word2vec-gpu

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned int hash = 0;
    for (int i = 0; i < strlen(word); i++)
        hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
} // SAME as word2vec-gpu

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1)
            return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    // return -1;
} // SAME as word2vec-gpu

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin))
        return -1;
    return SearchVocab(word);
} // SAME as word2vec-gpu

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    int hash, length = strlen(word) + 1;
    if (length > MAX_STRING)
        length = MAX_STRING;
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        // vocab = (struct vocab_word *) floatloc(vocab, vocab_max_size * sizeof(struct vocab_word));
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
} // SAME as word2vec-gpu

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
} // SAME as word2vec-gpu

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (int a = 0; a < vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }
    // memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    int size = vocab_size;
    train_words = 0;
    for (int i = 0; i < size; i++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[i].cn < min_count) && (i != 0)) {
            vocab_size--;
            free(vocab[i].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            int hash = GetWordHash(vocab[i].word);
            while (vocab_hash[hash] != -1)
                hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = i;
            train_words += vocab[i].cn;
        }
    }
    // vocab = (struct vocab_word *) floatloc(vocab, vocab_size * sizeof(struct vocab_word));
    vocab = (struct vocab_word *) realloc(vocab, (vocab_size) * sizeof(struct vocab_word));
} // SAME as word2vec-gpu (except for vocab.code and vocab.point, and memset)

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int count = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].cn > min_reduce) {
            vocab[count].cn = vocab[i].cn;
            vocab[count].word = vocab[i].word;
            count++;
        } else {
            free(vocab[i].word);
        }
    }
    vocab_size = count;
    for (int a = 0; a < vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }
    // memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    for (int i = 0; i < vocab_size; i++) {
        // Hash will be re-computed, as it is not actual
        int hash = GetWordHash(vocab[i].word);
        while (vocab_hash[hash] != -1)
            hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = i;
    }
    min_reduce++;
} // SAME as word2vec-gpu (except for memset)

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];

    for (int a = 0; a < vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }
    // memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    FILE *fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }

    train_words = 0;
    vocab_size = 0;
    AddWordToVocab((char *) "</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        int i = SearchVocab(word);
        if (i == -1) {
            int a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else
            vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7)
            ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    FILE *fo = fopen(save_vocab_file, "wb");
    for (int i = 0; i < vocab_size; i++)
        fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (int a = 0; a < vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }
    // memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    char c;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        int i = AddWordToVocab(word);
        fscanf(fin, "%d%c", &vocab[i].cn, &c);
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fclose(fin);

    // get file size
    FILE *fin2 = fopen(train_file, "rb");
    if (fin2 == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin2, 0, SEEK_END);
    file_size = ftell(fin2);
    fclose(fin2);
}

void InitNet() {
    W_in = (float *)malloc(vocab_size * hidden_size * sizeof(float));
    W_out = (float *)malloc(vocab_size * hidden_size * sizeof(float));

    if (!W_in || !W_out) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            W_in[i * hidden_size + j] = 0.f;
            W_out[i * hidden_size + j] = 0.f;
        }
    }

    // initialization
    unsigned long long next_random = 1;
    for (int i = 0; i < vocab_size * hidden_size; i++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        W_in[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
    }

    checkCUDAerr(cudaMalloc((void **)&d_W_in, vocab_size * hidden_size * sizeof(float)));
    checkCUDAerr(cudaMalloc((void **)&d_W_out, vocab_size * hidden_size * sizeof(float)));
    checkCUDAerr(cudaMemcpy(d_W_in, W_in, (vocab_size * hidden_size) * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDAerr(cudaMemcpy(d_W_out, W_out, (vocab_size * hidden_size) * sizeof(int), cudaMemcpyHostToDevice));

}




__global__ void par_sgns_kernel(int num_sentence_blk, int window, int hidden_size, int negative, float alpha, int table_size, int vocab_size, int batch_size, int total_num_sent, const int* __restrict__ d_table, const float* __restrict__ d_expTable, const int* __restrict__ d_corpus, const int* __restrict__ d_sen_len_ptr, float *d_W_in, float *d_W_out, float *M_in, float *M_in_update, float *M_out, float *M_out_update, float *M_grad, int *d_rand_window_batch, int *d_shared_negative)
{
    __shared__ float s_vector[1024]; // best occupancy

    const int s_sent_idx_pb = blockIdx.x*num_sentence_blk;
    const int e_sent_idx_pb = min(s_sent_idx_pb+num_sentence_blk, total_num_sent);

    if (s_sent_idx_pb < total_num_sent) {

        for (int sid = s_sent_idx_pb; sid < e_sent_idx_pb; sid++) {
            const int sent_idx_s = d_sen_len_ptr[sid];
            const int sent_idx_e = d_sen_len_ptr[sid + 1];
            const int sent_length = sent_idx_e - sent_idx_s;
            if (sent_length > 0) {
                unsigned long next_random = sid;
                float f = 0;
                for (int sentence_position = sent_idx_s; sentence_position < sent_idx_e; sentence_position++) {
                    int word = d_corpus[sentence_position];
                    if (word == -1) continue;
                    float neu1e = 0;
                    next_random = next_random * (unsigned long)25214903917 + 11;
                    int b = next_random % window;
                    for (int a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                        int c = sentence_position - window + a;
                        if (c < sent_idx_s) continue;
                        if (c >= sent_idx_e) continue;
                        int last_word = d_corpus[c];
                        if (last_word == -1) continue;
                        int l1 = last_word * hidden_size;
                        neu1e = 0;
                        int target, label;
                        target = word;
                        label = 1;
                        int l2 = target * hidden_size;
                        if (threadIdx.x / 32 == 0) {
                            float f2, f3, f4;
                            f = d_W_in[threadIdx.x + l1] * d_W_out[threadIdx.x + l2];
                            f2 = d_W_in[threadIdx.x + l1 + 32] * d_W_out[threadIdx.x + l2 + 32];
                            f3 = d_W_in[threadIdx.x + l1 + 64] * d_W_out[threadIdx.x + l2 + 64];
                            f4 = d_W_in[threadIdx.x + l1 + 96]  * d_W_out[threadIdx.x + l2 + 96];
                            f += f2 + f3 + f4;
                            f += __shfl_down(f, 16);
                            f += __shfl_down(f, 8);
                            f += __shfl_down(f, 4);
                            f += __shfl_down(f, 2);
                            f += __shfl_down(f, 1);
                        }
                        if (threadIdx.x == 0) {
                            if (f > MAX_EXP)
                                s_vector[0] = 0;
                            else if (f < -MAX_EXP)
                                s_vector[0] = alpha;
                            else
                                s_vector[0] = (label - d_expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                        }
                        __syncthreads();
                        neu1e += s_vector[0] * d_W_out[threadIdx.x + l2];
                        atomicAdd(&d_W_out[threadIdx.x + l2], s_vector[0] * d_W_in[threadIdx.x + l1]);
                        atomicAdd(&d_W_in[threadIdx.x + l1], neu1e);
                    }
                }
                __syncthreads();

                const int num_batch = (sent_length+batch_size-1) / batch_size;
                for (int batch_id = 0; batch_id < num_batch; batch_id++) {
                    const int batch_min_idx = sent_idx_s+(batch_id*batch_size);
                    const int batch_max_idx = min(sent_idx_s+(batch_id*batch_size)+batch_size,sent_idx_e);
                    const int input_size = batch_max_idx - batch_min_idx;
                    int rand_window;
                    unsigned int seed;
                    if (threadIdx.x == 0) {
                        seed = abs((int) clock()*100000000) + sid;
                        curandState s;
                        curand_init(seed, 0, 0, &s);
                        if (batch_min_idx-sent_idx_s < window || (sent_length-1+sent_idx_s)-batch_min_idx < window) {
                            rand_window = curand_uniform(&s) * (window-1);
                        }
                        else {
                            rand_window = curand_uniform(&s) * (2*window-1);
                        }
                        d_rand_window_batch[blockIdx.x] = rand_window;
                    }
                    __syncthreads();
                    const int output_size = d_rand_window_batch[blockIdx.x]*negative;
                    for (int j = threadIdx.x; j < output_size; j += blockDim.x) {
                        seed = abs((int) clock()*100000000) + threadIdx.x;
                        curandState s;
                        curand_init(seed, 0, 0, &s);
                        int neg_target = (curand_uniform(&s) * (vocab_size-2))+1;
                        d_shared_negative[blockIdx.x*2*window*negative + j] = neg_target;
                    }
                    __syncthreads();
                    for (int sen_pos = batch_min_idx; sen_pos < batch_max_idx; sen_pos++) {       // 16
                        int s1 = d_corpus[sen_pos] * hidden_size;
                        int d1 = (sen_pos-batch_min_idx) * hidden_size;
                        M_in[blockIdx.x*batch_size*hidden_size + d1 + threadIdx.x] = d_W_in[s1 + threadIdx.x];    
                    }
                    for (int ng_pos = 0; ng_pos < output_size; ng_pos++) {     // 40
                        int s2 = d_shared_negative[blockIdx.x*2*window*negative + ng_pos] * hidden_size;
                        int d2 = ng_pos * hidden_size;
                        M_out[blockIdx.x*2*window*negative*hidden_size + d2 + threadIdx.x] = d_W_out[s2 + threadIdx.x];
                    }
                    __syncthreads();

                    // MM1: 16*32 + 32*16 = 1024

                    for (int a = threadIdx.x; a < output_size*batch_size; a += blockDim.x) {
                        M_grad[blockIdx.x*2*window*negative*batch_size + a] = 0;
                    }

                    const int num_row_blocks_mm1 = (output_size+16-1) / 16;
                    const int num_col_blocks_mm1 = (input_size+16-1) / 16;
                    const int m_blocks_mm1 = hidden_size/32;

                    int thread_row = threadIdx.x / 16;
                    int thread_col = threadIdx.x % 16;

                    float reg_tiles[8];

                    for (int rb = 0; rb < num_row_blocks_mm1; rb++) {
                        int row_size = min(16, output_size-(rb*16));
                        for (int cb = 0; cb < num_col_blocks_mm1; cb++) {
                            int col_size = min(16, input_size-(cb*16));

                            for (int i = 0; i < 8; i++) {
                                reg_tiles[i] = 0.0;
                            }

                            for (int m = 0; m < m_blocks_mm1; m++) {
                                int m_dim = 32;
                                int num_sub_row_blocks = (row_size*32+hidden_size-1) / hidden_size; // 4
                                for (int sb = 0; sb < num_sub_row_blocks; sb++) {
                                    s_vector[sb*hidden_size+threadIdx.x] = 
                                    M_out[blockIdx.x*2*window*negative*hidden_size + rb*16*hidden_size + sb*4*hidden_size + m*32 + 
                                        ((int)threadIdx.x/32)*hidden_size + (threadIdx.x%32)];
                                }
                                int num_sub_col_blocks = (col_size*32+hidden_size-1) / hidden_size;
                                for (int sb = 0; sb < num_sub_col_blocks; sb ++) {
                                    s_vector[16*32 + sb*hidden_size+threadIdx.x] = 
                                    M_in[blockIdx.x*batch_size*hidden_size + cb*16*hidden_size + sb*4*hidden_size + m*32 +
                                        ((int)threadIdx.x/32)*hidden_size + (threadIdx.x%32)];
                                }
                                __syncthreads();

                                if (thread_col < col_size)
                                for (int k = 0; k < m_dim; k++) {
                                    float element_2nd = s_vector[16*32 + thread_col*32 + k];
                                    reg_tiles[0] += s_vector[thread_row*32 + k] * element_2nd;
                                    reg_tiles[1] += s_vector[(thread_row+8)*32 + k] * element_2nd;
                                }
                                __syncthreads();
                            }
                            if (thread_col < col_size) {
                                if (thread_row < row_size)
                                    M_grad[blockIdx.x*2*window*negative*batch_size + (rb*16 + thread_row)*batch_size + (cb*16) + thread_col] = reg_tiles[0];
                                if (thread_row + 8 < row_size)
                                    M_grad[blockIdx.x*2*window*negative*batch_size + (rb*16 + thread_row + 8)*batch_size + (cb*16) + thread_col] = reg_tiles[1];
                            }
                            
                        }
                    }
                    __syncthreads();

                    int label = 0;
                    for (int g = threadIdx.x; g < output_size*batch_size; g += blockDim.x) {
                        float gvalue;
                        float fvalue = M_grad[blockIdx.x*2*window*negative*batch_size + g];
                        if (fvalue > MAX_EXP)
                            gvalue = -alpha;
                        else if (fvalue < -MAX_EXP)
                            gvalue = 0;
                        else
                            gvalue = (label - d_expTable[(int) ((fvalue + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                        M_grad[blockIdx.x*2*window*negative*batch_size + g] = gvalue;
                    }
                    __syncthreads();

                    // MM2: 32*16 + 16*32 = 1024

                    for (int a = threadIdx.x; a < output_size*hidden_size; a += blockDim.x) {
                        M_out_update[blockIdx.x*2*window*negative*hidden_size + a] = 0;
                    }
                    const int num_row_blocks_mm2 = (output_size+32-1) / 32;
                    const int num_col_blocks_mm2 = hidden_size / 32;
                    const int m_blocks_mm2 = (input_size+16-1) / 16;

                    thread_row = threadIdx.x / 32;
                    thread_col = threadIdx.x % 32;

                    for (int rb = 0; rb < num_row_blocks_mm2; rb++) {
                        int row_size = min(32, output_size-(rb*32));
                        for (int cb = 0; cb < num_col_blocks_mm2; cb++) {
                            int col_size = 32;

                            for (int i = 0; i < 8; i++) {
                                reg_tiles[i] = 0.0;
                            }

                            for (int m = 0; m < m_blocks_mm2; m++) {
                                int m_dim = min(16,input_size-(m*16));
                                int num_sub_row_blocks = (row_size*16+hidden_size-1) / hidden_size;
                                for (int sb = 0; sb < num_sub_row_blocks; sb++) {
                                    s_vector[sb*hidden_size+threadIdx.x] =
                                    M_grad[blockIdx.x*2*window*negative*batch_size + rb*32*batch_size + sb*8*batch_size + m*16 +
                                        ((int)threadIdx.x/16)*batch_size + (threadIdx.x%16)];
                                }
                                int num_sub_col_blocks = (col_size*16+hidden_size-1) / hidden_size;
                                for (int sb = 0; sb < num_sub_col_blocks; sb ++) {
                                    s_vector[32*16 + sb*hidden_size+threadIdx.x] = 
                                    M_in[blockIdx.x*batch_size*hidden_size + m*16*hidden_size + sb*4*hidden_size + cb*32 +
                                        ((int)threadIdx.x/32)*hidden_size + (threadIdx.x%32)];
                                }
                                __syncthreads();

                                for (int k = 0; k < m_dim; k++) {
                                    float element_2nd = s_vector[32*16 + k*32 + thread_col];
                                    reg_tiles[0] += s_vector[thread_row*16 + k] * element_2nd;
                                    reg_tiles[1] += s_vector[(thread_row + 4)*16 + k] * element_2nd;
                                    reg_tiles[2] += s_vector[(thread_row + 8)*16 + k] * element_2nd;
                                    reg_tiles[3] += s_vector[(thread_row + 12)*16 + k] * element_2nd;
                                    reg_tiles[4] += s_vector[(thread_row + 16)*16 + k] * element_2nd;
                                    reg_tiles[5] += s_vector[(thread_row + 20)*16 + k] * element_2nd;
                                    reg_tiles[6] += s_vector[(thread_row + 24)*16 + k] * element_2nd;
                                    reg_tiles[7] += s_vector[(thread_row + 28)*16 + k] * element_2nd;
                                }
                                __syncthreads();
                            }

                            if (thread_row < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row)*hidden_size + (cb*32) + thread_col] = reg_tiles[0];
                            if (thread_row + 4 < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row + 4)*hidden_size + (cb*32) + thread_col] = reg_tiles[1];
                            if (thread_row + 8 < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row + 8)*hidden_size + (cb*32) + thread_col] = reg_tiles[2];
                            if (thread_row + 12 < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row + 12)*hidden_size + (cb*32) + thread_col] = reg_tiles[3];
                            if (thread_row + 16 < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row + 16)*hidden_size + (cb*32) + thread_col] = reg_tiles[4];
                            if (thread_row + 20 < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row + 20)*hidden_size + (cb*32) + thread_col] = reg_tiles[5];
                            if (thread_row + 24 < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row + 24)*hidden_size + (cb*32) + thread_col] = reg_tiles[6];
                            if (thread_row + 28 < row_size)
                                M_out_update[blockIdx.x*2*window*negative*hidden_size + (rb*32 + thread_row + 28)*hidden_size + (cb*32) + thread_col] = reg_tiles[7];

                        }
                    }
                    __syncthreads();


                    // MM3: 8*20 + 20*32 = 800

                    for (int a = threadIdx.x; a < input_size*hidden_size; a += blockDim.x) {
                        M_in_update[blockIdx.x*batch_size*hidden_size + a] = 0;
                    }

                    const int num_row_blocks_mm3 = (input_size+8-1) / 8;
                    const int num_col_blocks_mm3 = hidden_size / 32;
                    const int m_blocks_mm3 = (output_size+20-1) / 20;

                    thread_row = threadIdx.x / 32;
                    thread_col = threadIdx.x % 32;


                    for (int rb = 0; rb < num_row_blocks_mm3; rb++) {
                        int row_size = min(8,input_size-(rb*8));
                        for (int cb = 0; cb < num_col_blocks_mm3; cb++) {
                            int col_size = 32;

                            for (int i = 0; i < 8; i++) {
                                reg_tiles[i] = 0.0;
                            }

                            for (int m = 0; m < m_blocks_mm3; m++) {
                                int m_dim = min(20,output_size-(m*20));
                                if (threadIdx.x < 80) {
                                    int num_sub_row_blocks = (row_size*20+80-1) / 80;
                                    for (int sb = 0; sb < num_sub_row_blocks; sb++) {
                                        s_vector[sb*80+threadIdx.x] = 
                                        M_grad[blockIdx.x*2*window*negative*batch_size + m*20*batch_size + sb*10*batch_size + rb*8 + 
                                            ((int)threadIdx.x/8)*batch_size + (threadIdx.x%8)];
                                    }
                                }
                                int num_sub_col_blocks = (col_size*20+hidden_size-1) / hidden_size;
                                for (int sb = 0; sb < num_sub_col_blocks; sb++) {
                                    s_vector[8*20 + sb*hidden_size+threadIdx.x] = 
                                    M_out[blockIdx.x*2*window*negative*hidden_size + m*20*hidden_size + sb*4*hidden_size + cb*32 + 
                                        ((int)threadIdx.x/32)*hidden_size + (threadIdx.x%32)];
                                }
                                __syncthreads();

                                for (int k = 0; k < m_dim; k++) {
                                    float element_2nd = s_vector[8*20 + k*32 + thread_col];
                                    reg_tiles[0] += s_vector[k*8 + thread_row] * element_2nd;
                                    reg_tiles[1] += s_vector[k*8 + thread_row + 4] * element_2nd;
                                    reg_tiles[2] += s_vector[k*8 + thread_row + 8] * element_2nd;
                                    reg_tiles[3] += s_vector[k*8 + thread_row + 12] * element_2nd;
                                    reg_tiles[4] += s_vector[k*8 + thread_row + 16] * element_2nd;
                                    reg_tiles[5] += s_vector[k*8 + thread_row + 20] * element_2nd;
                                    reg_tiles[6] += s_vector[k*8 + thread_row + 24] * element_2nd;
                                    reg_tiles[7] += s_vector[k*8 + thread_row + 28] * element_2nd;
                                }
                                __syncthreads();
                            }

                            if (thread_row < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row)*hidden_size + (cb*32) + thread_col] = reg_tiles[0];
                            if (thread_row + 4 < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row + 4)*hidden_size + (cb*32) + thread_col] = reg_tiles[1];
                            if (thread_row + 8 < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row + 8)*hidden_size + (cb*32) + thread_col] = reg_tiles[2];
                            if (thread_row + 12 < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row + 12)*hidden_size + (cb*32) + thread_col] = reg_tiles[3];
                            if (thread_row + 16 < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row + 16)*hidden_size + (cb*32) + thread_col] = reg_tiles[4];
                            if (thread_row + 20 < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row + 20)*hidden_size + (cb*32) + thread_col] = reg_tiles[5];
                            if (thread_row + 24 < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row + 24)*hidden_size + (cb*32) + thread_col] = reg_tiles[6];
                            if (thread_row + 28 < row_size)
                                M_in_update[blockIdx.x*batch_size*hidden_size + (rb*8 + thread_row + 28)*hidden_size + (cb*32) + thread_col] = reg_tiles[7];
                        }
                    }
                    __syncthreads();

                    for (int ng_pos = 0; ng_pos < output_size; ng_pos++) {
                        int s4 = ng_pos * hidden_size;
                        int d4 = d_shared_negative[blockIdx.x*2*window*negative + ng_pos] * hidden_size;
                        // d_W_out[d4 + threadIdx.x] += M_out_update[blockIdx.x*2*window*negative*hidden_size + s4 + threadIdx.x];
                        atomicAdd(&d_W_out[d4 + threadIdx.x], M_out_update[blockIdx.x*2*window*negative*hidden_size + s4 + threadIdx.x]);
                    }
                    __syncthreads();
                    for (int sen_pos = batch_min_idx; sen_pos < batch_max_idx; sen_pos++) {
                        int s3 = (sen_pos-batch_min_idx) * hidden_size;
                        int d3 = d_corpus[sen_pos] * hidden_size;
                        // d_W_in[d3 + threadIdx.x] += M_in_update[blockIdx.x*batch_size*hidden_size + s3 + threadIdx.x];
                        atomicAdd(&d_W_in[d3 + threadIdx.x], M_in_update[blockIdx.x*batch_size*hidden_size + s3 + threadIdx.x]);
                    }
                    __syncthreads();
                }
            }
        }
    }
}

void Train_PAR_Word2Vec() {

    if (read_vocab_file[0] != 0) {
        ReadVocab();
    }
    else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    InitNet();
    InitUnigramTable();


/************** Loading entire dataset and wrtie it into the corpus_local ***************/

    FILE *fin = fopen(train_file, "rb");
    fseek(fin, 0, SEEK_SET);

    unsigned long long updated_train_words = 0;
    unsigned long long local_word_count;
    int *corpus_local; // stream
    if (!disk) {
        corpus_local = (int *)malloc((train_words + 1) * sizeof(int));
        local_word_count = 0;
        while (!feof(fin) && local_word_count < train_words) {
            int w = ReadWordIndex(fin);
            if (w == -1)
                continue;
            corpus_local[local_word_count] = w;
            local_word_count++;
            // printf("%lldwords%c",local_word_count,13);
            // fflush(stdout);
        }
        corpus_local[local_word_count] = 0; // set the last word as "</s>" // sentences are splitted by 0 (as EOS symbol)
        updated_train_words = local_word_count;
        fclose(fin);
    }

/************** Pre-processing dataset and write it into the corpus ********************/

    // Allocate HOST memory
    int *corpus;
    corpus = (int *)malloc((train_words+10) * sizeof(int));
    int *sen_len_ptr;
    sen_len_ptr = (int *)malloc((MAX_NUM_SENTENCES+1) * sizeof(int));

    int word, sentence_length = 0, sentence_position = 0, sen_id = 0;
    unsigned long long word_count = 0;
    int sen[MAX_SENTENCE_LENGTH];
    unsigned long long next_random = 0;
    unsigned long long word_position = 0;
    unsigned long long total_number_sentences = 0;
    unsigned long long total_word_tokens = 0;
    for (int i = 0; i < (MAX_NUM_SENTENCES+1); i++) {
        sen_len_ptr[i] = 0;
    }

    while (1) {

        if (sentence_length == 0) {
            while (1) {
                word = corpus_local[word_count];
                word_count++;
                if (word == 0) break; // last word: word == 0
                if (sample > 0) {
                    float ratio = (sample * train_words) / vocab[word].cn;
                    float ran = sqrtf(ratio) + ratio;
                    next_random = next_random * (unsigned long long) 25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / 65536.f)
                        continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }

        if ((word_count > train_words)) {
            break;
        }

        if (sentence_length != 0) {
            word = sen[sentence_position];
            corpus[word_position] = word;
            word_position++;
            sentence_position++;
        }

        if (sentence_position >= sentence_length) {
            total_word_tokens = total_word_tokens + sentence_length;
            printf("%dsentences%c",sen_id,13);
            fflush(stdout);
            sen_len_ptr[sen_id+1] = sen_len_ptr[sen_id] + sentence_length;
            sen_id++;
            total_number_sentences++;
            sentence_length = 0;
            continue;
        }

    }

    // Allocate DEVICE memory
    // cudaMemcpy host to device memory

    int GPU_num_blocks;
    if (GAMMA * NUM_SMS < total_number_sentences) {
        GPU_num_blocks = GAMMA * NUM_SMS;
    }
    else {
        GPU_num_blocks = total_number_sentences;
    }

    int *d_corpus, *d_sen_len_ptr;
    checkCUDAerr(cudaMalloc((void **)&d_corpus, word_position * sizeof(int)));
    checkCUDAerr(cudaMemcpy(d_corpus, corpus, word_position * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDAerr(cudaMalloc((void **)&d_sen_len_ptr, (total_number_sentences + 1) * sizeof(int)));
    checkCUDAerr(cudaMemcpy(d_sen_len_ptr, sen_len_ptr, (total_number_sentences + 1) * sizeof(int), cudaMemcpyHostToDevice));

    float *d_M_in, *d_M_in_update, *d_M_out, *d_M_out_update, *d_M_grad;
    checkCUDAerr(cudaMalloc((void **)&d_M_in, (16 * 3 * hidden_size * GPU_num_blocks) * sizeof(float)));
    checkCUDAerr(cudaMalloc((void **)&d_M_in_update, (batch_size * hidden_size * GPU_num_blocks) * sizeof(float)));

    checkCUDAerr(cudaMalloc((void **)&d_M_out, (32 * 4 * hidden_size * GPU_num_blocks) * sizeof(float))); // for graph
    checkCUDAerr(cudaMalloc((void **)&d_M_out_update, (32 * 4 * hidden_size * GPU_num_blocks) * sizeof(float))); // for graph

    checkCUDAerr(cudaMalloc((void **)&d_M_grad, (32 * 4 * 16 * 3 * GPU_num_blocks) * sizeof(float))); // for graph

    checkCUDAerr(cudaMemset(d_M_in, 0, sizeof(float) * (16 * 3 * hidden_size * GPU_num_blocks)));
    checkCUDAerr(cudaMemset(d_M_in_update, 0, sizeof(float) * (batch_size * hidden_size * GPU_num_blocks)));

    checkCUDAerr(cudaMemset(d_M_out, 0, sizeof(float) * (32 * 4 * hidden_size * GPU_num_blocks))); // for graph
    checkCUDAerr(cudaMemset(d_M_out_update, 0, sizeof(float) * (32 * 4 * hidden_size * GPU_num_blocks))); // for graph

    checkCUDAerr(cudaMemset(d_M_grad, 0, sizeof(float) * (32 * 4 * 16 * 3 * GPU_num_blocks))); // for graph

    int *d_rand_window_batch, *d_shared_negative;
    checkCUDAerr(cudaMalloc((void **)&d_rand_window_batch, GPU_num_blocks * sizeof(int)));
    checkCUDAerr(cudaMalloc((void **)&d_shared_negative, (2 * window * negative * GPU_num_blocks) * sizeof(int)));
    checkCUDAerr(cudaMemset(d_rand_window_batch, 0, sizeof(int) * GPU_num_blocks));
    checkCUDAerr(cudaMemset(d_shared_negative, 0, sizeof(int) * (2 * window * negative * GPU_num_blocks)));


/***********************************************************************************/

    float annealing_rate = alpha / iter;
    printf("annealing rate = %.5f\n",annealing_rate);

    float sum_total_time = 0.0;
    int local_iter = iter;

    while (1) {

        printf("training epoch %d...\t",iter-local_iter+1);
        printf("learning rate = %.5f\n",alpha);

        float mili = 0;

        int bDim = hidden_size;
        int gDim = GAMMA * NUM_SMS;
        
        int total_num_sent = total_number_sentences;
        int num_sentences_pb = (total_num_sent+gDim-1) / gDim;

        if (gDim < total_number_sentences) {

            cudaEvent_t start_cuda, stop_cuda;
            cudaEventCreate(&start_cuda);
            cudaEventCreate(&stop_cuda);
            cudaEventRecord(start_cuda);

            par_sgns_kernel<<<gDim, bDim>>>(num_sentences_pb, window, hidden_size, negative, alpha, table_size, vocab_size, batch_size, total_num_sent, d_table, d_expTable, d_corpus, d_sen_len_ptr, d_W_in, d_W_out, d_M_in, d_M_in_update, d_M_out, d_M_out_update, d_M_grad, d_rand_window_batch, d_shared_negative);
            
            cudaDeviceSynchronize();
            cudaEventRecord(stop_cuda);
            cudaEventSynchronize(stop_cuda);
            cudaEventElapsedTime(&mili, start_cuda, stop_cuda);
        }
        else {
            cudaEvent_t start_cuda, stop_cuda;
            cudaEventCreate(&start_cuda);
            cudaEventCreate(&stop_cuda);
            cudaEventRecord(start_cuda);

            par_sgns_kernel<<<total_num_sent, bDim>>>(num_sentences_pb, window, hidden_size, negative, alpha, table_size, vocab_size, batch_size, total_num_sent, d_table, d_expTable, d_corpus, d_sen_len_ptr, d_W_in, d_W_out, d_M_in, d_M_in_update, d_M_out, d_M_out_update, d_M_grad, d_rand_window_batch, d_shared_negative);

            cudaDeviceSynchronize();
            cudaEventRecord(stop_cuda);
            cudaEventSynchronize(stop_cuda);
            cudaEventElapsedTime(&mili, start_cuda, stop_cuda);
        }

        sum_total_time += mili;


        // checkCUDAerr(cudaMemcpy(W_in, d_W_in, vocab_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
        // string default_file = "PAR_Word2Vec_gpu_vectors_epoch_";
        // string local_epoch = to_string(iter-local_iter+1);
        // string text = ".txt";
        // string local_filename = default_file+local_epoch+text;
        // FILE *foutput = fopen(local_filename.c_str(), "wb");
        // fprintf(foutput, "%d %d\n", vocab_size, hidden_size);
        // for (int a = 0; a < vocab_size; a++) {
        //     fprintf(foutput, "%s ", vocab[a].word);
        //     if (binary)
        //         for (int b = 0; b < hidden_size; b++)
        //             fwrite(&W_in[a * hidden_size + b], sizeof(float), 1, foutput);
        //     else
        //         for (int b = 0; b < hidden_size; b++)
        //             fprintf(foutput, "%f ", W_in[a * hidden_size + b]);
        //     fprintf(foutput, "\n");
        // }
        // fclose(foutput);


        local_iter--;
        alpha = alpha - annealing_rate;

        if (local_iter == 0) {
            break;
        }
        continue;


    }

    cudaDeviceSynchronize();


    FILE *fpOut = fopen("PAR_Word2Vec_gpu_time", "w");
    fprintf(fpOut, "Elapsed %.2lf",sum_total_time/1000);
    fclose(fpOut);

    // printf("(CUDA timer) Accumulated total training time for %d epochs: %f seconds \n", iter, sum_total_time/1000);

    checkCUDAerr(cudaMemcpy(W_in, d_W_in, vocab_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));

    free(corpus);
    free(corpus_local);
    free(sen_len_ptr);
    
    cudaFree(d_corpus);
    cudaFree(d_sen_len_ptr);
    cudaFree(d_M_in);
    cudaFree(d_M_in_update);
    cudaFree(d_M_out);
    cudaFree(d_M_out_update);
    cudaFree(d_M_grad);
    cudaFree(d_rand_window_batch);
    cudaFree(d_shared_negative);

}

int ArgPos(char *str, int argc, char **argv) {
    for (int a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            return a;
        }
    return -1;
}

void saveModel() {
    // save the model
    FILE *fo = fopen(output_file, "wb");
    // Save the word vectors
    fprintf(fo, "%d %d\n", vocab_size, hidden_size);
    for (int a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary)
            for (int b = 0; b < hidden_size; b++)
                fwrite(&W_in[a * hidden_size + b], sizeof(float), 1, fo);
        else
            for (int b = 0; b < hidden_size; b++)
                fprintf(fo, "%f ", W_in[a * hidden_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int main(int argc, char **argv) {
    if (argc == 1) {
        printf("parallel word2vec (sgns) in shared memory system\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-batch-size <int>\n");
        printf("\t\tThe batch size used for mini-batch training; default is 11 (i.e., 2 * window + 1)\n");
        printf("\t-disk\n");
        printf("\t\tStream text from disk during training, otherwise the text will be loaded into memory before training\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -binary 0 -iter 3\n\n");
        return 0;
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    int i;
    if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
        hidden_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
        strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
        strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
        strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
        debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)
        binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
        alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-output", argc, argv)) > 0)
        strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
        window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
        sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
        negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-iter", argc, argv)) > 0)
        iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
        min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-batch-size", argc, argv)) > 0)
        batch_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-gamma", argc, argv)) > 0)
        GAMMA = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-max-num-sen", argc, argv)) > 0)
        MAX_NUM_SENTENCES = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-disk", argc, argv)) > 0)
        disk = true;

    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)malloc(vocab_hash_size * sizeof(int));
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));

    for (i = 0; i < EXP_TABLE_SIZE + 1; i++) {
        expTable[i] = exp((i / (float) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                     // Precompute f(x) = x / (x + 1)
    }

    checkCUDAerr(cudaMalloc((void **)&d_expTable, (EXP_TABLE_SIZE + 1) * sizeof(float)));
    checkCUDAerr(cudaMemcpy(d_expTable, expTable, (EXP_TABLE_SIZE + 1) * sizeof(float), cudaMemcpyHostToDevice));

    printf("number of iterations: %d\n", iter);
    printf("hidden size: %d\n", hidden_size);
    printf("number of negative samples: %d\n", negative);
    printf("window size: %d\n", window);
    printf("batch size: %d\n", batch_size);
    printf("gamma (number of thread blocks / 56): %d\n", GAMMA);
    printf("max number of sentences: %d\n", MAX_NUM_SENTENCES);
    printf("starting learning rate: %.5f\n", alpha);
    printf("starting training using file: %s\n\n", train_file);

    Train_PAR_Word2Vec();

    saveModel();

    free(table);
    free(expTable);
    free(W_in);
    free(W_out);
    free(vocab);
    free(vocab_hash);
    cudaFree(d_table);
    cudaFree(d_expTable);
    cudaFree(d_W_in);
    cudaFree(d_W_out);
    return 0;
}

