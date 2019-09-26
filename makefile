CXX = icc
CC = icc
CPPFLAGS = -std=c++11 -qopenmp -O3 -D USE_MKL -mkl=sequential -Wall -xhost
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

# ICPC = icpc
# ICFLAGS = -I ./ -std=c++11 -qopenmp -D USE_MKL -mkl=sequential -xhost -g

# SHARED := src/console.cpp \
# 	src/buffers/producers/ti_producer.cpp src/data_source/word_source_file.cpp \
# 	src/data_source/word_source_group.cpp src/data_source/word_source_file_group.cpp src/buffers/producers/sentence_producer.cpp \
# 	src/pht_model.cpp src/buffers/sen_buffer.cpp src/buffers/readers/sen_buffer.cpp \
# 	src/sgd_trainers/sgd_trainer.cpp src/buffers/tc_buffer.cpp src/buffers/readers/tc_buffer.cpp  \
# 	src/w2v-functions.cpp src/worker_model.cpp src/consumer.cpp \
# 	src/batch_consumer.cpp src/batch_model.cpp src/sgd_trainers/sgd_batch_trainer.cpp

GPU_ARCH_FLAG = arch=compute_60,code=sm_60
NVCC_HOME = /usr/local/cuda/9.2.88
NVCC = nvcc
CUDA_INC = -I$(NVCC_HOME)/include
CUDA_LIB = -L$(NVCC_HOME)/lib64 -lcudart -lcublas -lcusparse
CUDA_FLAGS = -O3 -std=c++11 -m64 -gencode $(GPU_ARCH_FLAG)

all: PAR_Word2Vec_cpu PAR_Word2Vec_gpu_text PAR_Word2Vec_gpu_graph pSGNScc_cpu pWord2Vec_cpu word2vec_cpu accSGNS_gpu_text8 accSGNS_gpu_1b accSGNS_gpu_blogcatalog accSGNS_gpu_ppi accSGNS_gpu_wikipedia accSGNS_gpu_facebook accSGNS_gpu_astroph

PAR_Word2Vec_cpu: PAR_Word2Vec_cpu.cpp
	$(CXX) PAR_Word2Vec_cpu.cpp -o PAR_Word2Vec_cpu $(CPPFLAGS)
PAR_Word2Vec_gpu_text: PAR_Word2Vec_gpu_text.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o PAR_Word2Vec_gpu_text PAR_Word2Vec_gpu_text.cu $(CUDA_LIB)
PAR_Word2Vec_gpu_graph: PAR_Word2Vec_gpu_graph.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o PAR_Word2Vec_gpu_graph PAR_Word2Vec_gpu_graph.cu $(CUDA_LIB)
pSGNScc_cpu: pSGNScc.cpp
	$(CXX) pSGNScc.cpp -o pSGNScc_cpu $(CPPFLAGS)
pWord2Vec_cpu: pWord2Vec.cpp
	$(CXX) pWord2Vec.cpp -o pWord2Vec_cpu $(CPPFLAGS)
word2vec_cpu : word2vec.c
	$(CC) word2vec.c -o word2vec_cpu $(CFLAGS)
accSGNS_gpu_text8: accSGNS_gpu_text8.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o accSGNS_gpu_text8 accSGNS_gpu_text8.cu $(CUDA_LIB)
accSGNS_gpu_1b: accSGNS_gpu_1b.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o accSGNS_gpu_1b accSGNS_gpu_1b.cu $(CUDA_LIB)
accSGNS_gpu_blogcatalog: accSGNS_gpu_blogcatalog.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o accSGNS_gpu_blogcatalog accSGNS_gpu_blogcatalog.cu $(CUDA_LIB)
accSGNS_gpu_ppi: accSGNS_gpu_ppi.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o accSGNS_gpu_ppi accSGNS_gpu_ppi.cu $(CUDA_LIB)
accSGNS_gpu_wikipedia: accSGNS_gpu_wikipedia.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o accSGNS_gpu_wikipedia accSGNS_gpu_wikipedia.cu $(CUDA_LIB)
accSGNS_gpu_facebook: accSGNS_gpu_facebook.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o accSGNS_gpu_facebook accSGNS_gpu_facebook.cu $(CUDA_LIB)
accSGNS_gpu_astroph: accSGNS_gpu_astroph.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o accSGNS_gpu_astroph accSGNS_gpu_astroph.cu $(CUDA_LIB)

# wombatSGNS_cpu : $(SHARED) wombat/src/main.cpp wombat/src/sgd_trainers/sgd_mkl_trainer.cpp
# 	$(ICPC) $? $(ICFLAGS) -O3 -o wombatSGNS_cpu

clean:
	rm -rf PAR_Word2Vec_cpu PAR_Word2Vec_gpu_text PAR_Word2Vec_gpu_graph pSGNScc_cpu pWord2Vec_cpu word2vec_cpu accSGNS_gpu_text8 accSGNS_gpu_1b accSGNS_gpu_blogcatalog accSGNS_gpu_ppi accSGNS_gpu_wikipedia accSGNS_gpu_facebook accSGNS_gpu_astroph

