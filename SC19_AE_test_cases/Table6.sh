#!/bin/bash

ncores=28 # SET logical cores of your machine

data=../data/text8
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output word2vec_cpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output pWord2Vec_cpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17

../wombat/wombatSGNS_cpu \
    -train $data \
    -cbow 0 -size 128 -window 8 -sample 1e-4 \
    -negative 5 \
    -hs 0 \
    -iter $niters \
    -num-threads $ncores \
    -debug 2 \
    -num-phys $ncores

binary=../pSGNScc_cpu
numactl --interleave=all $binary -train $data -output pSGNScc_cpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_text8
numactl --interleave=all $binary -train $data -output accSGNS_gpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_text
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 9400

echo ""
echo "------------------------------------------------------------"
echo "text8 dataset: Extrinsic evaluation results of Word2Vec-cpu"
echo "------------------------------------------------------------"
echo ""
cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../word2vec_cpu_vectors_text8.txt \
    word2vec_cpu_vectors_text8.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    word2vec_cpu_vectors_text8.log \
    --one-line

# cd ..

echo ""
echo "-------------------------------------------------------------"
echo "text8 dataset: Extrinsic evaluation results of pWord2Vec-cpu"
echo "-------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../pWord2Vec_cpu_vectors_text8.txt \
    pWord2Vec_cpu_vectors_text8.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    pWord2Vec_cpu_vectors_text8.log \
    --one-line
    
# cd ..

echo ""
echo "--------------------------------------------------------------"
echo "text8 dataset: Extrinsic evaluation results of wombatSGNS-cpu"
echo "--------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../wombatSGNS_cpu_vectors.txt \
    wombatSGNS_cpu_vectors.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    wombatSGNS_cpu_vectors.log \
    --one-line
    
# cd ..


echo ""
echo "-----------------------------------------------------------"
echo "text8 dataset: Extrinsic evaluation results of pSGNScc-cpu"
echo "-----------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../pSGNScc_cpu_vectors_text8.txt \
    pSGNScc_cpu_vectors_text8.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    pSGNScc_cpu_vectors_text8.log \
    --one-line
    
# cd ..


echo ""
echo "----------------------------------------------------------------"
echo "text8 dataset: Extrinsic evaluation results of PAR-Word2Vec-cpu"
echo "----------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../PAR_Word2Vec_cpu_vectors_text8.txt \
    PAR_Word2Vec_cpu_vectors_text8.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    PAR_Word2Vec_cpu_vectors_text8.log \
    --one-line
    
# cd ..


echo ""
echo "-----------------------------------------------------------"
echo "text8 dataset: Extrinsic evaluation results of accSGNS-gpu"
echo "-----------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../accSGNS_gpu_vectors_text8.txt \
    accSGNS_gpu_vectors_text8.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    accSGNS_gpu_vectors_text8.log \
    --one-line
    
# cd ..


echo ""
echo "----------------------------------------------------------------"
echo "text8 dataset: Extrinsic evaluation results of PAR-Word2Vec-gpu"
echo "----------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../PAR_Word2Vec_gpu_vectors_text8.txt \
    PAR_Word2Vec_gpu_vectors_text8.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    PAR_Word2Vec_gpu_vectors_text8.log \
    --one-line
    
cd ..


data=../data/1b
niters=10

binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output word2vec_cpu_vectors_1b.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output pWord2Vec_cpu_vectors_1b.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17

../wombat/wombatSGNS_cpu \
    -train $data \
    -cbow 0 -size 128 -window 8 -sample 1e-4 \
    -negative 5 \
    -hs 0 \
    -iter $niters \
    -num-threads $ncores \
    -debug 2 \
    -num-phys $ncores

binary=../pSGNScc_cpu
numactl --interleave=all $binary -train $data -output pSGNScc_cpu_vectors_1b.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_1b.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_1b
numactl --interleave=all $binary -train $data -output accSGNS_gpu_vectors_1b.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_text
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_1b.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 1024 -max-num-sen 30700000

echo ""
echo "--------------------------------------------------------------"
echo "1B-Word dataset: Extrinsic evaluation results of Word2Vec-cpu"
echo "--------------------------------------------------------------"
echo ""
cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../word2vec_cpu_vectors_1b.txt \
    word2vec_cpu_vectors_1b.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    word2vec_cpu_vectors_1b.log \
    --one-line
    
# cd ..


echo ""
echo "---------------------------------------------------------------"
echo "1B-Word dataset: Extrinsic evaluation results of pWord2Vec-cpu"
echo "---------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../pWord2Vec_cpu_vectors_1b.txt \
    pWord2Vec_cpu_vectors_1b.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    pWord2Vec_cpu_vectors_1b.log \
    --one-line
    
# cd ..


echo ""
echo "----------------------------------------------------------------"
echo "1B-Word dataset: Extrinsic evaluation results of wombatSGNS-cpu"
echo "----------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../wombatSGNS_cpu_vectors.txt \
    wombatSGNS_cpu_vectors.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    wombatSGNS_cpu_vectors.log \
    --one-line
    
# cd ..


echo ""
echo "-------------------------------------------------------------"
echo "1B-Word dataset: Extrinsic evaluation results of pSGNScc-cpu"
echo "-------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../pSGNScc_cpu_vectors_1b.txt \
    pSGNScc_cpu_vectors_1b.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    pSGNScc_cpu_vectors_1b.log \
    --one-line
    
# cd ..


echo ""
echo "------------------------------------------------------------------"
echo "1B-Word dataset: Extrinsic evaluation results of PAR-Word2Vec-cpu"
echo "------------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../PAR_Word2Vec_cpu_vectors_1b.txt \
    PAR_Word2Vec_cpu_vectors_1b.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    PAR_Word2Vec_cpu_vectors_1b.log \
    --one-line
    
# cd ..


echo ""
echo "-------------------------------------------------------------"
echo "1B-Word dataset: Extrinsic evaluation results of accSGNS-gpu"
echo "-------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../accSGNS_gpu_vectors_1b.txt \
    accSGNS_gpu_vectors_1b.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    accSGNS_gpu_vectors_1b.log \
    --one-line
    
# cd ..


echo ""
echo "------------------------------------------------------------------"
echo "1B-Word dataset: Extrinsic evaluation results of PAR-Word2Vec-gpu"
echo "------------------------------------------------------------------"
echo ""
# cd extrinsic_evaluation_word_embeddings

./run_extrinsic_word_experiments.sh \
    ../PAR_Word2Vec_gpu_vectors_1b.txt \
    PAR_Word2Vec_gpu_vectors_1b.log \
    ${HOME}/envs/keras/bin/python

./get_stats.sh \
    PAR_Word2Vec_gpu_vectors_1b.log \
    --one-line
    
cd ..