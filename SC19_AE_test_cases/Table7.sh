#!/bin/bash

ncores=28 # SET logical cores of your machine

data=../data/blog_catalog_random_walks
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output word2vec_cpu_vectors_blogcatalog.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output pWord2Vec_cpu_vectors_blogcatalog.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

../wombat/wombatSGNS_cpu \
    -train $data \
 	-cbow 0 -size 128 -window 10 -sample 1e-4 \
	-negative 5 \
	-hs 0 \
	-iter $niters \
	-num-threads $ncores \
	-debug 2 \
	-num-phys $ncores

binary=../pSGNScc_cpu
numactl --interleave=all $binary -train $data -output pSGNScc_cpu_vectors_blogcatalog.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_blogcatalog.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_blogcatalog
numactl --interleave=all $binary -train $data -output accSGNS_gpu_vectors_blogcatalog.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_blogcatalog.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 103500

# echo ""
# echo "------------------------------------------------------------"
# echo "BlogCatalog dataset: Extrinsic evaluation results"
# echo "------------------------------------------------------------"
# echo ""

cd extrinsic_evaluation_graph_embeddings

python Extrinsic_eval_blogcatalog.py

cd ..

data=../data/PPI_random_walks
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output word2vec_cpu_vectors_ppi.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output pWord2Vec_cpu_vectors_ppi.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

../wombat/wombatSGNS_cpu \
    -train $data \
    -cbow 0 -size 128 -window 10 -sample 1e-4 \
    -negative 5 \
    -hs 0 \
    -iter $niters \
    -num-threads $ncores \
    -debug 2 \
    -num-phys $ncores

binary=../pSGNScc_cpu
numactl --interleave=all $binary -train $data -output pSGNScc_cpu_vectors_ppi.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_ppi.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_ppi
numactl --interleave=all $binary -train $data -output accSGNS_gpu_vectors_ppi.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_ppi.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 39500

# echo ""
# echo "------------------------------------------------------------"
# echo "PPI dataset: Extrinsic evaluation results"
# echo "------------------------------------------------------------"
# echo ""

cd extrinsic_evaluation_graph_embeddings

python Extrinsic_eval_PPI.py

cd ..


data=../data/wikipedia_random_walks
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output word2vec_cpu_vectors_wikipedia.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output pWord2Vec_cpu_vectors_wikipedia.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

../wombat/wombatSGNS_cpu \
    -train $data \
    -cbow 0 -size 128 -window 10 -sample 1e-4 \
    -negative 5 \
    -hs 0 \
    -iter $niters \
    -num-threads $ncores \
    -debug 2 \
    -num-phys $ncores

binary=../pSGNScc_cpu
numactl --interleave=all $binary -train $data -output pSGNScc_cpu_vectors_wikipedia.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_wikipedia.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_wikipedia
numactl --interleave=all $binary -train $data -output accSGNS_gpu_vectors_wikipedia.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_wikipedia.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 48000


# echo ""
# echo "------------------------------------------------------------"
# echo "Wikipedia-2006 dataset: Extrinsic evaluation results"
# echo "------------------------------------------------------------"
# echo ""

cd extrinsic_evaluation_graph_embeddings

python Extrinsic_eval_wikipedia.py

cd ..


data=../data/facebook_random_walks
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output word2vec_cpu_vectors_facebook.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output pWord2Vec_cpu_vectors_facebook.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

../wombat/wombatSGNS_cpu \
    -train $data \
    -cbow 0 -size 128 -window 10 -sample 1e-4 \
    -negative 5 \
    -hs 0 \
    -iter $niters \
    -num-threads $ncores \
    -debug 2 \
    -num-phys $ncores

binary=../pSGNScc_cpu
numactl --interleave=all $binary -train $data -output pSGNScc_cpu_vectors_facebook.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_facebook.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_facebook
numactl --interleave=all $binary -train $data -output accSGNS_gpu_vectors_facebook.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_facebook.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 41000

# echo ""
# echo "------------------------------------------------------------"
# echo "Facebook dataset: Extrinsic evaluation results"
# echo "------------------------------------------------------------"
# echo ""

cd extrinsic_evaluation_graph_embeddings

python Extrinsic_eval_facebook.py

cd ..


data=../data/ASTRO_PH_random_walks
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output word2vec_cpu_vectors_astroph.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output pWord2Vec_cpu_vectors_astroph.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

../wombat/wombatSGNS_cpu \
    -train $data \
    -cbow 0 -size 128 -window 10 -sample 1e-4 \
    -negative 5 \
    -hs 0 \
    -iter $niters \
    -num-threads $ncores \
    -debug 2 \
    -num-phys $ncores

binary=../pSGNScc_cpu
numactl --interleave=all $binary -train $data -output pSGNScc_cpu_vectors_astroph.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_astroph.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_astroph
numactl --interleave=all $binary -train $data -output accSGNS_gpu_vectors_astroph.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_astroph.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 188000

# echo ""
# echo "------------------------------------------------------------"
# echo "ASTRO-PH dataset: Extrinsic evaluation results"
# echo "------------------------------------------------------------"
# echo ""

cd extrinsic_evaluation_graph_embeddings

python Extrinsic_eval_astroph.py

cd ..