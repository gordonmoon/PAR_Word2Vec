#!/bin/bash

ncores=28 # SET logical cores of your machine

data=../data/text8
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17

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
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_text8
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_text
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 9400

echo "text8 dataset: training time per epoch"
awk '{print "\tWord2Vec-cpu:\t\t" $2/10}' word2vec_cpu_time
awk '{print "\tpWord2Vec-cpu:\t\t" $2/10}' pWord2Vec_cpu_time
awk '{print "\twombatSGNS-cpu:\t\t" $2/10}' wombatSGNS_cpu_time
awk '{print "\tpSGNScc-cpu:\t\t" $2/10}' pSGNScc_cpu_time
awk '{print "\tPAR-Word2Vec-cpu:\t" $2/10}' PAR_Word2Vec_cpu_time
awk '{print "\taccSGNS-gpu:\t\t" $2/10}' accSGNS_gpu_time
awk '{print "\tPAR-Word2Vec-gpu:\t" $2/10}' PAR_Word2Vec_gpu_time


data=../data/1b
niters=10

binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17

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
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_1b
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_text
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 1024 -max-num-sen 30700000

echo "1B-Word dataset: training time per epoch"
awk '{print "\tWord2Vec-cpu:\t\t" $2/10}' word2vec_cpu_time
awk '{print "\tpWord2Vec-cpu:\t\t" $2/10}' pWord2Vec_cpu_time
awk '{print "\twombatSGNS-cpu:\t\t" $2/10}' wombatSGNS_cpu_time
awk '{print "\tpSGNScc-cpu:\t\t" $2/10}' pSGNScc_cpu_time
awk '{print "\tPAR-Word2Vec-cpu:\t" $2/10}' PAR_Word2Vec_cpu_time
awk '{print "\taccSGNS-gpu:\t\t" $2/10}' accSGNS_gpu_time
awk '{print "\tPAR-Word2Vec-gpu:\t" $2/10}' PAR_Word2Vec_gpu_time


data=../data/blog_catalog_random_walks
niters=10

binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

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
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_blogcatalog
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 103500

echo "BlogCatalog dataset: training time per epoch"
awk '{print "\tWord2Vec-cpu:\t\t" $2/10}' word2vec_cpu_time
awk '{print "\tpWord2Vec-cpu:\t\t" $2/10}' pWord2Vec_cpu_time
awk '{print "\twombatSGNS-cpu:\t\t" $2/10}' wombatSGNS_cpu_time
awk '{print "\tpSGNScc-cpu:\t\t" $2/10}' pSGNScc_cpu_time
awk '{print "\tPAR-Word2Vec-cpu:\t" $2/10}' PAR_Word2Vec_cpu_time
awk '{print "\taccSGNS-gpu:\t\t" $2/10}' accSGNS_gpu_time
awk '{print "\tPAR-Word2Vec-gpu:\t" $2/10}' PAR_Word2Vec_gpu_time



data=../data/PPI_random_walks
niters=10

binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

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
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_ppi
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 39500

echo "PPI dataset: training time per epoch"
awk '{print "\tWord2Vec-cpu:\t\t" $2/10}' word2vec_cpu_time
awk '{print "\tpWord2Vec-cpu:\t\t" $2/10}' pWord2Vec_cpu_time
awk '{print "\twombatSGNS-cpu:\t\t" $2/10}' wombatSGNS_cpu_time
awk '{print "\tpSGNScc-cpu:\t\t" $2/10}' pSGNScc_cpu_time
awk '{print "\tPAR-Word2Vec-cpu:\t" $2/10}' PAR_Word2Vec_cpu_time
awk '{print "\taccSGNS-gpu:\t\t" $2/10}' accSGNS_gpu_time
awk '{print "\tPAR-Word2Vec-gpu:\t" $2/10}' PAR_Word2Vec_gpu_time



data=../data/wikipedia_random_walks
niters=10

binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

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
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_wikipedia
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 48000

echo "Wikipedia-2006 dataset: training time per epoch"
awk '{print "\tWord2Vec-cpu:\t\t" $2/10}' word2vec_cpu_time
awk '{print "\tpWord2Vec-cpu:\t\t" $2/10}' pWord2Vec_cpu_time
awk '{print "\twombatSGNS-cpu:\t\t" $2/10}' wombatSGNS_cpu_time
awk '{print "\tpSGNScc-cpu:\t\t" $2/10}' pSGNScc_cpu_time
awk '{print "\tPAR-Word2Vec-cpu:\t" $2/10}' PAR_Word2Vec_cpu_time
awk '{print "\taccSGNS-gpu:\t\t" $2/10}' accSGNS_gpu_time
awk '{print "\tPAR-Word2Vec-gpu:\t" $2/10}' PAR_Word2Vec_gpu_time



data=../data/facebook_random_walks
niters=10

binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

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
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_facebook
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 41000

echo "Facebook dataset: training time per epoch"
awk '{print "\tWord2Vec-cpu:\t\t" $2/10}' word2vec_cpu_time
awk '{print "\tpWord2Vec-cpu:\t\t" $2/10}' pWord2Vec_cpu_time
awk '{print "\twombatSGNS-cpu:\t\t" $2/10}' wombatSGNS_cpu_time
awk '{print "\tpSGNScc-cpu:\t\t" $2/10}' pSGNScc_cpu_time
awk '{print "\tPAR-Word2Vec-cpu:\t" $2/10}' PAR_Word2Vec_cpu_time
awk '{print "\taccSGNS-gpu:\t\t" $2/10}' accSGNS_gpu_time
awk '{print "\tPAR-Word2Vec-gpu:\t" $2/10}' PAR_Word2Vec_gpu_time



data=../data/ASTRO_PH_random_walks
niters=10

binary=../word2vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -cbow 0

binary=../pWord2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21

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
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 21 -C 8 -T 500000

binary=../PAR_Word2Vec_cpu
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24

binary=../accSGNS_gpu_astroph
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -reuse-neg 0 -cbow 0 -hs 0

binary=../PAR_Word2Vec_gpu_graph
numactl --interleave=all $binary -train $data -output vectors.txt -size 128 -window 10 -negative 5 -sample 1e-4 -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 24 -gamma 64 -max-num-sen 188000

echo "ASTRO-PH dataset: training time per epoch"
awk '{print "\tWord2Vec-cpu:\t\t" $2/10}' word2vec_cpu_time
awk '{print "\tpWord2Vec-cpu:\t\t" $2/10}' pWord2Vec_cpu_time
awk '{print "\twombatSGNS-cpu:\t\t" $2/10}' wombatSGNS_cpu_time
awk '{print "\tpSGNScc-cpu:\t\t" $2/10}' pSGNScc_cpu_time
awk '{print "\tPAR-Word2Vec-cpu:\t" $2/10}' PAR_Word2Vec_cpu_time
awk '{print "\taccSGNS-gpu:\t\t" $2/10}' accSGNS_gpu_time
awk '{print "\tPAR-Word2Vec-gpu:\t" $2/10}' PAR_Word2Vec_gpu_time
