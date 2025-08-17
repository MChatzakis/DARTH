
cd ..
cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
make -C build -j faiss
make -C build -j hnsw_test
cd experiments

echo ""
echo ""
echo "============================="
echo ""
echo ""

INDEX_DIRECTORY=/data/mchatzakis/index/hnsw-index
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/
RESULTS_DIRECTORY=/home/mchatzakis/cfaiss/experiments/results

sample=1000
experiment_times=1

dataset_params=(
#  "SIFT100M 32 500 500"
#  "GLOVE100 16 500 500"
#  "GIST1M 32 500 1000"
  "DEEP100M 32 500 750"
#  "T2I100M 80 1000 2500"
)

k_values=(50)

mode=naive-early-stop-testing
mkdir ${RESULTS_DIRECTORY}/${mode}


SIFT100M_tuples_k10=(
  "0.80 1100" 
  "0.85 1564" 
  "0.90 1564" 
#  "0.95 2649" 
#  "0.99 2649" 
) 

SIFT100M_tuples_k25=(
  "0.80 1479" 
  "0.85 2049" 
  "0.90 2514" 
#  "0.95 3247" 
#  "0.99 4628" 
) 

SIFT100M_tuples_k50=(
#  "0.80 1891" 
#  "0.85 2437" 
#  "0.90 2979" 
#  "0.95 4428" 
#  "0.99 6566" 
   "0.90 25000"
) 

SIFT100M_tuples_k75=(
  "0.80 2215" 
  "0.85 2780" 
  "0.90 3641" 
#  "0.95 5222" 
#  "0.99 7670" 
) 

SIFT100M_tuples_k100=(
  "0.80 2491" 
  "0.85 3082" 
  "0.90 3958" 
#  "0.95 5453" 
#  "0.99 7602" 
) 

GLOVE100_tuples_k10=(
  "0.80 317" 
  "0.85 423" 
  "0.90 423" 
#  "0.95 639" 
#  "0.99 639" 
) 

GLOVE100_tuples_k25=(
  "0.80 352" 
  "0.85 431" 
  "0.90 492" 
#  "0.95 591" 
#  "0.99 794" 
) 

GLOVE100_tuples_k50=(
#  "0.80 414" 
#  "0.85 483" 
#  "0.90 543" 
#  "0.95 681" 
#  "0.99 978" 
  "0.90 8000"
) 

GLOVE100_tuples_k75=(
  "0.80 514" 
  "0.85 583" 
  "0.90 674" 
#  "0.95 822" 
#  "0.99 1172" 
) 

GLOVE100_tuples_k100=(
  "0.80 610" 
  "0.85 691" 
  "0.90 789" 
#  "0.95 913" 
#  "0.99 1183" 
) 

GIST1M_tuples_k10=(
  "0.80 1716" 
  "0.85 2547" 
  "0.90 2547" 
#  "0.95 4317" 
#  "0.99 4317" 
) 

GIST1M_tuples_k25=(
  "0.80 2213" 
  "0.85 3137" 
  "0.90 3903" 
#  "0.95 5093" 
#  "0.99 7063" 
) 

GIST1M_tuples_k50=(
#  "0.80 2684" 
#  "0.85 3477" 
#  "0.90 4289" 
#  "0.95 6536" 
#  "0.99 9475" 
  "0.90 30000"
) 

GIST1M_tuples_k75=(
  "0.80 3021" 
  "0.85 3814" 
  "0.90 5080" 
#  "0.95 7486" 
#  "0.99 10770" 
) 

GIST1M_tuples_k100=(
  "0.80 3313" 
  "0.85 4116" 
  "0.90 5342" 
#  "0.95 7565" 
#  "0.99 10730" 
) 

DEEP100M_tuples_k10=(
  "0.80 1318" 
  "0.85 1943" 
  "0.90 1943" 
#  "0.95 3312" 
#  "0.99 3312" 
) 

DEEP100M_tuples_k25=(
  "0.80 1801" 
  "0.85 2516" 
  "0.90 3105" 
#  "0.95 4043" 
#  "0.99 5663" 
) 

DEEP100M_tuples_k50=(
#  "0.80 2288" 
#  "0.85 2972" 
#  "0.90 3638" 
#  "0.95 5405" 
#  "0.99 8021" 
  "0.90 20000"
) 

DEEP100M_tuples_k75=(
  "0.80 2675" 
  "0.85 3366" 
  "0.90 4418" 
#  "0.95 6311" 
#  "0.99 9287" 
) 

DEEP100M_tuples_k100=(
  "0.80 2997" 
  "0.85 3726" 
  "0.90 4774" 
#  "0.95 6563" 
#  "0.99 9237" 
) 

T2I100M_tuples_k10=(
  "0.80 4827" 
  "0.85 6492" 
  "0.90 6492" 
#  "0.95 10191" 
#  "0.99 10191" 
) 

T2I100M_tuples_k25=(
  "0.80 5557" 
  "0.85 7402" 
  "0.90 8833" 
#  "0.95 11253" 
#  "0.99 15635" 
) 

T2I100M_tuples_k50=(
  "0.80 6415" 
  "0.85 7955" 
  "0.90 9532" 
#  "0.95 13671" 
#  "0.99 20747" 
) 

T2I100M_tuples_k75=(
  "0.80 7032" 
  "0.85 8610" 
  "0.90 11054" 
#  "0.95 15482" 
#  "0.99 23835" 
) 

T2I100M_tuples_k100=(
  "0.80 7485" 
  "0.85 9075" 
  "0.90 11500" 
#  "0.95 15555" 
#  "0.99 22891" 
) 

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS <<< "$dataset_param"
    
    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/noisy/

    for k in ${k_values[@]}
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

        if [ "$ds" == "SIFT100M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${SIFT100M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${SIFT100M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${SIFT100M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${SIFT100M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${SIFT100M_tuples_k75[@]}")
            fi
        elif [ "$ds" == "DEEP100M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${DEEP100M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${DEEP100M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${DEEP100M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${DEEP100M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${DEEP100M_tuples_k75[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GIST1M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${GIST1M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${GIST1M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${GIST1M_tuples_k75[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GLOVE100_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${GLOVE100_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${GLOVE100_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${GLOVE100_tuples_k75[@]}")
            fi
        elif [ "$ds" == "T2I100M" ]; then
            if [ "$k" == "100" ]; then
                tuples=("${T2I100M_tuples_k100[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${T2I100M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${T2I100M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${T2I100M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${T2I100M_tuples_k75[@]}")
            fi
        fi

        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}
        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/
        for tuple in "${tuples[@]}"; do
            read target_recall dist_threshold <<< "$tuple"

            if false; then
              for time in $(seq 1 $experiment_times)
              do
                  ./../build/hnsw-test/hnsw_test \
                      --dataset ${ds} \
                      --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                      --query-num ${sample} --k $k \
                      --output ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt \
                      --mode ${mode} \
                      --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                      --dataset-dir-prefix ${DATASET_DIRECTORY} \
                      --target-recall ${target_recall} \
                      --initial-prediction-interval ${dist_threshold} \
                      --query-type testing
              done

              python result_merger.py \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall} \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}.txt \
                  $experiment_times
            fi

            if true; then
              mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/
              mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times/
              for noise in 12 #14 16 18 20 22 24 26 28 30
              do
                  for time in $(seq 1 $experiment_times)
                  do
                      ./../build/hnsw-test/hnsw_test \
                          --dataset ${ds} \
                          --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                          --query-num ${sample} --k $k \
                          --output ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times/tuned_noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt \
                          --mode ${mode} \
                          --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                          --dataset-dir-prefix ${DATASET_DIRECTORY} \
                          --target-recall ${target_recall} \
                          --initial-prediction-interval ${dist_threshold} \
                          --query-type noisy-testing \
                          --gnoise ${noise}
                  done

                  python result_merger.py \
                      ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times/tuned_noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall} \
                      ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/tuned_noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}.txt \
                      $experiment_times
              done
            fi
        
        done
        
    done

    echo ""
    echo ""
done
