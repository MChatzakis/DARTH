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
MODEL_DIRECTORY=/home/mchatzakis/cfaiss/predictor_models

sample=1000
train_queries=10000
experiment_times=1

mode=laet-early-stop-testing
mkdir ${RESULTS_DIRECTORY}/${mode}


dataset_params=(
#  "SIFT100M 32 500 500 1"
#  "GLOVE100 16 500 500 1"
#  "GIST1M 32 500 1000 1"
  "DEEP100M 32 500 750 1"
  #"T2I100M 80 1000 2500 2"
)

k_values=(50)

SIFT100M_tuples_k10=(
  "0.80 241 0.50" 
  "0.85 241 0.65" 
  "0.90 241 0.80" 
#  "0.95 241 1.10" 
#  "0.99 241 2.20" 
) 

SIFT100M_tuples_k25=(
  "0.80 241 0.35" 
  "0.85 241 0.45" 
  "0.90 241 0.55" 
#  "0.95 241 0.80" 
#  "0.99 241 1.70" 
) 

SIFT100M_tuples_k50=(
#  "0.80 241 0.30" 
#  "0.85 241 0.35" 
#  "0.90 241 0.45" 
#  "0.95 241 0.70" 
#  "0.99 241 1.40" 
   "0.90 241 1.90"
) 

SIFT100M_tuples_k75=(
  "0.80 241 0.30" 
  "0.85 241 0.35" 
  "0.90 241 0.45" 
#  "0.95 241 0.65" 
#  "0.99 241 1.30" 
) 

SIFT100M_tuples_k100=(
  "0.80 241 0.25" 
  "0.85 241 0.35" 
  "0.90 241 0.45" 
#  "0.95 241 0.65" 
#  "0.99 241 1.30" 
) 

GLOVE100_tuples_k10=(
  "0.80 200 0.35" 
  "0.85 200 0.55" 
  "0.90 200 0.75" 
#  "0.95 200 1.10" 
#  "0.99 200 3.00" 
) 

GLOVE100_tuples_k25=(
  "0.80 200 0.35" 
  "0.85 200 0.50" 
  "0.90 200 0.60" 
#  "0.95 200 0.80" 
#  "0.99 200 2.15" 
) 

GLOVE100_tuples_k50=(
#  "0.80 200 0.45" 
#  "0.85 200 0.50" 
#  "0.90 200 0.60" 
#  "0.95 200 0.75" 
#  "0.99 200 1.50" 
   "0.90 200 1.50"
) 

GLOVE100_tuples_k75=(
  "0.80 200 0.45" 
  "0.85 200 0.50" 
  "0.90 200 0.60" 
#  "0.95 200 0.75" 
#  "0.99 200 1.25" 
) 

GLOVE100_tuples_k100=(
  "0.80 200 0.45" 
  "0.85 200 0.50" 
  "0.90 200 0.60" 
#  "0.95 200 0.75" 
#  "0.99 200 1.15" 
) 

GIST1M_tuples_k10=(
  "0.80 1260 0.30" 
  "0.85 1260 0.45" 
  "0.90 1260 0.60" 
#  "0.95 1260 1.00" 
#  "0.99 1260 2.45" 
) 

GIST1M_tuples_k25=(
  "0.80 1260 0.25" 
  "0.85 1260 0.35" 
  "0.90 1260 0.45" 
#  "0.95 1260 0.70" 
#  "0.99 1260 1.75" 
) 

GIST1M_tuples_k50=(
#  "0.80 1260 0.25" 
#  "0.85 1260 0.30" 
#  "0.90 1260 0.40" 
#  "0.95 1260 0.60" 
#  "0.99 1260 1.40" 
   "0.90 1260 5.00"
) 

GIST1M_tuples_k75=(
  "0.80 1260 0.25" 
  "0.85 1260 0.30" 
  "0.90 1260 0.40" 
#  "0.95 1260 0.60" 
#  "0.99 1260 1.35" 
) 

GIST1M_tuples_k100=(
  "0.80 1260 0.25" 
  "0.85 1260 0.30" 
  "0.90 1260 0.35" 
#  "0.95 1260 0.55" 
#  "0.99 1260 1.30" 
) 

DEEP100M_tuples_k10=(
  "0.80 368 0.45" 
  "0.85 368 0.55" 
  "0.90 368 0.70" 
#  "0.95 368 1.05" 
#  "0.99 368 2.30" 
) 

DEEP100M_tuples_k25=(
  "0.80 368 0.30" 
  "0.85 368 0.40" 
  "0.90 368 0.50" 
#  "0.95 368 0.80" 
#  "0.99 368 1.65" 
) 

DEEP100M_tuples_k50=(
#  "0.80 368 0.25" 
#  "0.85 368 0.35" 
#  "0.90 368 0.45" 
#  "0.95 368 0.65" 
#  "0.99 368 1.35" 
   "0.90 368 1.00"
) 

DEEP100M_tuples_k75=(
  "0.80 368 0.25" 
  "0.85 368 0.30" 
  "0.90 368 0.40" 
#  "0.95 368 0.60" 
#  "0.99 368 1.35" 
) 

DEEP100M_tuples_k100=(
  "0.80 368 0.25" 
  "0.85 368 0.30" 
  "0.90 368 0.40" 
#  "0.95 368 0.60" 
#  "0.99 368 1.30" 
)

T2I100M_tuples_k10=(
  "0.80 400 0.55" 
  "0.85 400 0.75" 
  "0.90 400 1.25" 
#  "0.95 400 3.15" 
) 

T2I100M_tuples_k25=(
  "0.80 400 0.35" 
  "0.85 400 0.50" 
  "0.90 400 0.80" 
#  "0.95 400 2.05" 
) 

T2I100M_tuples_k50=(
  "0.80 400 0.30" 
  "0.85 400 0.40" 
  "0.90 400 0.70" 
#  "0.95 400 1.90" 
) 

T2I100M_tuples_k75=(
  "0.80 400 0.25" 
  "0.85 400 0.40" 
  "0.90 400 0.60" 
#  "0.95 400 2.10" 
) 

T2I100M_tuples_k100=(
  "0.80 400 0.25" 
  "0.85 400 0.35" 
  "0.90 400 0.55" 
#  "0.95 400 1.45" 
) 

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/noisy

    for k in ${k_values[@]}
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""

        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}
        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/

        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy
        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times/
        
        if [ "$ds" == "SIFT100M" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${SIFT100M_tuples_k1[@]}")
            elif [ "$k" == "5" ]; then
                tuples=("${SIFT100M_tuples_k5[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${SIFT100M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${SIFT100M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${SIFT100M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${SIFT100M_tuples_k75[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${SIFT100M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "DEEP100M" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${DEEP100M_tuples_k1[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${DEEP100M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${DEEP100M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${DEEP100M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${DEEP100M_tuples_k75[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${DEEP100M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GIST1M" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${GIST1M_tuples_k1[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GIST1M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${GIST1M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${GIST1M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${GIST1M_tuples_k75[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GIST1M_tuples_k100[@]}")
            fi
        elif [ "$ds" == "GLOVE100" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${GLOVE100_tuples_k1[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${GLOVE100_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${GLOVE100_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${GLOVE100_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${GLOVE100_tuples_k75[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${GLOVE100_tuples_k100[@]}")
            fi
        elif [ "$ds" == "T2I100M" ]; then
            if [ "$k" == "1" ]; then
                tuples=("${T2I100M_tuples_k1[@]}")
            elif [ "$k" == "10" ]; then
                tuples=("${T2I100M_tuples_k10[@]}")
            elif [ "$k" == "25" ]; then
                tuples=("${T2I100M_tuples_k25[@]}")
            elif [ "$k" == "50" ]; then
                tuples=("${T2I100M_tuples_k50[@]}")
            elif [ "$k" == "75" ]; then
                tuples=("${T2I100M_tuples_k75[@]}")
            elif [ "$k" == "100" ]; then
                tuples=("${T2I100M_tuples_k100[@]}")
            fi
        fi
        
        for tuple in "${tuples[@]}"; do
            read target_recall F multiplier <<< "$tuple"

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
                          --fixed-amount-of-search ${F} --prediction-multiplier ${multiplier} \
                          --query-type testing \
                          --predictor-model-path ${MODEL_DIRECTORY}/laet/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}.txt
              done

              python result_merger.py \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall} \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}.txt \
                  $experiment_times
            fi

            if true; then 
              for noise in 12 #14 16 18 20 22 24 26 28 30
              do
                  echo "noise: ${noise}"
              
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
                              --fixed-amount-of-search ${F} --prediction-multiplier ${multiplier} \
                              --query-type noisy-testing \
                              --gnoise-perc ${noise} \
                              --predictor-model-path ${MODEL_DIRECTORY}/laet/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}.txt
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
