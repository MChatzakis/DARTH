cd ../
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
RESULTS_DIRECTORY=/home/mchatzakis/cfaiss/experiments/results
EXECUTABLE_FILE=/home/mchatzakis/cfaiss/build/hnsw-test/hnsw_test
MODEL_DIRECTORY=/home/mchatzakis/cfaiss/predictor_models
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/

sample=1000
mode=early-stop-testing
n_estim=100
train_queries=10000
experiment_times=1

dataset_params=(
  "DEEP100M 32 500 750 1"
  #"GLOVE100 16 500 500 1"
  #"GIST1M 32 500 1000 1"
  #"SIFT100M 32 500 500 1"
  #"T2I100M 80 1000 2500 2"
)

k_values=(50)

SIFT100M_tuples_k10=(
  "0.80 550 110" 
  "0.85 550 110" 
  "0.90 782 156" 
  #"0.95 782 156" 
  #"0.99 1325 265" 
) 

SIFT100M_tuples_k25=(
  "0.80 740 148" 
  "0.85 740 148" 
  "0.90 1257 251" 
  #"0.95 1257 251" 
  #"0.99 2314 463" 
) 

SIFT100M_tuples_k50=(
  "0.80 946 189" 
  "0.85 946 189" 
  "0.90 1489 298" 
  #"0.95 1489 298" 
  #"0.99 3283 657" 
) 

SIFT100M_tuples_k75=(
  "0.80 1108 222" 
  "0.85 1108 222" 
  "0.90 1820 364" 
  #"0.95 1820 364" 
  #"0.99 3835 767" 
) 

SIFT100M_tuples_k100=(
  "0.80 1246 249" 
  "0.85 1246 249" 
  "0.90 1979 396" 
  #"0.95 1979 396" 
  #"0.99 4131 826" 
) 

GLOVE100_tuples_k10=(
  "0.80 158 32" 
  "0.85 158 32" 
  "0.90 212 42" 
  #"0.95 212 42" 
  #"0.99 319 64" 
) 

GLOVE100_tuples_k25=(
  "0.80 176 35" 
  "0.85 176 35" 
  "0.90 246 49" 
  #"0.95 246 49" 
  #"0.99 397 79" 
) 

GLOVE100_tuples_k50=(
  "0.80 207 41" 
  "0.85 207 41" 
  "0.90 272 54" 
  #"0.95 272 54" 
  #"0.99 489 98" 
) 

GLOVE100_tuples_k75=(
  "0.80 257 51" 
  "0.85 257 51" 
  "0.90 337 67" 
  #"0.95 337 67" 
  #"0.99 586 117" 
) 

GLOVE100_tuples_k100=(
  "0.80 305 61" 
  "0.85 305 61" 
  "0.90 395 79" 
  #"0.95 395 79" 
  #"0.99 711 142" 
) 

GIST1M_tuples_k10=(
  "0.80 858 172" 
  "0.85 858 172" 
  "0.90 1273 255" 
  #"0.95 1273 255" 
  #"0.99 2158 432" 
) 

GIST1M_tuples_k25=(
  "0.80 1106 221" 
  "0.85 1106 221" 
  "0.90 1951 390" 
  #"0.95 1951 390" 
  #"0.99 3532 706" 
) 

GIST1M_tuples_k50=(
  "0.80 1342 268" 
  "0.85 1342 268" 
  "0.90 2145 429" 
  #"0.95 2145 429" 
  #"0.99 4737 947" 
) 

GIST1M_tuples_k75=(
  "0.80 1511 302" 
  "0.85 1511 302" 
  "0.90 2540 508" 
  #"0.95 2540 508" 
  #"0.99 5385 1077" 
) 

GIST1M_tuples_k100=(
  "0.80 1657 331" 
  "0.85 1657 331" 
  "0.90 2671 534" 
  #"0.95 2671 534" 
  #"0.99 5746 1149" 
) 

DEEP100M_tuples_k10=(
  "0.80 659 132" 
  "0.85 659 132" 
  "0.90 972 194" 
  #"0.95 972 194" 
  #"0.99 1656 331" 
) 

DEEP100M_tuples_k25=(
  "0.80 900 180" 
  "0.85 900 180" 
  "0.90 1552 310" 
  #"0.95 1552 310" 
  #"0.99 2831 566" 
) 

DEEP100M_tuples_k50=(
  #"0.80 1144 229" 
  #"0.85 1144 229" 
  #"0.90 1819 364" 
  #"0.95 1819 364" 
  #"0.99 4011 802" 
  #"0.996 4011 802"
  "1.0 4011 802"
) 

DEEP100M_tuples_k75=(
  "0.80 1337 267" 
  "0.85 1337 267" 
  "0.90 2209 442" 
  #"0.95 2209 442" 
  #"0.99 4643 929" 
) 

DEEP100M_tuples_k100=(
  "0.80 1498 300" 
  "0.85 1498 300" 
  "0.90 2387 477" 
#  "0.95 2387 477" 
#  "0.99 5076 1015" 
) 

T2I100M_tuples_k10=(
  "0.80 2413 483" 
  "0.85 2413 483" 
  "0.90 3246 649" 
#  "0.95 3246 649" 
#  "0.99 5096 1019" 
) 

T2I100M_tuples_k25=(
  "0.80 2778 556" 
  "0.85 2778 556" 
  "0.90 4416 883" 
#  "0.95 4416 883" 
#  "0.99 7818 1564" 
) 

T2I100M_tuples_k50=(
  "0.80 3208 642" 
  "0.85 3208 642" 
  "0.90 4766 953" 
#  "0.95 4766 953" 
#  "0.99 10373 2075" 
) 

T2I100M_tuples_k75=(
  "0.80 3516 703" 
  "0.85 3516 703" 
  "0.90 5527 1105" 
#  "0.95 5527 1105" 
#  "0.99 11918 2384" 
) 

T2I100M_tuples_k100=(
  "0.80 3742 748" 
  "0.85 3742 748" 
  "0.90 5750 1150" 
#  "0.95 5750 1150" 
#  "0.99 13070 2614" 
) 

mkdir ${RESULTS_DIRECTORY}/${mode}

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}

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
            elif [ "$k" == "5" ]; then
                tuples=("${SIFT100M_tuples_k5[@]}")
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
        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/detailed/
        mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy

        for tuple in "${tuples[@]}"; do
          read target_recall initial_prediction_interval min_prediction_interval <<< "$tuple"

          if true; then 
            for time in $(seq 1 $experiment_times)
            do
              ./../build/hnsw-test/hnsw_test \
                --dataset ${ds} \
                --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                --query-num ${sample} --k $k \
                --output $RESULTS_DIRECTORY/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}.txt \
                --mode ${mode} \
                --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                --dataset-dir-prefix ${DATASET_DIRECTORY} \
                --target-recall ${target_recall} \
                --initial-prediction-interval ${initial_prediction_interval} \
                --min-prediction-interval ${min_prediction_interval} \
                --query-type testing \
                --predictor-model-path ${MODEL_DIRECTORY}/darth/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt
            done

            python result_merger.py \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
                  $experiment_times

            ./../build/hnsw-test/hnsw_test \
              --dataset ${ds} \
              --M ${M} --efConstruction ${efC} --efSearch ${efS} \
              --query-num ${sample} --k $k \
              --output $RESULTS_DIRECTORY/${mode}/${ds}/k${k}/detailed/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
              --mode ${mode} \
              --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
              --dataset-dir-prefix ${DATASET_DIRECTORY} \
              --target-recall ${target_recall} \
              --initial-prediction-interval ${initial_prediction_interval} \
              --min-prediction-interval ${min_prediction_interval} \
              --predictor-model-path ${MODEL_DIRECTORY}/darth/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt \
              --per-prediction-logging \
              --query-type testing
          fi

          if false; then 
            mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times
            for noise in 14 16 18 20 22 24 26 28 30
            do
                for time in $(seq 1 $experiment_times)
                do
                    ./../build/hnsw-test/hnsw_test \
                      --dataset ${ds} \
                      --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                      --query-num ${sample} --k $k \
                      --output ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}.txt \
                      --mode ${mode} \
                      --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                      --dataset-dir-prefix ${DATASET_DIRECTORY} \
                      --target-recall ${target_recall} \
                      --initial-prediction-interval ${initial_prediction_interval} \
                      --min-prediction-interval ${min_prediction_interval} \
                      --query-type noisy-testing \
                      --predictor-model-path ${MODEL_DIRECTORY}/darth/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt \
                      --gnoise ${noise}
                done

                python result_merger.py \
                    ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
                    ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
                    $experiment_times
            done
          fi

        done
        
    done

    echo ""
    echo ""
done