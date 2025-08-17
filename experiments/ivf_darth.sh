cd ../
cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
make -C build -j faiss
make -C build -j ivf_test
cd experiments

echo ""
echo ""
echo "============================="
echo ""
echo ""

INDEX_DIRECTORY=/data/mchatzakis/index/ivf-index
RESULTS_DIRECTORY=/home/mchatzakis/cfaiss/experiments/ivf-results
MODEL_DIRECTORY=/home/mchatzakis/cfaiss/predictor_models
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/

sample=1000
n_estim=100
train_queries=10000
experiment_times=2
mode=darth

dataset_params=(
    "SIFT100M 10000 150 50"
    "GLOVE100 1000 100 20"
    "GIST1M 1000 200 20"
    "DEEP100M 10000 150 50"
)

k_values=(50)

SIFT100M_tuples_k50=(
  "0.80 15750 3150"
  "0.85 21350 4250"
  "0.90 26800 5350"
  "0.95 40950 8200"
  "0.99 58350 11650"
)

DEEP100M_tuples_k50=(
  "0.80 13600 2700"
  "0.85 18000 3600"
  "0.90 22350 4450"
  "0.95 33250 6650"
  "0.99 48200 9650"
)

GIST1M_tuples_k50=(
  "0.80 3640 720"
  "0.85 4740 940"
  "0.90 5840 1160"
  "0.95 8800 1760"
  "0.99 13080 2620"
)

GLOVE100_tuples_k50=(
  "0.80 960 200"
  "0.85 1020 200"
  "0.90 1080 220"
  "0.95 1260 260"
  "0.99 1860 380"
)

mkdir ${RESULTS_DIRECTORY}/${mode}

for dataset_param in "${dataset_params[@]}"
do
    read ds nlist nprobe logging_interval <<< "$dataset_param"

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
              ./../build/hnsw-test/ivf_test \
                --dataset ${ds} \
                --nlist ${nlist} --nprobe ${nprobe} \
                --query-num ${sample} --k $k \
                --output $RESULTS_DIRECTORY/${mode}/${ds}/k${k}/times/nlist${nlist}_nprobe${nprobe}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}.txt \
                --mode ${mode} \
                --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.nlist${nlist}.index \
                --dataset-dir-prefix ${DATASET_DIRECTORY} \
                --target-recall ${target_recall} \
                --initial-prediction-interval ${initial_prediction_interval} \
                --min-prediction-interval ${min_prediction_interval} \
                --query-type testing \
                --predictor-model-path ${MODEL_DIRECTORY}/ivf/darth/${ds}_nlist${nlist}_nprobe${nprobe}_s${train_queries}_k${k}_nestim${n_estim}_lr0.1_li${logging_interval}_rl0_all_feats.txt \
                --logging-interval ${logging_interval}
            done

            python result_merger.py \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/nlist${nlist}_nprobe${nprobe}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
                  ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/nlist${nlist}_nprobe${nprobe}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
                  $experiment_times

            #./../build/hnsw-test/ivf_test \
            #  --dataset ${ds} \
            #  --nlist ${nlist} --nprobe ${nprobe} \
            #  --query-num ${sample} --k $k \
            #  --output $RESULTS_DIRECTORY/${mode}/${ds}/k${k}/detailed/nlist${nlist}_nprobe${nprobe}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
            #  --mode ${mode} \
            #  --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.nlist${nlist}.index \
            #  --dataset-dir-prefix ${DATASET_DIRECTORY} \
            #  --target-recall ${target_recall} \
            #  --initial-prediction-interval ${initial_prediction_interval} \
            #  --min-prediction-interval ${min_prediction_interval} \
            #  --predictor-model-path ${MODEL_DIRECTORY}/ivf/darth/${ds}_nlist${nlist}_nprobe${nprobe}_s${train_queries}_k${k}_nestim${n_estim}_lr0.1_li${logging_interval}_rl0_all_feats.txt \
            #  --per-prediction-logging \
            #  --query-type testing \
            #  --logging-interval ${logging_interval}
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