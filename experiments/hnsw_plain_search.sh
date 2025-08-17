#
# This script generates the baseline runs that do not use Early Termination at all.
# It generated the baseline results both for validation and testing sets
#
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

sample=1000
experiment_times=1

dataset_params=(
    "GIST1M 32 500 1000"
    "GLOVE100 16 500 500"
    "SIFT100M 32 500 500"
    "DEEP100M 32 500 750"
)

k_values=(50)

INDEX_DIRECTORY=/data/mchatzakis/index/hnsw-index
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/
RESULTS_DIRECTORY=/home/mchatzakis/cfaiss/experiments/results

for query_type in testing
do
    mode=no-early-stop
    mkdir ${RESULTS_DIRECTORY}/${mode}
    mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}
    for dataset_param in "${dataset_params[@]}"
    do
        read ds M efC efS <<< "$dataset_param"

        mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}
        for k in ${k_values[@]}
        do
            echo ""
            echo "--------------- k=${k} ---------------"
            echo ""

            mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}
            mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/times/
            mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/noisy/
            mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/noisy/times
            
            if false; then 
                for time in $(seq 1 $experiment_times)
                do
                    ./../build/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${sample} --k $k \
                        --mode ${mode} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                        --query-type ${query_type} \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --output ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_t${time}.txt
                done 
                
                python result_merger.py \
                    ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample} \
                    ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}.txt \
                    $experiment_times

                ./../build/hnsw-test/hnsw_test \
                    --dataset ${ds} \
                    --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                    --query-num ${sample} --k $k \
                    --output ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/detailed.M${M}_efC${efC}_efS${efS}_qs${sample}.txt \
                    --mode optimal-early-stop-testing \
                    --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                    --query-type ${query_type} \
                    --dataset-dir-prefix ${DATASET_DIRECTORY}
            fi
            
            if true; then
                for noise in 14 16 18 20 22 24 26 28 30
                do
                    echo "noise: ${noise}"
                
                    for time in $(seq 1 $experiment_times)
                    do
                        ./../build/hnsw-test/hnsw_test \
                            --dataset ${ds} \
                            --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                            --query-num ${sample} --k $k \
                            --output ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/noisy/times/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_t${time}.txt \
                            --mode ${mode} \
                            --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                            --dataset-dir-prefix ${DATASET_DIRECTORY} \
                            --query-type noisy-testing \
                            --gnoise-perc ${noise}
                    done

                    python result_merger.py \
                        ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/noisy/times/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample} \
                        ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/noisy/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}.txt \
                        $experiment_times
                done
            fi

        done

        echo ""
        echo ""

    done
done



