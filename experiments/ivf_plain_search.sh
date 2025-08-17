#
# This script generates the baseline runs that do not use Early Termination at all.
# It generated the baseline results both for validation and testing sets
#
cd ..
cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
make -C build -j faiss
make -C build -j ivf_test
cd experiments

echo ""
echo ""
echo "============================="
echo ""
echo ""

sample=1000
experiment_times=2

dataset_params=(
    "GLOVE100 1000 100"
    "GIST1M 1000 200"
    "DEEP100M 10000 150"
    "SIFT100M 10000 150"
)

k_values=(50)

INDEX_DIRECTORY=/data/mchatzakis/index/ivf-index
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/
RESULTS_DIRECTORY=/home/mchatzakis/cfaiss/experiments/ivf-results

for query_type in testing
do
    mode=plain-ivf
    mkdir ${RESULTS_DIRECTORY}/${mode}
    mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}
    for dataset_param in "${dataset_params[@]}"
    do
        read ds nlist nprobe <<< "$dataset_param"

        mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}
        for k in ${k_values[@]}
        do
            echo ""
            echo "--------------- k=${k} ---------------"
            echo ""

            mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}
            mkdir ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/times/
            
            for time in $(seq 1 $experiment_times)
            do
                ./../build/hnsw-test/ivf_test \
                    --dataset ${ds} \
                    --dataset-dir-prefix ${DATASET_DIRECTORY} \
                    --nlist ${nlist} --nprobe ${nprobe} \
                    --query-num ${sample} --k $k \
                    --mode ${mode} \
                    --query-type ${query_type} \
                    --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.nlist${nlist}.index \
                    --output ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/times/nlist${nlist}_nprobe${nprobe}_qs${sample}_t${time}.txt
                    #--save-index \
            done 
                
            python result_merger.py \
                ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/times/nlist${nlist}_nprobe${nprobe}_qs${sample} \
                ${RESULTS_DIRECTORY}/${mode}/${query_type}/${ds}/k${k}/nlist${nlist}_nprobe${nprobe}_qs${sample}.txt \
                $experiment_times

        done

        echo ""
        echo ""

    done
done



