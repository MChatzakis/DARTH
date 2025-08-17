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

sample=10

dataset_params=(
    "GIST1M 1000 200 20"
    #"GIST1M 1000 200 10"
    #"SIFT100M 32 500 500"
    "GLOVE100 1000 100 20"
    "DEEP100M 10000 150 50"
    #"SIFT100M 10000 150 50"
)

k_values=(50)

INDEX_DIRECTORY=/data/mchatzakis/index/ivf-index
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/
TRAINING_DATA_DIRECTORY=/data/mchatzakis/et_training_data/ivf/test-fast-calc

for query_type in training
do
    mode=training-data-generation
    mkdir ${TRAINING_DATA_DIRECTORY}/${mode}
    mkdir ${TRAINING_DATA_DIRECTORY}/${mode}/${query_type}
    for dataset_param in "${dataset_params[@]}"
    do
        read ds nlist nprobe logging_interval <<< "$dataset_param"

        mkdir ${TRAINING_DATA_DIRECTORY}/${mode}/${query_type}/${ds}
        for k in ${k_values[@]}
        do
            mkdir ${TRAINING_DATA_DIRECTORY}/${mode}/${query_type}/${ds}/${k}

            echo ""
            echo "--------------- k=${k} ---------------"
            echo ""
            
            ./../build/hnsw-test/ivf_test \
                --dataset ${ds} \
                --dataset-dir-prefix ${DATASET_DIRECTORY} \
                --nlist ${nlist} --nprobe ${nprobe} \
                --query-num ${sample} --k $k \
                --logging-interval ${logging_interval} \
                --mode ${mode} \
                --query-type ${query_type} \
                --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.nlist${nlist}.index \
                --output ${TRAINING_DATA_DIRECTORY}/${mode}/${query_type}/${ds}/${k}/nlist${nlist}_nprobe${nprobe}_qs${sample}_li${logging_interval}.txt
        done

        echo ""
        echo ""

    done
done



