cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
make -C build -j faiss
make -C build -j hnsw_test

echo ""
echo ""
echo "============================="
echo ""
echo ""

dataset_params=(
    "GIST1M 32 500 1000"
    "GLOVE100 16 500 500"
    "SIFT100M 32 500 500"
    "DEEP100M 32 500 750"
    "T2I100M 80 1000 2500"
)

INDEX_DIRECTORY=/home/mchatzakis/hnsw-index
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/

mode=no-early-stop
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS <<< "$dataset_param"

    ./build/hnsw-test/hnsw_test \
        --dataset ${ds} \
        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
        --query-num ${sample} --k 100 \
        --mode ${mode} \
        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
        --query-type ${query_type} \
        --dataset-dir-prefix ${DATASET_DIRECTORY} \
        --save-index
done
