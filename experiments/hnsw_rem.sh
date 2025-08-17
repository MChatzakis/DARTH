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
  "SIFT100M 32 500 500"
  "GLOVE100 16 500 500"
  "GIST1M 32 500 1000"
  "DEEP100M 32 500 750"
  #"T2I100M 80 1000 2500"
)

k_values=(50)

SIFT100M_tuples_k10=(
  "0.80 50" 
  "0.85 50" 
  "0.90 75" 
#  "0.95 100" 
#  "0.99 200" 
) 

SIFT100M_tuples_k25=(
  "0.80 50" 
  "0.85 50" 
  "0.90 75" 
#  "0.95 125" 
#  "0.99 300" 
) 

SIFT100M_tuples_k50=(
  "0.80 50" 
  "0.85 75" 
  "0.90 100" 
#  "0.95 150" 
#  "0.99 350" 
) 

SIFT100M_tuples_k75=(
  "0.80 75" 
  "0.85 75" 
  "0.90 125" 
#  "0.95 175" 
#  "0.99 400" 
) 

SIFT100M_tuples_k100=(
  "0.80 75" 
  "0.85 100" 
  "0.90 125" 
#  "0.95 200" 
#  "0.99 450" 
) 

GLOVE100_tuples_k10=(
  "0.80 25" 
  "0.85 25" 
  "0.90 50" 
#  "0.95 75" 
#  "0.99 450" 
) 

GLOVE100_tuples_k25=(
  "0.80 25" 
  "0.85 25" 
  "0.90 25" 
#  "0.95 50" 
#  "0.99 375" 
) 

GLOVE100_tuples_k50=(
  "0.80 25" 
  "0.85 50" 
  "0.90 50" 
#  "0.95 50" 
#  "0.99 300" 
) 

GLOVE100_tuples_k75=(
  "0.80 50" 
  "0.85 50" 
  "0.90 50" 
#  "0.95 75" 
#  "0.99 275" 
) 

GLOVE100_tuples_k100=(
  "0.80 50" 
  "0.85 75" 
  "0.90 75" 
#  "0.95 100" 
#  "0.99 250" 
) 

GIST1M_tuples_k10=(
  "0.80 50" 
  "0.85 50" 
  "0.90 75" 
#  "0.95 150" 
#  "0.99 475" 
) 

GIST1M_tuples_k25=(
  "0.80 50" 
  "0.85 75" 
  "0.90 100" 
#  "0.95 175" 
#  "0.99 575" 
) 

GIST1M_tuples_k50=(
  "0.80 75" 
  "0.85 100" 
  "0.90 125" 
#  "0.95 225" 
#  "0.99 650" 
) 

GIST1M_tuples_k75=(
  "0.80 75" 
  "0.85 100" 
  "0.90 150" 
#  "0.95 250" 
#  "0.99 750" 
) 

GIST1M_tuples_k100=(
  "0.80 100" 
  "0.85 125" 
  "0.90 175" 
#  "0.95 275" 
#  "0.99 800" 
) 

DEEP100M_tuples_k10=(
  "0.80 25" 
  "0.85 50" 
  "0.90 50" 
#  "0.95 100" 
#  "0.99 225" 
) 

DEEP100M_tuples_k25=(
  "0.80 50" 
  "0.85 50" 
  "0.90 75" 
#  "0.95 125" 
#  "0.99 325" 
) 

DEEP100M_tuples_k50=(
  "0.80 50" 
  "0.85 75" 
  "0.90 100" 
#  "0.95 150" 
#  "0.99 400" 
) 

DEEP100M_tuples_k75=(
  "0.80 75" 
  "0.85 75" 
  "0.90 125" 
#  "0.95 175" 
#  "0.99 475" 
) 

DEEP100M_tuples_k100=(
  "0.80 75" 
  "0.85 100" 
  "0.90 125" 
#  "0.95 200" 
#  "0.99 550" 
)

T2I100M_tuples_k10=(
  "0.80 90" 
  "0.85 140" 
  "0.90 270" 
#  "0.95 1080" 
) 

T2I100M_tuples_k25=(
  "0.80 100" 
  "0.85 160" 
  "0.90 320" 
#  "0.95 1340" 
) 

T2I100M_tuples_k50=(
  "0.80 120" 
  "0.85 190" 
  "0.90 380" 
#  "0.95 1810" 
) 

T2I100M_tuples_k75=(
  "0.80 140" 
  "0.85 210" 
  "0.90 420" 
#  "0.95 1990" 
) 

T2I100M_tuples_k100=(
  "0.80 150" 
  "0.85 220" 
  "0.90 420" 
#  "0.95 1690" 
) 

mode=no-early-stop
mkdir ${RESULTS_DIRECTORY}/classic-hnsw
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efSFULL <<< "$dataset_param"
    
    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir ${RESULTS_DIRECTORY}/classic-hnsw/${ds}
    mkdir ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/noisy/

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

        mkdir ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}
        mkdir ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/times/
        for tuple in "${tuples[@]}"; do
          read target_recall efS <<< "$tuple"

          if false; then
            for time in $(seq 1 $experiment_times)
            do
                  ./../build/hnsw-test/hnsw_test \
                      --dataset ${ds} \
                      --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                      --query-num ${sample} --k $k \
                      --output ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt \
                      --mode ${mode} \
                      --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                      --dataset-dir-prefix ${DATASET_DIRECTORY} \
                      --target-recall ${target_recall} \
                      --query-type testing
            done

            python result_merger.py \
                  ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall} \
                  ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}.txt \
                  $experiment_times
          fi

          if true; then
            mkdir ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/noisy/
            mkdir ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/noisy/times
            for noise in 14 16 18 20 22 24 26 28 30
            do
              echo "noise: ${noise}"
              
              for time in $(seq 1 $experiment_times)
              do
                ./../build/hnsw-test/hnsw_test \
                              --dataset ${ds} \
                              --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                              --query-num ${sample} --k $k \
                              --output ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/noisy/times/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt \
                              --mode ${mode} \
                              --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                              --dataset-dir-prefix ${DATASET_DIRECTORY} \
                              --target-recall ${target_recall} \
                              --query-type noisy-testing \
                              --gnoise-perc ${noise}
              done

              python result_merger.py \
                ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/noisy/times/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall} \
                ${RESULTS_DIRECTORY}/classic-hnsw/${ds}/k${k}/noisy/noise${noise}_M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}.txt \
                $experiment_times
            done
          fi

        done
        
    done

    echo ""
    echo ""
done
