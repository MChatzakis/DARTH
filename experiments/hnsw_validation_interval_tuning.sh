cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
make -C build -j faiss
make -C build -j hnsw_test

echo ""
echo ""
echo "============================="
echo ""
echo ""

sample=1000
mode=early-stop-testing
mkdir results/${mode}

n_estim=100
train_queries=10000
li=2
experiment_times=1

INDEX_DIRECTORY=/home/mchatzakis/hnsw-index
DATASET_DIRECTORY=/data/mchatzakis/datasets/processed/

dataset_params=(
    #"SIFT100M 32 500 500"
    #"GIST1M 32 500 1000"
    #"GLOVE100 16 500 500"
    #"DEEP100M 32 500 750"
    "T2I100M 80 1000 2500"
)

k_values=(50)
target_recalls=(0.95)

SIFT100M_tuples_k50=(
  "0.90 2979 2979" 
  "0.90 2979 1489" 
  "0.90 2979 745" 
  "0.90 2979 372" 
  "0.90 2979 298" 
  "0.90 2979 186" 
  "0.90 2979 149" 
  "0.90 1489 2979" 
  "0.90 1489 1489" 
  "0.90 1489 745" 
  "0.90 1489 372" 
  "0.90 1489 298" 
  "0.90 1489 186" 
  "0.90 1489 149" 
  "0.90 745 2979" 
  "0.90 745 1489" 
  "0.90 745 745" 
  "0.90 745 372" 
  "0.90 745 298" 
  "0.90 745 186" 
  "0.90 745 149" 
  "0.90 372 2979" 
  "0.90 372 1489" 
  "0.90 372 745" 
  "0.90 372 372" 
  "0.90 372 298" 
  "0.90 372 186" 
  "0.90 372 149" 
  "0.99 6566 6566" 
  "0.99 6566 3283" 
  "0.99 6566 1641" 
  "0.99 6566 821" 
  "0.99 6566 657" 
  "0.99 6566 410" 
  "0.99 6566 328" 
  "0.99 3283 6566" 
  "0.99 3283 3283" 
  "0.99 3283 1641" 
  "0.99 3283 821" 
  "0.99 3283 657" 
  "0.99 3283 410" 
  "0.99 3283 328" 
  "0.99 1641 6566" 
  "0.99 1641 3283" 
  "0.99 1641 1641" 
  "0.99 1641 821" 
  "0.99 1641 657" 
  "0.99 1641 410" 
  "0.99 1641 328" 
  "0.99 821 6566" 
  "0.99 821 3283" 
  "0.99 821 1641" 
  "0.99 821 821" 
  "0.99 821 657" 
  "0.99 821 410" 
  "0.99 821 328" 
) 

GLOVE100_tuples_k50=(
  "0.90 543 543" 
  "0.90 543 272" 
  "0.90 543 136" 
  "0.90 543 68" 
  "0.90 543 54" 
  "0.90 543 34" 
  "0.90 543 27" 
  "0.90 272 543" 
  "0.90 272 272" 
  "0.90 272 136" 
  "0.90 272 68" 
  "0.90 272 54" 
  "0.90 272 34" 
  "0.90 272 27" 
  "0.90 136 543" 
  "0.90 136 272" 
  "0.90 136 136" 
  "0.90 136 68" 
  "0.90 136 54" 
  "0.90 136 34" 
  "0.90 136 27" 
  "0.90 68 543" 
  "0.90 68 272" 
  "0.90 68 136" 
  "0.90 68 68" 
  "0.90 68 54" 
  "0.90 68 34" 
  "0.90 68 27" 
  "0.99 978 978" 
  "0.99 978 489" 
  "0.99 978 245" 
  "0.99 978 122" 
  "0.99 978 98" 
  "0.99 978 61" 
  "0.99 978 49" 
  "0.99 489 978" 
  "0.99 489 489" 
  "0.99 489 245" 
  "0.99 489 122" 
  "0.99 489 98" 
  "0.99 489 61" 
  "0.99 489 49" 
  "0.99 245 978" 
  "0.99 245 489" 
  "0.99 245 245" 
  "0.99 245 122" 
  "0.99 245 98" 
  "0.99 245 61" 
  "0.99 245 49" 
  "0.99 122 978" 
  "0.99 122 489" 
  "0.99 122 245" 
  "0.99 122 122" 
  "0.99 122 98" 
  "0.99 122 61" 
  "0.99 122 49" 
) 

GIST1M_tuples_k50=(
  "0.90 4289 4289" 
  "0.90 4289 2145" 
  "0.90 4289 1072" 
  "0.90 4289 536" 
  "0.90 4289 429" 
  "0.90 4289 268" 
  "0.90 4289 214" 
  "0.90 2145 4289" 
  "0.90 2145 2145" 
  "0.90 2145 1072" 
  "0.90 2145 536" 
  "0.90 2145 429" 
  "0.90 2145 268" 
  "0.90 2145 214" 
  "0.90 1072 4289" 
  "0.90 1072 2145" 
  "0.90 1072 1072" 
  "0.90 1072 536" 
  "0.90 1072 429" 
  "0.90 1072 268" 
  "0.90 1072 214" 
  "0.90 536 4289" 
  "0.90 536 2145" 
  "0.90 536 1072" 
  "0.90 536 536" 
  "0.90 536 429" 
  "0.90 536 268" 
  "0.90 536 214" 
  "0.99 9475 9475" 
  "0.99 9475 4737" 
  "0.99 9475 2369" 
  "0.99 9475 1184" 
  "0.99 9475 947" 
  "0.99 9475 592" 
  "0.99 9475 474" 
  "0.99 4737 9475" 
  "0.99 4737 4737" 
  "0.99 4737 2369" 
  "0.99 4737 1184" 
  "0.99 4737 947" 
  "0.99 4737 592" 
  "0.99 4737 474" 
  "0.99 2369 9475" 
  "0.99 2369 4737" 
  "0.99 2369 2369" 
  "0.99 2369 1184" 
  "0.99 2369 947" 
  "0.99 2369 592" 
  "0.99 2369 474" 
  "0.99 1184 9475" 
  "0.99 1184 4737" 
  "0.99 1184 2369" 
  "0.99 1184 1184" 
  "0.99 1184 947" 
  "0.99 1184 592" 
  "0.99 1184 474" 
) 

DEEP100M_tuples_k50=(
  "0.90 3638 3638" 
  "0.90 3638 1819" 
  "0.90 3638 910" 
  "0.90 3638 455" 
  "0.90 3638 364" 
  "0.90 3638 227" 
  "0.90 3638 182" 
  "0.90 1819 3638" 
  "0.90 1819 1819" 
  "0.90 1819 910" 
  "0.90 1819 455" 
  "0.90 1819 364" 
  "0.90 1819 227" 
  "0.90 1819 182" 
  "0.90 910 3638" 
  "0.90 910 1819" 
  "0.90 910 910" 
  "0.90 910 455" 
  "0.90 910 364" 
  "0.90 910 227" 
  "0.90 910 182" 
  "0.90 455 3638" 
  "0.90 455 1819" 
  "0.90 455 910" 
  "0.90 455 455" 
  "0.90 455 364" 
  "0.90 455 227" 
  "0.90 455 182" 
  "0.99 8021 8021" 
  "0.99 8021 4011" 
  "0.99 8021 2005" 
  "0.99 8021 1003" 
  "0.99 8021 802" 
  "0.99 8021 501" 
  "0.99 8021 401" 
  "0.99 4011 8021" 
  "0.99 4011 4011" 
  "0.99 4011 2005" 
  "0.99 4011 1003" 
  "0.99 4011 802" 
  "0.99 4011 501" 
  "0.99 4011 401" 
  "0.99 2005 8021" 
  "0.99 2005 4011" 
  "0.99 2005 2005" 
  "0.99 2005 1003" 
  "0.99 2005 802" 
  "0.99 2005 501" 
  "0.99 2005 401" 
  "0.99 1003 8021" 
  "0.99 1003 4011" 
  "0.99 1003 2005" 
  "0.99 1003 1003" 
  "0.99 1003 802" 
  "0.99 1003 501" 
  "0.99 1003 401" 
) 

T2I100M_tuples_k50=(
  "0.90 4766 2383" 
  "0.90 4766 1192" 
  "0.90 4766 953" 
  "0.90 2383 2383" 
  "0.90 2383 1192" 
  "0.90 2383 953" 
  "0.95 4766 2383" 
  "0.95 4766 1192" 
  "0.95 4766 953" 
  "0.95 2383 2383" 
  "0.95 2383 1192" 
  "0.95 2383 953" 
) 


mkdir results/interval-tuning/heuristics/
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS <<< "$dataset_param"
    
    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir results/interval-tuning/heuristics/${ds}

    for k in ${k_values[@]}
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""
        
        mkdir results/interval-tuning/heuristics/${ds}/k${k}
        mkdir results/interval-tuning/heuristics/${ds}/k${k}/times/

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

        for tuple in "${tuples[@]}"; do
            read target_recall initial_prediction_interval min_prediction_interval <<< "$tuple"

            for time in $(seq 1 $experiment_times)
            do
                ./build/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${sample} --k $k \
                        --output results/interval-tuning/heuristics/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}.txt \
                        --mode ${mode} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --target-recall ${target_recall} \
                        --initial-prediction-interval ${initial_prediction_interval} \
                        --min-prediction-interval ${min_prediction_interval} \
                        --query-type validation \
                        --predictor-model-path predictor_models/lightgbm/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt
            done

            python result_merger.py \
                results/interval-tuning/heuristics/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
                results/interval-tuning/heuristics/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
                $experiment_times
        
        done

    done

done

exit

minimum_intervals=(10) #(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200)
initial_intervals=(10) #(400 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000)
mkdir results/interval-tuning/
mkdir results/interval-tuning/search/
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS <<< "$dataset_param"
    
    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir results/interval-tuning/search/${ds}
    for k in ${k_values[@]}
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""
        
        mkdir results/interval-tuning/search/${ds}/k${k}
        mkdir results/interval-tuning/search/${ds}/k${k}/times/
        for target_recall in "${target_recalls[@]}"; do
            for initial_prediction_interval in "${initial_intervals[@]}"; do
                for min_prediction_interval in "${minimum_intervals[@]}"; do
                    
                    for time in $(seq 1 $experiment_times)
                    do
                        ./build/hnsw-test/hnsw_test \
                            --dataset ${ds} \
                            --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                            --query-num ${sample} --k $k \
                            --output results/interval-tuning/search/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}.txt \
                            --mode ${mode} \
                            --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                            --dataset-dir-prefix ${DATASET_DIRECTORY} \
                            --target-recall ${target_recall} \
                            --initial-prediction-interval ${initial_prediction_interval} \
                            --min-prediction-interval ${min_prediction_interval} \
                            --query-type validation \
                            --predictor-model-path predictor_models/lightgbm/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt
                    done

                    #python result_merger.py \
                    #    results/interval-tuning/search/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
                    #    results/interval-tuning/search/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
                    #    $experiment_times
                    
                done
            done
        done
    done

    echo ""
    echo ""
done

exit 

intervals=(500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000 4250 4500 4750 5000 5250 5500 5750 6000 6250 6500 6750 7000)
mkdir results/interval-tuning/static/
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS <<< "$dataset_param"
    
    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir results/interval-tuning/static/${ds}

    for k in ${k_values[@]}
    do
        echo ""
        echo "--------------- k=${k} ---------------"
        echo ""
        
        mkdir results/interval-tuning/static/${ds}/k${k}
        mkdir results/interval-tuning/static/${ds}/k${k}/times/
        
        for target_recall in "${target_recalls[@]}"; do
            
            for interval in "${intervals[@]}"; do
              initial_prediction_interval=$interval
              min_prediction_interval=$interval

              for time in $(seq 1 $experiment_times)
              do
                ./build/hnsw-test/hnsw_test \
                    --dataset ${ds} \
                    --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                    --query-num ${sample} --k $k \
                    --output results/interval-tuning/static/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}.txt \
                    --mode ${mode} \
                    --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                    --dataset-dir-prefix ${DATASET_DIRECTORY} \
                    --target-recall ${target_recall} \
                    --initial-prediction-interval ${initial_prediction_interval} \
                    --min-prediction-interval ${min_prediction_interval} \
                    --query-type validation \
                    --predictor-model-path predictor_models/lightgbm/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt
              done

                python result_merger.py \
                  results/interval-tuning/static/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
                  results/interval-tuning/static/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
                  $experiment_times  
            done
        done
        
    done

    echo ""
    echo ""
done





exit


exit
