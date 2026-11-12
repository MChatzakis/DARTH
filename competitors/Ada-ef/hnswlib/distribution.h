#pragma once

#include <Eigen/Core>
#include <memory>
#include <thread>
#include "boost/math/distributions/normal.hpp"
#include <iostream>
#include <fstream>

namespace hnswdis
{
    // Typedefs for row-major matrices
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    using RowVectorXf = Eigen::RowVectorXf;
    using RowVectorXi = Eigen::RowVectorXi;

    // Function to compute the cross-covariance matrix
    MatrixXf compute_cross_covariance_matrix(const MatrixXf &data1, const MatrixXf &data2,
                                             const RowVectorXf &means1, const RowVectorXf &means2)
    {
        MatrixXf centered_data1 = data1.rowwise() - means1;
        MatrixXf centered_data2 = data2.rowwise() - means2;
        size_t n_samples = data1.rows();
        MatrixXf cov_matrix(data1.cols(), data2.cols());
        cov_matrix.noalias() = (centered_data1.transpose() * centered_data2) / (n_samples - 1);
        return cov_matrix;
    }

    // Function to compute the covariance matrix and mean vector
    MatrixXf compute_covariance_matrix(const MatrixXf &data, const RowVectorXf &means)
    {
        auto start = std::chrono::high_resolution_clock::now();
        MatrixXf centered_data = data.rowwise() - means;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Centered matrix computation time: " << duration.count() << " ms" << std::endl;

        return (centered_data.transpose() * centered_data) / (data.rows() - 1);
    }

    MatrixXf compute_covariance_matrix2(const MatrixXf &data, const RowVectorXf &means)
    {
        const int n = data.rows();
        const int d = data.cols();

        MatrixXf cov = MatrixXf::Zero(d, d);

        int num_threads = std::max(1u, std::thread::hardware_concurrency() / 4);
        omp_set_num_threads(num_threads);

        std::cout << "Number of threads: " << num_threads << std::endl;
        std::vector<MatrixXf> local_covs(num_threads, MatrixXf::Zero(d, d));

        const int block_size = 64;

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &local = local_covs[tid];

#pragma omp for schedule(static)
            for (int k = 0; k < n; k += block_size)
            {
                int current_block = std::min(block_size, n - k);
                MatrixXf block = data.middleRows(k, current_block);
                block.rowwise() -= means; // in-place rowwise center

                local.noalias() += block.transpose() * block;
            }
        }

        // Aggregate all thread-local results
        for (const auto &mat : local_covs)
            cov += mat;

        cov /= float(n - 1);
        return cov;
    }

    RowVectorXf compute_mean_parallel(const MatrixXf &data)
    {
        const int n = data.rows();
        const int d = data.cols();

        RowVectorXf mean = RowVectorXf::Zero(d);

        int num_threads = std::max(1u, std::thread::hardware_concurrency() / 4);
        omp_set_num_threads(num_threads);
        std::vector<RowVectorXf> local_sums(num_threads, RowVectorXf::Zero(d));

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &local = local_sums[tid];

#pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
            {
                local += data.row(i);
            }
        }

        // Aggregate
        for (const auto &vec : local_sums)
        {
            mean += vec;
        }

        mean /= float(n);
        return mean;
    }

    // Abstract Estimator class
    class Estimator
    {
    protected:
        MatrixXf covariance_matrix;
        RowVectorXf means;
        RowVectorXf variances;

    public:
        Estimator() = default;

        Estimator(const MatrixXf &data_vecs)
        {
            auto start = std::chrono::high_resolution_clock::now();
            // this->means = data_vecs.colwise().mean();
            this->means = compute_mean_parallel(data_vecs);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Mean computation time: " << duration.count() << " ms" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            // this->covariance_matrix = compute_covariance_matrix(data_vecs, means);
            this->covariance_matrix = compute_covariance_matrix2(data_vecs, means);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Covariance matrix computation time: " << duration.count() << " ms" << std::endl;

            this->variances = this->covariance_matrix.diagonal();
        }

        virtual ~Estimator() = default;

        virtual std::tuple<float, float> get_ideal_distribution(const RowVectorXf &q) const = 0;
        virtual std::tuple<float, float> get_practical_distribution(const RowVectorXf &q) const = 0;

        virtual std::string get_name() const = 0;

        virtual void serialize(std::ostream &out) const
        {
            int rows = covariance_matrix.rows(), cols = covariance_matrix.cols();
            out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
            out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
            out.write(reinterpret_cast<const char *>(covariance_matrix.data()), sizeof(float) * rows * cols);

            int mean_size = means.size();
            out.write(reinterpret_cast<const char *>(&mean_size), sizeof(int));
            out.write(reinterpret_cast<const char *>(means.data()), sizeof(float) * mean_size);

            int var_size = variances.size();
            out.write(reinterpret_cast<const char *>(&var_size), sizeof(int));
            out.write(reinterpret_cast<const char *>(variances.data()), sizeof(float) * var_size);
        }

        virtual void deserialize(std::istream &in)
        {
            int rows, cols;
            in.read(reinterpret_cast<char *>(&rows), sizeof(int));
            in.read(reinterpret_cast<char *>(&cols), sizeof(int));
            covariance_matrix.resize(rows, cols);
            in.read(reinterpret_cast<char *>(covariance_matrix.data()), sizeof(float) * rows * cols);

            int size;
            in.read(reinterpret_cast<char *>(&size), sizeof(int));
            means.resize(size);
            in.read(reinterpret_cast<char *>(means.data()), sizeof(float) * size);

            in.read(reinterpret_cast<char *>(&size), sizeof(int));
            variances.resize(size);
            in.read(reinterpret_cast<char *>(variances.data()), sizeof(float) * size);
        }

        // n: old data size
        // m: new data size
        void add_points(const MatrixXf &new_data, int old_count)
        {
            int n = old_count;
            int m = new_data.rows();
            if (m == 0)
                return;

            RowVectorXf means_new = compute_mean_parallel(new_data);
            MatrixXf cov_new = compute_covariance_matrix2(new_data, means_new);

            int d = new_data.cols();

            RowVectorXf delta = this->means - means_new;
            MatrixXf delta_outer = delta.transpose() * delta;

            float total_count = float(n + m);
            float scale_old = float(n - 1);
            float scale_new = float(m - 1);
            float correction = float(n * m) / total_count;

            this->means = (n * this->means + m * means_new) / total_count;

            // MatrixXf updated_cov = scale_old * this->covariance_matrix +
            //                        scale_new * cov_new +
            //                        correction * delta_outer;
            // updated_cov /= (total_count - 1.0f);
            this->covariance_matrix *= scale_old; // in-place scale
            this->covariance_matrix.noalias() += scale_new * cov_new;
            this->covariance_matrix.noalias() += correction * delta_outer;
            this->covariance_matrix /= (total_count - 1.0f); // in-place final scale

            this->variances = this->covariance_matrix.diagonal();
        }

        // In class Estimator
        void delete_points(const MatrixXf &del_data, int current_total_count)
        {
            const int N = current_total_count; // before deletion
            const int m = del_data.rows();
            if (m == 0)
                return;

            const int n = N - m; // remaining
            if (n <= 0)
            {
                // Degenerate: removing all (or more) points; reset state
                covariance_matrix.setZero(covariance_matrix.rows(), covariance_matrix.cols());
                means.setZero(means.size());
                variances.setZero(variances.size());
                return;
            }
            if (n == 1)
            {
                // With 1 sample, unbiased covariance is undefined; we choose zeros
                RowVectorXf means_del = compute_mean_parallel(del_data);
                // Update mean to the lone remaining point’s mean:
                // mu_A = (N*mu - m*muB)/1
                means = (float(N) * means - float(m) * means_del);
                covariance_matrix.setZero(covariance_matrix.rows(), covariance_matrix.cols());
                variances = covariance_matrix.diagonal();
                return;
            }

            // Stats of batch to delete
            RowVectorXf means_del = compute_mean_parallel(del_data);
            MatrixXf cov_del = compute_covariance_matrix2(del_data, means_del); // unbiased (m-1)

            // Pre-compute deltas
            RowVectorXf delta = means - means_del; // (mu - mu_B)
            MatrixXf delta_outer = delta.transpose() * delta;

            // Update mean first (uses stable formula)
            // mu_A = (N*mu - m*mu_B) / n
            RowVectorXf new_mean = (float(N) * means - float(m) * means_del) / float(n);

            // Update covariance using unbiased pooled-variance algebra, in-place for efficiency:
            // (n-1) S_A = (N-1) S - (m-1) S_B - (m*N/n) * (mu - mu_B)(mu - mu_B)^T
            covariance_matrix *= (float(N) - 1.0f);
            covariance_matrix.noalias() -= (float(m) - 1.0f) * cov_del;
            covariance_matrix.noalias() -= (float(m) * float(N) / float(n)) * delta_outer;
            covariance_matrix /= (float(n) - 1.0f);

            // Commit mean + diag
            means = std::move(new_mean);
            variances = covariance_matrix.diagonal();
        }
    };

    // InnerProductEstimator
    class InnerProductEstimator : public Estimator
    {
    public:
        using Estimator::Estimator;
        InnerProductEstimator() = default;

        std::tuple<float, float> get_ideal_distribution(const RowVectorXf &q) const override
        {
            return {q.dot(means), q.array().square().matrix().dot(variances)};
        }

        std::tuple<float, float> get_practical_distribution(const RowVectorXf &q) const override
        {
            return {q.dot(means), (q * covariance_matrix * q.transpose())(0, 0)};
        }
        std::string get_name() const override
        {
            return "InnerProductEstimator";
        }
    };

    // CosineSimilarityEstimator
    class CosineSimilarityEstimator : public Estimator
    {
    public:
        // assuming data_vecs is already normalized
        using Estimator::Estimator;
        CosineSimilarityEstimator() = default;
        // CosineSimilarityEstimator(const MatrixXf &data_vecs)
        //     : Estimator(data_vecs) {}

        std::tuple<float, float> get_ideal_distribution(const RowVectorXf &q) const override
        {
            // RowVectorXf q_normalized = q.normalized(); // assuming q is alreayd normalized
            return {q.dot(means), q.array().square().matrix().dot(variances)};
        }

        std::tuple<float, float> get_practical_distribution(const RowVectorXf &q) const override
        {
            // RowVectorXf q_normalized = q.normalized(); // assuming q is alreayd normalized
            return {q.dot(means), (q * covariance_matrix * q.transpose())(0, 0)};
        }

        std::string get_name() const override
        {
            return "CosineSimilarityEstimator";
        }
    };

    // CosineDistanceEstimator
    class CosineDistanceEstimator : public CosineSimilarityEstimator
    {
    public:
        using CosineSimilarityEstimator::CosineSimilarityEstimator;
        CosineDistanceEstimator() = default;

        std::tuple<float, float> get_ideal_distribution(const RowVectorXf &q) const override
        {
            auto [ideal_mean, ideal_var] = CosineSimilarityEstimator::get_ideal_distribution(q);
            return {1 - ideal_mean, ideal_var};
        }

        std::tuple<float, float> get_practical_distribution(const RowVectorXf &q) const override
        {
            auto [practical_mean, practical_var] = CosineSimilarityEstimator::get_practical_distribution(q);
            return {1 - practical_mean, practical_var};
        }
        std::string get_name() const override
        {
            return "CosineDistanceEstimator";
        }
    };

    void save_estimator_to_file(const Estimator &estimator, const std::string &filename)
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out)
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        std::string type = estimator.get_name();
        size_t type_len = type.size();
        out.write(reinterpret_cast<const char *>(&type_len), sizeof(type_len));
        out.write(type.c_str(), type_len);

        estimator.serialize(out);
        out.close();
    }

    std::shared_ptr<Estimator> load_estimator_from_file(const std::string &filename)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        size_t type_len;
        in.read(reinterpret_cast<char *>(&type_len), sizeof(type_len));
        std::string type(type_len, ' ');
        in.read(&type[0], type_len);

        std::shared_ptr<Estimator> estimator;
        if (type == "InnerProductEstimator")
        {
            estimator = std::make_shared<InnerProductEstimator>();
        }
        else if (type == "CosineSimilarityEstimator")
        {
            estimator = std::make_shared<CosineSimilarityEstimator>();
        }
        else if (type == "CosineDistanceEstimator")
        {
            estimator = std::make_shared<CosineDistanceEstimator>();
        }
        else
        {
            throw std::runtime_error("Unknown estimator type: " + type);
        }

        estimator->deserialize(in);
        in.close();
        return estimator;
    }

    // class ScoreCalculator
    // {

    // private:
    //     int num_bins;
    //     std::vector<float> threshold_z;
    //     std::vector<float> bin_weight;

    //     std::vector<float> get_threshold_z(const int num_bins)
    //     {
    //         std::vector<float> bin_threshold(num_bins + 1);
    //         boost::math::normal_distribution<> normal_dist(0.0, 1.0);
    //         bin_threshold[0] = -std::numeric_limits<float>::infinity();
    //         for (int i = 1; i < num_bins; ++i)
    //         {
    //             float p = static_cast<float>(i) / num_bins;
    //             bin_threshold[i] = boost::math::quantile(normal_dist, p);
    //         }
    //         bin_threshold[num_bins] = std::numeric_limits<float>::infinity();
    //         return bin_threshold;
    //     }

    //     std::vector<float> get_bin_weight(const int num_bins)
    //     {
    //         // Parameters for exponential decay
    //         float P0 = 100.0;         // Initial value
    //         float lambda_decay = 1.0; // Decay constant
    //         std::vector<float> weights(num_bins);
    //         for (int i = 0; i < num_bins; ++i)
    //         {
    //             weights[i] = P0 * std::exp(-lambda_decay * i);
    //         }
    //         std::reverse(weights.begin(), weights.end());
    //         return weights;
    //     }

    //     std::vector<float> estimate_threshold(
    //         const float mean, const float var) const
    //     {
    //         float std = std::sqrt(var);

    //         // Calculate bin intervals
    //         std::vector<float> thresholds(threshold_z.size());

    //         for (size_t i = 0; i < threshold_z.size(); ++i)
    //         {
    //             thresholds[i] = mean + threshold_z[i] * std;
    //         }

    //         return thresholds;
    //     }

    // public:
    //     ScoreCalculator(int num_bins)
    //     {
    //         this->num_bins = num_bins;
    //         threshold_z = get_threshold_z(num_bins);
    //         bin_weight = get_bin_weight(num_bins);
    //     }

    //     float calculate_score(
    //         std::vector<float> &dist_list,
    //         const float mean, const float var) const
    //     {
    //         std::vector<float> bin_thresholds = estimate_threshold(mean, var);

    //         std::vector<int> cnt(num_bins, 0);

    //         for (const float dist : dist_list)
    //         {
    //             auto it = std::upper_bound(bin_thresholds.begin(), bin_thresholds.end(), dist);
    //             int bin_index = std::distance(bin_thresholds.begin(), it) - 1;
    //             cnt[bin_index]++;
    //         }

    //         float score = 0.0;
    //         int total = dist_list.size();
    //         for (int i = 0; i < num_bins; ++i)
    //         {
    //             float weight = static_cast<float>(cnt[i]) * 1.0 / total;
    //             score += weight * bin_weight[i];
    //         }

    //         return score;
    //     }
    // };

    class ApproximatedScoreCalculator
    {
    private:
        int num_bins;
        float quantile_step;
        bool isTopBased = false;
        std::vector<float> threshold_z;
        std::vector<float> bin_weight;

        std::shared_ptr<Estimator> estimator;

    public:
        enum class WeightDecayType
        {
            None,
            Linear,
            Exponential
        };

    private:
        WeightDecayType decay_type = WeightDecayType::Exponential;

        // Initialize quantile thresholds based on whether higher or lower is better
        void initialize_threshold_z()
        {
            threshold_z.resize(num_bins);
            boost::math::normal_distribution<> normal_dist(0.0, 1.0);

            if (isTopBased)
            {
                for (int i = 0; i < num_bins; ++i)
                {
                    threshold_z[i] = boost::math::quantile(
                        normal_dist,
                        (1 - quantile_step * num_bins) + quantile_step * i); // e.g., 0.995, 0.996, ..., 0.999
                }
            }
            else
            {
                for (int i = 0; i < num_bins; ++i)
                {
                    threshold_z[i] = boost::math::quantile(
                        normal_dist,
                        quantile_step * (i + 1)); // e.g., 0.001, 0.002, ..., 0.005
                }
            }
        }

        // Initialize weights based on decay type; max weight = 100
        void initialize_bin_weight()
        {
            bin_weight.resize(num_bins);
            constexpr float P0 = 100.0;         // Initial value
            constexpr float lambda_decay = 1.0; // Decay constant

            // for (int i = 0; i < num_bins; ++i) // original implementation
            // {
            //     bin_weight[i] = P0 * std::exp(-lambda_decay * i); // descending order, in the case of topbased, it requires to be in ascending order
            // }

            for (int i = 0; i < num_bins; ++i)
            {
                switch (decay_type)
                {
                case WeightDecayType::None:
                    bin_weight[i] = P0;
                    break;

                case WeightDecayType::Linear:
                {
                    // Linearly decaying from 100 → 0
                    float t = static_cast<float>(i) / (num_bins - 1);
                    bin_weight[i] = P0 * (1.0f - t);
                    break;
                }

                case WeightDecayType::Exponential:
                    // Exponential decay, normalized so the first bin = 100
                    bin_weight[i] = P0 * std::exp(-lambda_decay * i);
                    break;
                }
            }

            if (isTopBased)
            {
                std::reverse(std::begin(bin_weight), std::end(bin_weight));
            }
        }

        float calculate_score(
            const void *dist_list, const size_t n,
            const float mean, const float var) const
        {

            float std = std::sqrt(var);

            // Calculate bin intervals
            float bin_thresholds[num_bins];
            for (int i = 0; i < num_bins; ++i)
            {
                bin_thresholds[i] = mean + threshold_z[i] * std;
            }

            int cnt[num_bins] = {};

            if (isTopBased)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    float dist = ((float *)dist_list)[j];
                    for (int i = num_bins - 1; i >= 0; --i)
                    {
                        if (dist > bin_thresholds[i])
                        {
                            ++cnt[i];
                            break;
                        }
                    }
                }
            }
            else
            { // not top based
                for (size_t j = 0; j < n; ++j)
                {
                    float dist = ((float *)dist_list)[j];
                    for (int i = 0; i < num_bins; ++i)
                    {
                        if (dist < bin_thresholds[i])
                        {
                            ++cnt[i];
                            break;
                        }
                    }
                }
            }

            float score = 0.0;
            for (size_t i = 0; i < num_bins; ++i)
            {
                score += (float)cnt[i] / (float)n * bin_weight[i];
            }

            return score;
        }

    public:
        ApproximatedScoreCalculator(
            std::shared_ptr<Estimator> estimator,
            float quantile_step,
            WeightDecayType decay_type = WeightDecayType::Exponential,
            int num_bins = 5
            // number of bins is 5 by default because we use exponential decay function as default which decays quickly and the weights for more than 5 bins are very small;
            // but for decay ablation study, we test other kinds of decay functions and set the number of bins to a larger value, i.e., 1/quantile_step

            ) : estimator(estimator), quantile_step(quantile_step), num_bins(num_bins), decay_type(decay_type)
        {
            // Check if the estimator is top-based
            // top based is the case where the higher the score, the better the similarity. in this case, score is calculated using top bins, e.g., 0.995, ..., 0.999
            // bottom based is the case where the lower the score, the better the similarity. in this case, score is calculated using bottom bins, e.g., 0.001, ..., 0.005
            isTopBased = false;
            if (estimator->get_name() == "InnerProductEstimator" || estimator->get_name() == "CosineSimilarityEstimator")
            {
                isTopBased = true;
            }
            std::cout << "Start creating ApproximatedScoreCalculator" << std::endl;

            initialize_threshold_z();
            initialize_bin_weight();

            std::cout << "ApproximatedScoreCalculator created" << std::endl;
        }

        float compute_score(const void *q, const size_t q_size,
                            const void *dist_list, const size_t dist_list_size) const
        {
            Eigen::Map<const hnswdis::RowVectorXf> q_vec((float *)q, q_size);
            auto [mean, var] = estimator->get_practical_distribution(q_vec);

            return calculate_score(dist_list, dist_list_size, mean, var);
        }
    };

    std::string weight_decay_type_to_string(ApproximatedScoreCalculator::WeightDecayType type)
    {
        switch (type)
        {
        case ApproximatedScoreCalculator::WeightDecayType::None:
            return "None";
        case ApproximatedScoreCalculator::WeightDecayType::Linear:
            return "Linear";
        case ApproximatedScoreCalculator::WeightDecayType::Exponential:
            return "Exponential";
        default:
            return "Unknown";
        }
    }

    std::shared_ptr<hnswdis::Estimator> init_estimator(const std::string &metric, const hnswdis::MatrixXf &data_vectors)
    {
        std::shared_ptr<hnswdis::Estimator> estimator;
        if (metric == "cd")
        {
            estimator = std::make_shared<hnswdis::CosineDistanceEstimator>(data_vectors);
            std::cout << "CosineDistanceEstimator created" << std::endl;
        }
        else if (metric == "ipd")
        {
            estimator = std::make_shared<hnswdis::InnerProductEstimator>(data_vectors);
            std::cout << "InnerProductEstimator created" << std::endl;
        }
        else
        {
            std::cerr << "Invalid metric: " << metric << std::endl;
            throw std::invalid_argument("Unsupported metric: " + metric);
        }
        return estimator;
    };

    // SquaredEuclideanDistanceEstimator
    // class SquaredEuclideanDistanceEstimator : public Estimator
    // {
    // private:
    //     RowVectorXf squared_means;
    //     RowVectorXf squared_variances;

    //     MatrixXf cov_matrix_2_1;
    //     MatrixXf cov_matrix_1_2;
    //     RowVectorXf diag_vec_cov_2_1;

    //     float sum_squared_means;
    //     float sum_squared_variances;
    //     float sum_squared_cov_matrix;

    //     RowVectorXf c_upper_2_1; // 1 x n
    //     RowVectorXf c_upper_1_2; // n x 1

    // public:
    //     SquaredEuclideanDistanceEstimator(const MatrixXf &data_vecs)
    //         : Estimator(data_vecs)
    //     {
    //         MatrixXf squared_data_vecs = data_vecs.array().square();
    //         squared_means = squared_data_vecs.colwise().mean();
    //         MatrixXf squared_cov_matrix = compute_covariance_matrix(squared_data_vecs, squared_means);
    //         squared_variances = squared_cov_matrix.diagonal();

    //         cov_matrix_2_1 = compute_cross_covariance_matrix(squared_data_vecs, data_vecs, squared_means, means);
    //         cov_matrix_1_2 = cov_matrix_2_1.transpose();
    //         diag_vec_cov_2_1 = cov_matrix_2_1.diagonal();

    //         sum_squared_means = squared_means.sum();
    //         sum_squared_variances = squared_variances.sum();
    //         sum_squared_cov_matrix = squared_cov_matrix.sum();

    //         c_upper_2_1 = cov_matrix_2_1.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().colwise().sum();
    //         c_upper_1_2 = cov_matrix_1_2.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().rowwise().sum();
    //     }

    //     std::tuple<float, float> get_ideal_distribution(const RowVectorXf &q) const override
    //     {
    //         RowVectorXf q_squared = q.array().square();
    //         float mu = q_squared.sum() + sum_squared_means - 2.0 * q.dot(means);
    //         float var = sum_squared_variances + 4.0 * q_squared.dot(variances) - 4.0 * diag_vec_cov_2_1.dot(q);

    //         return {mu, var};
    //     }

    //     std::tuple<float, float> get_practical_distribution(const RowVectorXf &q) const override
    //     {
    //         float mu = q.array().square().sum() + sum_squared_means - 2.0 * q.dot(means);

    //         float term1 = sum_squared_cov_matrix;
    //         float term2 = 4 * (q * covariance_matrix * q.transpose())(0, 0);
    //         float term3 = -4.0 * diag_vec_cov_2_1.dot(q);

    //         float term4 = -4.0 * c_upper_2_1.dot(q);
    //         float term5 = -4.0 * c_upper_1_2.dot(q);

    //         return {mu, term1 + term2 + term3 + term4 + term5};
    //     }
    //     std::string get_name() const override
    //     {
    //         return "SquaredEuclideanDistanceEstimator";
    //     }
    // };
}