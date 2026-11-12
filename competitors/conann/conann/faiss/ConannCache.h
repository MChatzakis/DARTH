#ifndef CONANN_CACHE_H
#define CONANN_CACHE_H

#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <type_traits>

// for testing
#include <cassert>
#include <cmath>

namespace faiss {

// supports disk caching of vectors or nested vectors of floats or faiss::idx_t
namespace conann_cache {

template <typename T>
void _write_nested_vector(std::ofstream& file, std::vector<T> data) {
    // Write container size
    size_t size = data.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    // reached innermost vector
    if constexpr (std::is_same_v<T,float> || std::is_same_v<T, int64_t>) {
        // Write data
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    } else {
        for (auto element : data) {
            _write_nested_vector<typename T::value_type>(file, element);
        }
    }
}

template <typename T>
void _write_to_file(const std::string &filename, std::vector<T>data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    _write_nested_vector<T>(file, data);
    file.close();
}

template <typename T>
void write_to_cache(std::string key, const std::vector<T> &data){
    namespace fs = std::filesystem;
    fs::create_directory("./conann-cache");

    // create file and write "plain old data"
    std::ostringstream filename;
    filename << "./conann-cache/" << key;

    _write_to_file<T>(filename.str(), data);
};

template <typename T>
void _read_nested_vector(std::ifstream& file, std::vector<T>& data) {
    // Read container size
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    data.resize(size);
    // reached innermost vector
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int64_t>) {
        // Read data
        file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    } else {
        for (size_t i = 0; i < size; i++) {
            _read_nested_vector<typename T::value_type>(file, data[i]);
        }
    }
}

template <typename T>
std::vector<T> _read_from_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::vector<T> data{};
    _read_nested_vector<T>(file, data);
    file.close();
    return data;
}

template <typename T>
T read_from_cache(std::string key) {
    // Create cache directory if it doesn't exist
    namespace fs = std::filesystem;
    fs::create_directory("./conann-cache");

    // find file and read "plain old data"
    std::ostringstream filename;
    filename << "./conann-cache/" << key;
    return _read_from_file<typename T::value_type>(filename.str());
};

bool check_cached_file(std::string key) {
    std::ostringstream filename;
    filename << "./conann-cache/" << key;
    std::ifstream file(filename.str());
    return file.good();
}

void test_cache_io() {
    // Test 1: Simple vector<float>
    {
        std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        faiss::conann_cache::write_to_cache("test1", test_data);
        auto read_data = faiss::conann_cache::read_from_cache<std::vector<float>>("test1");
        
        // for (auto value: read_data) {
        //     std::cout << value << std::endl;
        // }
        assert(test_data.size() == read_data.size());
        for (size_t i = 0; i < test_data.size(); i++) {
            assert(std::abs(test_data[i] - read_data[i]) < 1e-6);
        }
    }
    
    // Test 2: Nested vector<vector<float>>
    {
        std::vector<std::vector<float>> test_data = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}
        };
        faiss::conann_cache::write_to_cache("test2", test_data);
        auto read_data = faiss::conann_cache::read_from_cache<std::vector<std::vector<float>>>("test2");
        
        assert(test_data.size() == read_data.size());
        for (size_t i = 0; i < test_data.size(); i++) {
            assert(test_data[i].size() == read_data[i].size());
            for (size_t j = 0; j < test_data[i].size(); j++) {
                assert(std::abs(test_data[i][j] - read_data[i][j]) < 1e-6);
            }
        }
    }

    // Test 3: Nested vector<vector<vector<int64_t>>>
    {
        std::vector<std::vector<std::vector<int64_t>>> test_data = {
            {{1, 2}, {3, 4}},
            {{5, 6}, {7, 8}},
            {{9, 10}, {11, 12}}
        };
        faiss::conann_cache::write_to_cache("test3", test_data);
        auto read_data = faiss::conann_cache::read_from_cache<std::vector<std::vector<std::vector<int64_t>>>>("test3");

        assert(test_data.size() == read_data.size());
        for (size_t i = 0; i < test_data.size(); i++) {
            assert(test_data[i].size() == read_data[i].size());
            for (size_t j = 0; j < test_data[i].size(); j++) {
                assert(test_data[i][j].size() == read_data[i][j].size());
                for (size_t k = 0; k < test_data[i][j].size(); k++) {
                    assert(test_data[i][j][k] == read_data[i][j][k]);
                }
            }
        }
    }
    
    // Test 4: Empty vectors
    {
        std::vector<std::vector<float>> empty_data;
        faiss::conann_cache::write_to_cache("test3", empty_data);
        auto read_data = faiss::conann_cache::read_from_cache<std::vector<std::vector<float>>>("test3");
        assert(read_data.empty());
    }

    // Test 5: Cache checks
    {
        auto shouldBeCached = faiss::conann_cache::check_cached_file("test3");
        assert(shouldBeCached);

        auto shouldNotBeCached = faiss::conann_cache::check_cached_file("unknown");
        assert(!shouldNotBeCached);
    }
    
    // Cleanup
    std::filesystem::remove_all("./conann-cache");
    
    std::cout << "All cache I/O tests passed!" << std::endl;
}


} 

}

#endif // CONANN_CACHE