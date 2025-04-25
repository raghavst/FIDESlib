#include "helper.hpp"
#include "data.hpp"
#include "encoding.hpp"
#include "fhe.hpp"

#include <iostream>
#include <tuple>

std::string dataset_path(const dataset_t dataset, const exec_t exec) {

    std::string filename;
    switch (dataset) {
        case dataset_t::RANDOM:
            filename = "../data/random_data";
            break;
        case dataset_t::MNIST:
            filename = "../data/mnist_data";
            break;
        default:
            exit(EXIT_FAILURE);
    }

    switch (exec) {
        case exec_t::TRAIN:
            filename += "_train.csv";
            break;
        case exec_t::VALIDATION:
            filename += "_validation.csv";
            break;
        default:
            exit(EXIT_FAILURE);
    }

    return filename;
}

size_t dataset_label_index(const dataset_t dataset) {
    switch (dataset) {
        case dataset_t::RANDOM:
            return 25;
        case dataset_t::MNIST:
            return 196;
        default:
            exit(EXIT_FAILURE);
    }
}

std::string dataset_name (const dataset_t dataset) {
    switch (dataset) {
        case dataset_t::RANDOM:
            return "random";
        case dataset_t::MNIST:
            return "mnist";
        default:
            exit(EXIT_FAILURE);
    }
}

dataset_t dataset_from (const std::string& dataset) {
    if (dataset == "random") return dataset_t::RANDOM;
    if (dataset == "mnist") return dataset_t::MNIST;
    exit(EXIT_FAILURE);
}

backend_t backend_from(const std::string& backend) {
    if (backend == "naive") return backend_t::NAIVE;
    if (backend == "cpu") return backend_t::CPU;
    if (backend == "gpu") return backend_t::GPU;
    exit(EXIT_FAILURE);
}

std::string backend_name(backend_t backend) {
    switch (backend) {
        case backend_t::NAIVE:
            return "naive";
        case backend_t::CPU:
            return "cpu";
        case backend_t::GPU:
            return "gpu";
        default:
            exit(EXIT_FAILURE);
    }
    exit(EXIT_FAILURE);
}

bool is_acceler_from(const std::string& acc) {
    if (acc == "accelerated") return true;
    if (acc == "normal") return false;
    exit(EXIT_FAILURE);
}

std::string weights_path(const backend_t backend, const dataset_t dataset) {
    std::string filename = "../weights/" + backend_name(backend) + "_" + dataset_name(dataset) + ".csv";
    return filename;
}

std::string times_path(const dataset_t dataset, const exec_t exec, const backend_t backend) {
    std::string boot = bootstrap_every_two ? "2" : "1";
    std::string filename = "../times/" + backend_name(backend) + "_"+ dataset_name(dataset) + "_boot" + boot;
    switch (exec) {
        case TRAIN:
            filename += "_train.csv";
            break;
        case VALIDATION:
            filename += "_validation.csv";
            break;
        default:
            exit(EXIT_FAILURE);
    }
    return filename;
}

size_t prepare_data_csv(const dataset_t dataset, const exec_t exec,
                        std::vector<std::vector<double> > &data,
                        std::vector<double> &results)
{
    data.clear();
    results.clear();

    /// --------- Load the data. ---------

    std::vector<std::vector<std::string> > raw_data;

    const auto filename = dataset_path(dataset, exec);

    bool res = load_csv(filename, raw_data);

    if (!res) {
        std::cerr << "Failed to load data from " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    /// --------- Parse the data. ---------

    const std::tuple<size_t, size_t> data_info = parse_data(raw_data, data, results, dataset_label_index(dataset));

    const size_t num_features = std::get<0>(data_info);
    const size_t num_samples = std::get<1>(data_info);

    std::cout << "Parsed " << num_samples << " data samples of " << num_features << " features each" << std::endl;

    res = check_data(data, results);
    if (!res) {
        std::cerr << "Data check failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    /// --------- Partition the data. ---------

    return num_features;
}

std::tuple<size_t, size_t, size_t> pack_data_fhe(const std::vector<std::vector<double> > &data,
                                                    const std::vector<double> &results,
                                                    const std::vector<double> &weights,
                                                    std::vector<std::vector<double> > &data_fhe,
                                                    std::vector<std::vector<double> > &results_fhe,
                                                    std::vector<double> &weights_fhe,
                                                    const size_t num_slots) {
    data_fhe.clear();
    results_fhe.clear();

    const auto adapted_training_data_info = pack_data(data, data_fhe, num_slots);

    const size_t ciphertext_matrix_columns = std::get<0>(adapted_training_data_info);
    const size_t ciphertext_matrix_rows = std::get<1>(adapted_training_data_info);
    const size_t last_ciphertext_matrix_rows = std::get<2>(adapted_training_data_info);

    const auto adapted_training_result_info = pack_results(results, results_fhe, ciphertext_matrix_columns);

    if (std::get<0>(adapted_training_data_info) != std::get<0>(adapted_training_result_info)) {
        std::cerr << "Error on number of features per data sample" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (std::get<1>(adapted_training_data_info) != std::get<1>(adapted_training_result_info)) {
        std::cerr << "Error on number of data samples per ciphertext" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (std::get<2>(adapted_training_data_info) != std::get<2>(adapted_training_result_info)) {
        std::cerr << "Error on last ciphertext data samples count" << std::endl;
        exit(EXIT_FAILURE);
    }


    std::cout << "Number of packed data samples: " << data.size() << std::endl;
    std::cout << "Augmented data features to " << ciphertext_matrix_columns<< std::endl;
    std::cout << "Using " << num_slots << " slots yields " << data_fhe.size() << " ciphertexts with "<< ciphertext_matrix_rows << " data samples each" << std::endl;
    std::cout << "Last ciphertext contains only " << last_ciphertext_matrix_rows << " data samples" << std::endl;

    if (ciphertext_matrix_columns*ciphertext_matrix_rows != num_slots) {
        std::cerr << "Wrong number of elements per ciphertext" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!pack_weights(weights, weights_fhe, ciphertext_matrix_columns, ciphertext_matrix_rows)) {
        std::cerr << "Failed to pack weights" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (weights_fhe.size() != ciphertext_matrix_rows*ciphertext_matrix_columns) {
        std::cerr << "Wrong number elements on FHE weights" << std::endl;
        exit(EXIT_FAILURE);
    }

    return std::make_tuple(ciphertext_matrix_columns, ciphertext_matrix_rows, last_ciphertext_matrix_rows);
}