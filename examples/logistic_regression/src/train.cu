#include "train.cuh"
#include "helper.hpp"
#include "fhe.hpp"
#include "crypt.hpp"
#include "fides.cuh"
#include "data.hpp"
#include "encoding.hpp"
#include "naive.hpp"

#include <vector>
#include <iostream>

#include <FIDESlib/CKKS/Context.cuh>

std::vector<iteration_time_t> cpu_training(const std::vector<std::vector<double> > &data, const std::vector<double> &results, std::vector<double> &weights, const size_t training_iterations, const bool use_accelerated_training) {

    /// --------- Data adaptation to FHE packing. ---------

    std::vector<std::vector<double> > training_data_fhe;
    std::vector<std::vector<double> > training_results_fhe;
    std::vector<double> weights_fhe;

    std::cout << "Packing training data CPU..." << std::endl;

    const auto fhe_adapt_info = pack_data_fhe(data, results, weights, training_data_fhe, training_results_fhe, weights_fhe, num_slots);

    const size_t ciphertext_matrix_columns = std::get<0>(fhe_adapt_info);
    const size_t ciphertext_matrix_rows = std::get<1>(fhe_adapt_info);
    const size_t last_ciphertext_matrix_rows = std::get<2>(fhe_adapt_info);

    /// --------- Create the CPU context. ---------

    create_cpu_context(use_accelerated_training);

    const auto keys = cc_cpu->KeyGen();

    prepare_cpu_context(keys, ciphertext_matrix_columns, ciphertext_matrix_rows);

    /// --------- Encrypt the training data. ---------
    auto enc_weights = encrypt_data(weights_fhe, keys.publicKey);

    /// --------- Training. ---------

    std::vector<iteration_time_t> times;
    if (use_accelerated_training) {
        times = logistic_regression_cpu_train_accelerated(training_data_fhe, training_results_fhe, enc_weights, ciphertext_matrix_rows,
                                        ciphertext_matrix_columns, last_ciphertext_matrix_rows, training_iterations, keys.publicKey);
    }
    else {
        times = logistic_regression_cpu_train(training_data_fhe, training_results_fhe, enc_weights, ciphertext_matrix_rows,
                                        ciphertext_matrix_columns, last_ciphertext_matrix_rows, training_iterations, keys.publicKey);
    }

    /// --------- Decrypt, unpack and save the weights. ---------

    const auto dec_weights = decrypt_data(enc_weights, keys.secretKey);
    std::vector<double> new_weights;
    unpack_weights(dec_weights, new_weights, weights.size());

    weights = new_weights;

    return times;
}

std::pair<std::vector<iteration_time_t>, double> cpu_inference(const std::vector<std::vector<double> > &data, const std::vector<double> &results, const std::vector<double> &weights) {
    /// --------- Data adaptation to FHE packing. ---------

    std::vector<std::vector<double> > training_data_fhe;
    std::vector<std::vector<double> > training_results_fhe;
    std::vector<double> weights_fhe;

    std::cout << "Packing validation data CPU..." << std::endl;

    const auto fhe_adapt_info = pack_data_fhe(data, results, weights, training_data_fhe, training_results_fhe, weights_fhe, num_slots);

    const size_t ciphertext_matrix_columns = std::get<0>(fhe_adapt_info);
    const size_t ciphertext_matrix_rows = std::get<1>(fhe_adapt_info);
    const size_t last_ciphertext_matrix_rows = std::get<2>(fhe_adapt_info);

    /// --------- Create the CPU context. ---------

    create_cpu_context(false);

    const auto keys = cc_cpu->KeyGen();

    prepare_cpu_context(keys, ciphertext_matrix_columns, ciphertext_matrix_rows);

    /// --------- Encrypt the weights. ---------

    const auto enc_weights = encrypt_data(weights_fhe, keys.publicKey);

    /// --------- Inference. ---------

    auto times = logistic_regression_cpu_inference(training_data_fhe, enc_weights, ciphertext_matrix_rows, ciphertext_matrix_columns, last_ciphertext_matrix_rows, keys);

    /// --------- Decrypt the data ---------

    std::vector<std::vector<double>> unpacked_data;
    unpack_data(training_data_fhe, unpacked_data, ciphertext_matrix_rows, ciphertext_matrix_columns, last_ciphertext_matrix_rows, weights.size());

    size_t tp = 0.0, tn = 0.0, fp = 0.0, fn = 0.0;
    size_t correct = 0.0;

    for (size_t i = 0; i < unpacked_data.size(); i++) {
        if (unpacked_data[i][0] < 0.5 && results[i] == 0.0) {
            tn += 1;
            correct += 1;
        }
        else if (unpacked_data[i][0] < 0.5&& results[i] == 1.0) {
            fn += 1;
        }
        else if (unpacked_data[i][0] >= 0.5 && results[i] == 0.0) {
            fp += 1;
        }
        if (unpacked_data[i][0] >= 0.5 && results[i] == 1.0) {
            tp += 1;
            correct += 1;
        }
    }

    const double acc = (static_cast<double>(correct) / static_cast<double>(unpacked_data.size())) * 100;

    std::cout << "True positives: " << tp << ", False positives: " << fp << ", True negatives: " << tn << ", False negatives: " << fn << std::endl;
    std::cout << "Correct: " << correct << " from " << unpacked_data.size() << " results. Percentage: "  << acc << "%."<< std::endl;

    return std::make_pair(times, acc);
}

std::vector<iteration_time_t> naive_training(const std::vector<std::vector<double> > &data, const std::vector<double> &results, std::vector<double> &weights, size_t training_iterations , const bool use_accelerated_training) {

    /// --------- Data adaptation to FHE packing. ---------

    std::vector<std::vector<double> > training_data_fhe;
    std::vector<std::vector<double> > training_results_fhe;
    std::vector<double> weights_fhe;

    std::cout << "Packing training data Naive..." << std::endl;

    const auto fhe_adapt_info = pack_data_fhe(data, results, weights, training_data_fhe, training_results_fhe, weights_fhe, num_slots);

    const size_t ciphertext_matrix_columns = std::get<0>(fhe_adapt_info);
    const size_t ciphertext_matrix_rows = std::get<1>(fhe_adapt_info);
    const size_t last_ciphertext_matrix_rows = std::get<2>(fhe_adapt_info);

    /// --------- Training. ---------

    std::vector<iteration_time_t> times;
    if (use_accelerated_training) {
        times = logistic_regression_naive_train_accelerated(training_data_fhe, training_results_fhe, weights_fhe, ciphertext_matrix_rows,
                                        ciphertext_matrix_columns, last_ciphertext_matrix_rows, training_iterations);
    }
    else {
        times = logistic_regression_naive_train(training_data_fhe, training_results_fhe, weights_fhe, ciphertext_matrix_rows,
                                        ciphertext_matrix_columns, last_ciphertext_matrix_rows, training_iterations);
    }

    /// --------- Decrypt, unpack and save the weights. ---------

    std::vector<double> new_weights;
    unpack_weights(weights_fhe, new_weights, weights.size());
    weights = new_weights;

    return times;
}


std::pair<std::vector<iteration_time_t>, double> naive_inference(const std::vector<std::vector<double> > &data, const std::vector<double> &results, const std::vector<double> &weights) {

    /// --------- Data adaptation to FHE packing. ---------

    std::vector<std::vector<double> > training_data_fhe;
    std::vector<std::vector<double> > training_results_fhe;
    std::vector<double> weights_fhe;

    std::cout << "Packing validation data Naive..." << std::endl;

    const auto fhe_adapt_info = pack_data_fhe(data, results, weights, training_data_fhe, training_results_fhe, weights_fhe, num_slots);

    const size_t ciphertext_matrix_columns = std::get<0>(fhe_adapt_info);
    const size_t ciphertext_matrix_rows = std::get<1>(fhe_adapt_info);
    const size_t last_ciphertext_matrix_rows = std::get<2>(fhe_adapt_info);

    /// --------- Training. ---------

    auto times = logistic_regression_naive_inference(training_data_fhe, weights_fhe, ciphertext_matrix_rows,
                                    ciphertext_matrix_columns, last_ciphertext_matrix_rows);

    /// --------- Decrypt the data ---------

    std::vector<std::vector<double>> unpacked_data;
    unpack_data(training_data_fhe, unpacked_data, ciphertext_matrix_rows, ciphertext_matrix_columns, last_ciphertext_matrix_rows, weights.size());

    size_t tp = 0.0, tn = 0.0, fp = 0.0, fn = 0.0;
    size_t correct = 0.0;

    for (size_t i = 0; i < unpacked_data.size(); i++) {
        if (unpacked_data[i][0] < 0.5 && results[i] == 0.0) {
            tn += 1;
            correct += 1;
        }
        else if (unpacked_data[i][0] < 0.5&& results[i] == 1.0) {
            fn += 1;
        }
        else if (unpacked_data[i][0] >= 0.5 && results[i] == 0.0) {
            fp += 1;
        }
        if (unpacked_data[i][0] >= 0.5 && results[i] == 1.0) {
            tp += 1;
            correct += 1;
        }
    }

    const double acc = (static_cast<double>(correct) / static_cast<double>(unpacked_data.size())) * 100;

    std::cout << "True positives: " << tp << ", False positives: " << fp << ", True negatives: " << tn << ", False negatives: " << fn << std::endl;
    std::cout << "Correct: " << correct << " from " << unpacked_data.size() << " results. Percentage: "  << acc << "%."<< std::endl;

    return std::make_pair(times, acc);
}

std::vector<iteration_time_t> gpu_training(const std::vector<std::vector<double> > &data, const std::vector<double> &results, std::vector<double> &weights, size_t training_iterations, const bool use_accelerated_training) {

    /// --------- Data adaptation to FHE packing. ---------

    std::vector<std::vector<double> > training_data_fhe;
    std::vector<std::vector<double> > training_results_fhe;
    std::vector<double> weights_fhe;

    std::cout << "Packing training data GPU..." << std::endl;

    const auto fhe_adapt_info = pack_data_fhe(data, results, weights, training_data_fhe, training_results_fhe, weights_fhe, num_slots);

    const size_t ciphertext_matrix_columns = std::get<0>(fhe_adapt_info);
    const size_t ciphertext_matrix_rows = std::get<1>(fhe_adapt_info);
    const size_t last_ciphertext_matrix_rows = std::get<2>(fhe_adapt_info);

    /// --------- Create the CPU and GPU context. ---------

    create_cpu_context(use_accelerated_training);

    const auto keys = cc_cpu->KeyGen();

    prepare_cpu_context(keys, ciphertext_matrix_columns, ciphertext_matrix_rows);

    std::cout << "Generating GPU context..." << std::endl;

    auto raw_params = FIDESlib::CKKS::GetRawParams(cc_cpu);
    auto adapted_params = params.adaptTo(raw_params);
    FIDESlib::CKKS::Context cc_gpu(adapted_params, {0});

    prepare_gpu_context(cc_gpu, keys, ciphertext_matrix_columns, ciphertext_matrix_rows);

    /// --------- Encrypt the training data and load to GPU. ---------

    auto enc_weights = encrypt_data(weights_fhe, keys.publicKey);

    auto enc_weights_gpu = move_ciphertext(cc_gpu, enc_weights);

    /// --------- Training. ---------

    std::vector<iteration_time_t> times;
    if (use_accelerated_training) {
        times = logistic_regression_gpu_train_accelerated(training_data_fhe, training_results_fhe, enc_weights_gpu, ciphertext_matrix_rows,
                                        ciphertext_matrix_columns, last_ciphertext_matrix_rows, training_iterations, keys.publicKey);
    }
    else {
        times = logistic_regression_gpu_train(training_data_fhe, training_results_fhe, enc_weights_gpu, ciphertext_matrix_rows,
                                        ciphertext_matrix_columns, last_ciphertext_matrix_rows, training_iterations, keys.publicKey);
    }
    /// --------- Unload GPU, decrypt, unpack and save the weights. ---------

    move_back(cc_gpu, enc_weights, enc_weights_gpu);

    std::vector<double> dec_weights;

    dec_weights = decrypt_data(enc_weights, keys.secretKey);
    std::vector<double> new_weights;
    unpack_weights(dec_weights, new_weights, weights.size());
    weights = new_weights;
    
    return times;
}

std::pair<std::vector<iteration_time_t>, double> gpu_inference(const std::vector<std::vector<double> > &data, const std::vector<double> &results, const std::vector<double> &weights) {
    /// --------- Data adaptation to FHE packing. ---------

    std::vector<std::vector<double> > training_data_fhe;
    std::vector<std::vector<double> > training_results_fhe;
    std::vector<double> weights_fhe;

    std::cout << "Packing validation data GPU..." << std::endl;

    const auto fhe_adapt_info = pack_data_fhe(data, results, weights, training_data_fhe, training_results_fhe, weights_fhe, num_slots);

    const size_t ciphertext_matrix_columns = std::get<0>(fhe_adapt_info);
    const size_t ciphertext_matrix_rows = std::get<1>(fhe_adapt_info);
    const size_t last_ciphertext_matrix_rows = std::get<2>(fhe_adapt_info);

    /// --------- Create the CPU context. ---------

    create_cpu_context(false);

    const auto keys = cc_cpu->KeyGen();

    prepare_cpu_context(keys, ciphertext_matrix_columns, ciphertext_matrix_rows);

    auto raw_params = FIDESlib::CKKS::GetRawParams(cc_cpu);
    auto adapted_params = params.adaptTo(raw_params);
    FIDESlib::CKKS::Context cc_gpu(adapted_params, {0});

    prepare_gpu_context(cc_gpu, keys, ciphertext_matrix_columns, ciphertext_matrix_rows);

    /// --------- Encrypt the weights. ---------

    const auto enc_weights = encrypt_data(weights_fhe, keys.publicKey);
    auto enc_weights_gpu = move_ciphertext(cc_gpu, enc_weights);

    /// --------- Inference. ---------

    auto times = logistic_regression_gpu_inference(training_data_fhe, enc_weights_gpu, ciphertext_matrix_rows, ciphertext_matrix_columns, last_ciphertext_matrix_rows, keys);

    /// --------- Decrypt the data ---------

    std::vector<std::vector<double>> unpacked_data;
    unpack_data(training_data_fhe, unpacked_data, ciphertext_matrix_rows, ciphertext_matrix_columns, last_ciphertext_matrix_rows, weights.size());


    size_t tp = 0.0, tn = 0.0, fp = 0.0, fn = 0.0;
    size_t correct = 0.0;

    for (size_t i = 0; i < unpacked_data.size(); i++) {
        if (unpacked_data[i][0] < 0.5 && results[i] == 0.0) {
            tn += 1;
            correct += 1;
        }
        else if (unpacked_data[i][0] < 0.5&& results[i] == 1.0) {
            fn += 1;
        }
        else if (unpacked_data[i][0] >= 0.5 && results[i] == 0.0) {
            fp += 1;
        }
        if (unpacked_data[i][0] >= 0.5 && results[i] == 1.0) {
            tp += 1;
            correct += 1;
        }
    }

    const double acc = (static_cast<double>(correct) / static_cast<double>(unpacked_data.size())) * 100;

    std::cout << "True positives: " << tp << ", False positives: " << fp << ", True negatives: " << tn << ", False negatives: " << fn << std::endl;
    std::cout << "Correct: " << correct << " from " << unpacked_data.size() << " results. Percentage: "  << acc  << "%."<< std::endl;

    return std::make_pair(times, acc);
}