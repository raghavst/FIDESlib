#ifndef TRAIN_CUH
#define TRAIN_CUH

#include "data.hpp"

#include <vector>

std::vector<iteration_time_t> cpu_training(const std::vector<std::vector<double> > &data, const std::vector<double> &results, std::vector<double> &weights, size_t training_iterations, bool use_accelerated_training);

std::pair<std::vector<iteration_time_t>, double> cpu_inference(const std::vector<std::vector<double> > &data, const std::vector<double> &results, const std::vector<double> &weights);

std::vector<iteration_time_t> naive_training(const std::vector<std::vector<double> > &data, const std::vector<double> &results, std::vector<double> &weights, size_t training_iterations, bool use_accelerated_training);

std::pair<std::vector<iteration_time_t>, double> naive_inference(const std::vector<std::vector<double> > &data, const std::vector<double> &results, const std::vector<double> &weights);

std::vector<iteration_time_t> gpu_training(const std::vector<std::vector<double> > &data, const std::vector<double> &results, std::vector<double> &weights, size_t training_iterations, bool use_accelerated_training);

std::pair<std::vector<iteration_time_t>, double> gpu_inference(const std::vector<std::vector<double> > &data, const std::vector<double> &results, const std::vector<double> &weights);

#endif //TRAIN_CUH
