#ifndef HELPER_HPP
#define HELPER_HPP

#include "vector"
#include "string"

/**
 * Enum of available datasets.
 */
typedef enum {
    RANDOM,
    MNIST,
} dataset_t;

/**
 * Enum of types of execution.
 */
typedef enum {
    TRAIN,
    VALIDATION,
    PERFORMANCE,
} exec_t;

/**
 * Enum with the backends.
 */
typedef enum {
    NAIVE,
    CPU,
    GPU,
} backend_t;

/**
 * Get the dataset name as a string.
 * @param dataset Desired dataset.
 * @return Dataset name.
 */
std::string dataset_name (dataset_t dataset);

/**
 * Get the dataset type from a name.
 * @param dataset Dataset name.
 * @return Dataset type.
 */
dataset_t dataset_from (const std::string& dataset);

/**
 * Get the backend from his name.
 * @param backend Backend name.
 * @return Backend type.
 */
backend_t backend_from(const std::string& backend);

/**
 * Check for accelerated status on a string.
 * @param acc Accelerated string.
 * @return Is accelerated.
 */
bool is_acceler_from(const std::string& acc);

/**
 * Get the weight file path.
 * @param backend Desired backend.
 * @param dataset Used dataset.
 * @return Weight file path.
 */
std::string weights_path(backend_t backend, dataset_t dataset);

/**
 * Create the path for a file to store the times.
 * @param dataset Dataset used.
 * @param exec Execution type.
 * @param backend Backend used.
 * @return File path.
 */
std::string times_path(const dataset_t dataset, const exec_t exec, const backend_t backend);

/**
 * Loads and prepares the data to be LR compliant.
 * @param dataset Dataset used.
 * @param exec Execution type. Decide what data to load.
 * @param data Stores the training data samples.
 * @param results Stores the expected results for each validation data sample.
 * @return Number of features of the data set samples.
 */
size_t prepare_data_csv(dataset_t dataset, exec_t exec,
                  std::vector<std::vector<double> > &data,
                  std::vector<double> &results);

/**
 *
 * @param data Data samples to be packed.
 * @param results Data results to be packed.
 * @param weights Weights to be packed.
 * @param data_fhe Resulting packed data.
 * @param results_fhe Resulting packed results.
 * @param weights_fhe Resulting packed weights.
 * @param num_slots
 * @return Tuple that contains:
 * 1º Approximated number of features for each datum.
 * 2º Number of datum encoded on each row of data_fhe.
 * 3º Number of datum encoded on the last row of data_fhe (needed if data.size() is not power of 2).
 */
std::tuple<size_t, size_t, size_t> pack_data_fhe(const std::vector<std::vector<double> > &data,
                                                    const std::vector<double> &results,
                                                    const std::vector<double> &weights,
                                                    std::vector<std::vector<double> > &data_fhe,
                                                    std::vector<std::vector<double> > &results_fhe,
                                                    std::vector<double> &weights_fhe,
                                                    size_t num_slots);

#endif //HELPER_HPP
