#include "fides.cuh"

#include <CKKS/Bootstrap.cuh>
#include <CKKS/BootstrapPrecomputation.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <CKKS/Ciphertext.cuh>

#include "crypt.hpp"
#include "data.hpp"
#include "fhe.hpp"

std::vector<FIDESlib::PrimeRecord> p64{
		    {.p = 2305843009218281473}, {.p = 2251799661248513}, {.p = 2251799661641729}, {.p = 2251799665180673},
            {.p = 2251799682088961},	{.p = 2251799678943233}, {.p = 2251799717609473}, {.p = 2251799710138369},
            {.p = 2251799708827649},	{.p = 2251799707385857}, {.p = 2251799713677313}, {.p = 2251799712366593},
            {.p = 2251799716691969},	{.p = 2251799714856961}, {.p = 2251799726522369}, {.p = 2251799726129153},
            {.p = 2251799747493889},	{.p = 2251799741857793}, {.p = 2251799740416001}, {.p = 2251799746707457},
            {.p = 2251799756013569},	{.p = 2251799775805441}, {.p = 2251799763091457}, {.p = 2251799767154689},
            {.p = 2251799765975041},	{.p = 2251799770562561}, {.p = 2251799769776129}, {.p = 2251799772266497},
            {.p = 2251799775281153},	{.p = 2251799774887937}, {.p = 2251799797432321}, {.p = 2251799787995137},
            {.p = 2251799787601921},	{.p = 2251799791403009}, {.p = 2251799789568001}, {.p = 2251799795466241},
            {.p = 2251799807131649},	{.p = 2251799806345217}, {.p = 2251799805165569}, {.p = 2251799813554177},
            {.p = 2251799809884161},	{.p = 2251799810670593}, {.p = 2251799818928129}, {.p = 2251799816568833},
            {.p = 2251799815520257}};

std::vector<FIDESlib::PrimeRecord> sp64{
		    {.p = 2305843009218936833}, {.p = 2305843009220116481}, {.p = 2305843009221820417}, {.p = 2305843009224179713},
            {.p = 2305843009225228289}, {.p = 2305843009227980801}, {.p = 2305843009229160449}, {.p = 2305843009229946881},
            {.p = 2305843009231650817}, {.p = 2305843009235189761}, {.p = 2305843009240301569}, {.p = 2305843009242923009},
            {.p = 2305843009244889089}, {.p = 2305843009245413377}, {.p = 2305843009247641601}};

FIDESlib::CKKS::Parameters params{.logN = 16, .L = 29, .dnum = 4, .primes = p64, .Sprimes = sp64, .batch=12};

std::unique_ptr<FIDESlib::CKKS::Plaintext> first_column_mask_gpu = nullptr;

void prepare_gpu_context(FIDESlib::CKKS::Context &cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys, const size_t matrix_cols, const size_t matrix_rows) {
	// Safety check
	if (matrix_cols*matrix_rows != num_slots) {
		std::cerr << "Matrix size is different from number of slots" << std::endl;
		exit(EXIT_FAILURE);
	}
	// Multiplication keys.
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_gpu);
	eval_key_gpu.Initialize(cc_gpu, eval_key);
	FIDESlib::CKKS::Context::AddEvalKey(std::move(eval_key_gpu));
	// Rotation keys for same row value propagation and accumulation by rows and columns.
	for (size_t j = 1; j < matrix_cols; j <<= 1) {
		auto pos_rot_key = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, static_cast<int>(j), cc_cpu);
		auto neg_rot_key = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, -static_cast<int>(j), cc_cpu);
		FIDESlib::CKKS::KeySwitchingKey pos_rot_key_gpu(cc_gpu);
		FIDESlib::CKKS::KeySwitchingKey neg_rot_key_gpu(cc_gpu);
		pos_rot_key_gpu.Initialize(cc_gpu, pos_rot_key);
		neg_rot_key_gpu.Initialize(cc_gpu, neg_rot_key);
		cc_gpu.AddRotationKey(static_cast<int>(j),std::move(pos_rot_key_gpu));
		cc_gpu.AddRotationKey(-static_cast<int>(j),std::move(neg_rot_key_gpu));
	}
	for (size_t i = matrix_cols; i < matrix_cols*matrix_rows; i <<= 1) {
		auto col_rot_key = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, static_cast<int>(i), cc_cpu);
		FIDESlib::CKKS::KeySwitchingKey col_rot_key_gpu(cc_gpu);
		col_rot_key_gpu.Initialize(cc_gpu, col_rot_key);
		cc_gpu.AddRotationKey(static_cast<int>(i),std::move(col_rot_key_gpu));
	}
	// First column matrix mask creation.
	auto naive_1st_col_mask = std::vector<double>(matrix_cols*matrix_rows, 0);
	for (size_t i = 0; i < matrix_cols*matrix_rows; i+=matrix_cols) {
		naive_1st_col_mask[i] = 1;
	}
	auto first_column_mask = cc_cpu->MakeCKKSPackedPlaintext(naive_1st_col_mask);
	auto raw_pt = FIDESlib::CKKS::GetRawPlainText(cc_cpu, first_column_mask);

	if (first_column_mask_gpu == nullptr) {
		first_column_mask_gpu = std::make_unique<FIDESlib::CKKS::Plaintext>(FIDESlib::CKKS::Plaintext(cc_gpu, raw_pt));
	}


	// Bootstrapping config.
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc_cpu, keys, static_cast<int>(matrix_cols), cc_gpu);
}

FIDESlib::CKKS::Ciphertext move_ciphertext(FIDESlib::CKKS::Context &cc_gpu, const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) {
	const FIDESlib::CKKS::RawCipherText raw_ct = FIDESlib::CKKS::GetRawCipherText(cc_cpu, ct);
	FIDESlib::CKKS::Ciphertext ct_gpu (cc_gpu, raw_ct);
	return ct_gpu;
}

std::vector<FIDESlib::CKKS::Ciphertext> move_ciphertext(FIDESlib::CKKS::Context &cc_gpu, const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &cts) {
	std::vector<FIDESlib::CKKS::Ciphertext> cts_gpu;
	for (auto & ct : cts) {
		cts_gpu.push_back(move_ciphertext(cc_gpu, ct));
	}
	return cts_gpu;
}

void move_back(const FIDESlib::CKKS::Context &cc_gpu, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &res, FIDESlib::CKKS::Ciphertext &ct) {
	FIDESlib::CKKS::RawCipherText raw_ct;
	ct.store(cc_gpu, raw_ct);
	FIDESlib::CKKS::GetOpenFHECipherText(res, raw_ct);
}

void move_back(const FIDESlib::CKKS::Context &cc_gpu, std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &res, std::vector<FIDESlib::CKKS::Ciphertext> &ct) {
	for (size_t i = 0; i < ct.size(); i++) {
		move_back(cc_gpu, res[i], ct[i]);
	}
}

/**
 * Approximation of the sigmoid function.
 * @param ct FIDESlib ciphertext.
 */
void activation_function_cpu(FIDESlib::CKKS::Ciphertext &ct) {
	// Auxiliary ciphertext.
	FIDESlib::CKKS::Ciphertext ct3(ct.cc);
	ct3.copy(ct);
	// Get the x^3 term.
	ct3.square(FIDESlib::CKKS::Context::GetEvalKey());
	ct3.mult(ct, FIDESlib::CKKS::Context::GetEvalKey());
	// Multiply coefficients.
	ct.multScalar(0.15);
	ct3.multScalar(-0.0015);
	// Add terms.
	ct.addScalar(0.5);
	ct.add(ct3);
}

/**
 * Accumulate the values of each row on the first column of the ciphertext matrix.
 * @param ct Matrix where to perform the accumulation.
 * @param num_columns Number of columns of the matrix.
 */
void row_accumulate(FIDESlib::CKKS::Ciphertext &ct, const size_t num_columns) {
	FIDESlib::CKKS::Ciphertext rot(ct.cc);
	for (size_t j = 1; j < num_columns; j <<= 1) {
		rot.copy(ct);
		rot.rotate(static_cast<int>(j), ct.cc.GetRotationKey(static_cast<int>(j)));
		ct.add(rot);
	}
}

/**
 * Propagate the values of the first column to the rest columns of the ciphertext matrix.
 * @param ct Matrix where to perform the propagation.
 * @param num_columns Number of columns of the matrix.
 */
void row_propagate(FIDESlib::CKKS::Ciphertext &ct, const size_t num_columns) {
	FIDESlib::CKKS::Ciphertext rot(ct.cc);
	for (size_t j = 1; j < num_columns; j <<= 1) {
		rot.copy(ct);
		rot.rotate(-static_cast<int>(j), ct.cc.GetRotationKey(-static_cast<int>(j)));
		ct.add(rot);
	}
}

/**
 * Accumulate the values by column. Each column ends with the same value on all rows.
 * @param ct Matrix where to perform the accumulation.
 * @param num_rows Number of rows of the matrix.
 * @param num_columns Number of columns of the matrix.
 */
void column_accumulate(FIDESlib::CKKS::Ciphertext &ct, const size_t num_rows, const size_t num_columns) {
	FIDESlib::CKKS::Ciphertext rot(ct.cc);
	for (size_t j = num_columns ; j < num_rows*num_columns; j <<= 1) {
		rot.copy(ct);
		rot.rotate(static_cast<int>(j), ct.cc.GetRotationKey(static_cast<int>(j)));
		ct.add(rot);
	}
}

/**
 * Perform an iteration of LR Training.
 * @param data Data matrix.
 * @param results Results matrix.
 * @param weights Weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @param learning_rate Desired learning rate for the iteration.
 * @return Iteration times.
 */
iteration_time_t logistic_regression_gpu_train_iteration(const FIDESlib::CKKS::Ciphertext &data,
											 const FIDESlib::CKKS::Ciphertext &results,
											 FIDESlib::CKKS::Ciphertext &weights,
											 const size_t rows,
											 const size_t columns,
											 const size_t batch_size,
											 const double learning_rate) {
	auto start_time = std::chrono::high_resolution_clock::now();

	/// Step 1. Multiply weight matrix by data matrix.
	FIDESlib::CKKS::Ciphertext ct(data.cc);
	ct.copy(data);
	ct.mult(weights, FIDESlib::CKKS::Context::GetEvalKey());

	/// Step 2. Accumulate results on the first column (inner product result).
	row_accumulate(ct, columns);

	/// Step 3. Apply the activation function.
	activation_function_cpu(ct);

	/// Step 4. Remove garbage from the ciphertext by masking the last result.
	ct.multPt(*first_column_mask_gpu, true);

	/// Step 5. Compute loss (ours - expected).
	ct.sub(results);

	/// Step 6. Propagation of first column value to the rest of the columns.
	row_propagate(ct, columns);

	/// Step 7. Multiply the result by the original data.
	ct.mult(data, FIDESlib::CKKS::Context::GetEvalKey());

	/// Step 8. Compute the gradient loss across all data rows.
	column_accumulate(ct, rows, columns);

	/// Step 9. Adjust to learning rate and batch configuration.
	ct.multScalar((learning_rate)/static_cast<double>(batch_size));

	/// Step 10. Update original weights.
	weights.sub(ct);

	auto boot_time = std::chrono::high_resolution_clock::now();

	/// Boostrapping
	if (bootstrap_every_two) {
        static bool do_boot = false;
		if (do_boot) {
			FIDESlib::CKKS::Bootstrap(weights, static_cast<int>(columns));
		}
		do_boot = !do_boot;
    }
    else {
    	FIDESlib::CKKS::Bootstrap(weights, static_cast<int>(columns));
    }

	auto end_time = std::chrono::high_resolution_clock::now();

	auto elapsed_total = std::chrono::duration_cast<time_unit_t>(end_time - start_time);
	auto elapsed_boot = std::chrono::duration_cast<time_unit_t>(end_time - boot_time);
	return std::make_pair(elapsed_total, elapsed_boot);
}

std::vector<iteration_time_t> logistic_regression_gpu_train(const std::vector<std::vector<double>> &data,
								   const std::vector<std::vector<double>> &results,
								   FIDESlib::CKKS::Ciphertext &weights,
								   const size_t rows,
								   const size_t columns,
								   const size_t samples_last_ciphertext,
								   const size_t training_iterations,
								   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &public_key) {

	std::vector<iteration_time_t> times(training_iterations);
	std::cout << "Doing " << training_iterations << " training iterations" << std::endl;
	for (size_t it = 0; it < training_iterations; ++it) {
		const size_t data_idx = it % data.size();
		const size_t batch_size = data_idx == data.size() - 1 ? samples_last_ciphertext : rows;
		const double learning_rate = 10/(static_cast<double>(it)+1) > 0.005 ? 10/(static_cast<double>(it)+1) : 0.005;
		const auto enc_data = encrypt_data(data[data_idx], public_key);
		const auto enc_results = encrypt_data(results[data_idx], public_key);
		auto enc_data_gpu = move_ciphertext(weights.cc, enc_data);
		auto enc_res_gpu = move_ciphertext(weights.cc, enc_results);
		times[it] = logistic_regression_gpu_train_iteration(enc_data_gpu, enc_res_gpu, weights, rows, columns, batch_size, learning_rate);
	}
	return times;
}

/**
 * Perform an iteration of LR Inference.
 * @param data Data matrix.
 * @param weights Weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @return Iteration times.
 */
iteration_time_t logistic_regression_gpu_inference_iteration(FIDESlib::CKKS::Ciphertext &data,
									   const FIDESlib::CKKS::Ciphertext &weights,
									   const size_t rows,
									   const size_t columns,
									   const size_t batch_size) {
	auto start_time = std::chrono::high_resolution_clock::now();

	/// Step 1. Multiply weight matrix by data matrix.
	data.mult(weights, FIDESlib::CKKS::Context::GetEvalKey());

	/// Step 2. Accumulate results on the first column (inner product result).
	row_accumulate(data, columns);

	/// Step 3. Apply the activation function.
	activation_function_cpu(data);

	/// Step 4. Remove garbage from the ciphertext by masking the last result.
	data.multPt(*first_column_mask_gpu);

	auto end_time = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<time_unit_t>(end_time - start_time);
	return std::make_pair(elapsed, time_unit_t::zero());
}

std::vector<iteration_time_t> logistic_regression_gpu_inference(std::vector<std::vector<double>> &data,
								   const FIDESlib::CKKS::Ciphertext &weights,
								   const size_t rows,
								   const size_t columns,
								   const size_t samples_last_ciphertext,
								   const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys) {
	std::vector<iteration_time_t> times(data.size());
	for (size_t it = 0; it < data.size(); ++it) {
		const size_t batch_size = it == data.size() - 1 ? samples_last_ciphertext : rows;
		auto enc_data = encrypt_data(data[it], keys.publicKey);
		auto enc_data_gpu = move_ciphertext(weights.cc, enc_data);
		times[it] = logistic_regression_gpu_inference_iteration(enc_data_gpu, weights, rows, columns, batch_size);
		move_back(weights.cc, enc_data, enc_data_gpu);
		data[it]= decrypt_data(enc_data, keys.secretKey);
	}
	return times;
}

/**
 * Perform an iteration of LR Training (with NAG).
 * @param data Data matrix.
 * @param results Results matrix.
 * @param weights Weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @param learning_rate Desired learning rate for the iteration.
 * @param momentum Momentum of accelerated learning.
 * @return Iteration times.
 */
iteration_time_t logistic_regression_gpu_train_iteration_accelerated(const FIDESlib::CKKS::Ciphertext &data,
											 const FIDESlib::CKKS::Ciphertext &results,
											 FIDESlib::CKKS::Ciphertext &weights,
											 const size_t rows,
											 const size_t columns,
											 const size_t batch_size,
											 const double learning_rate,
											 const double momentum,
											 FIDESlib::CKKS::Ciphertext &phi_gpu,
											 FIDESlib::CKKS::Ciphertext &phi_prev_gpu) {

	const auto start_time = std::chrono::high_resolution_clock::now();

	/// Step 1. Multiply weight matrix by data matrix.
	FIDESlib::CKKS::Ciphertext ct(data.cc);
	ct.copy(data);
	ct.mult(weights, FIDESlib::CKKS::Context::GetEvalKey());

	/// Step 2. Accumulate results on the first column (inner product result).
	row_accumulate(ct, columns);

	/// Step 3. Apply the activation function.
	activation_function_cpu(ct);

	/// Step 4. Remove garbage from the ciphertext by masking the last result.
	ct.multPt(*first_column_mask_gpu, true);

	/// Step 5. Compute loss (ours - expected).
	ct.sub(results);

	/// Step 6. Propagation of first column value to the rest of the columns.
	row_propagate(ct, columns);

	/// Step 7. Multiply the result by the original data.
	ct.mult(data, FIDESlib::CKKS::Context::GetEvalKey());

	/// Step 8. Compute the gradient loss across all data rows.
	column_accumulate(ct, rows, columns);

	/// Step 9. Adjust to learning rate and batch configuration.
	ct.multScalar((learning_rate)/static_cast<double>(batch_size));

	// Step 10. Calculate current momentum.
	weights.sub(ct);
	phi_gpu.copy(weights);

	// Step 11. Calculate phi.
	phi_prev_gpu.sub(phi_gpu);
	phi_prev_gpu.multScalar(momentum);
	phi_gpu.sub(phi_prev_gpu);

	// Step 12. Save momentum for next iteration.
	phi_prev_gpu.copy(weights);

	// Step 13. Save final values.
	weights.copy(phi_gpu);

	const auto boot_time = std::chrono::high_resolution_clock::now();

	/// Boostrapping
	if (bootstrap_every_two) {
		static bool do_boot = false;
		if (do_boot) {
			FIDESlib::CKKS::Bootstrap(weights, static_cast<int>(columns));
			FIDESlib::CKKS::Bootstrap(phi_prev_gpu, static_cast<int>(columns));
		}
		do_boot = !do_boot;
	}
	else {
		FIDESlib::CKKS::Bootstrap(weights, static_cast<int>(columns));
	}

	const auto end_time = std::chrono::high_resolution_clock::now();
	auto elapsed_total = std::chrono::duration_cast<time_unit_t>(end_time - start_time);
	auto elapsed_boot = std::chrono::duration_cast<time_unit_t>(end_time - boot_time);
	return std::make_pair(elapsed_total, elapsed_boot);
}

std::vector<iteration_time_t> logistic_regression_gpu_train_accelerated(const std::vector<std::vector<double>> &data,
								   const std::vector<std::vector<double>> &results,
								   FIDESlib::CKKS::Ciphertext &weights,
								   const size_t rows,
								   const size_t columns,
								   const size_t samples_last_ciphertext,
								   const size_t training_iterations,
								   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk) {

	auto phi_gpu = FIDESlib::CKKS::Ciphertext(weights.cc, FIDESlib::CKKS::GetRawCipherText(cc_cpu, phi));
	auto phi_prev_gpu = FIDESlib::CKKS::Ciphertext(weights.cc, FIDESlib::CKKS::GetRawCipherText(cc_cpu, phi_prev));
	std::vector<iteration_time_t> times(training_iterations);
	std::cout << "Doing " << training_iterations << " training iterations (NAG)" << std::endl;
	for (size_t it = 0; it < training_iterations; ++it) {
		const size_t data_idx = it % data.size();
		const size_t batch_size = data_idx == data.size() - 1 ? samples_last_ciphertext : rows;
		const double learning_rate = 10/(static_cast<double>(it)+1) > 0.005 ? 10/(static_cast<double>(it)+1) : 0.005;
		const double momentum = 1.0 / static_cast<double>(training_iterations);
		const auto enc_data = encrypt_data(data[data_idx], pk);
		const auto enc_results = encrypt_data(results[data_idx], pk);
		auto enc_data_gpu = move_ciphertext(weights.cc, enc_data);
		auto enc_res_gpu = move_ciphertext(weights.cc, enc_results);
		times[it] = logistic_regression_gpu_train_iteration_accelerated(enc_data_gpu, enc_res_gpu, weights, rows, columns, batch_size, learning_rate, momentum, phi_gpu, phi_prev_gpu);
	}
	return times;
}