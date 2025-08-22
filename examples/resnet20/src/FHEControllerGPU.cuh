#ifndef FHECONTROLLER_H
#define FHECONTROLLER_H

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <FIDESlib/CKKS/Ciphertext.cuh>
#include <FIDESlib/CKKS/Context.cuh>
#include <FIDESlib/CKKS/Plaintext.cuh>

#include <thread>

#include "utils.hpp"

using namespace lbcrypto;
using namespace std;
using namespace std::chrono;

using namespace utils;

using Ptxt = FIDESlib::CKKS::Plaintext;
using Ctxt = FIDESlib::CKKS::Ciphertext;
using PtxtCPU = Plaintext;
using CtxtCPU = Ciphertext<DCRTPoly>;

extern int dim1;

class FHEControllerGPU {
public:
	CryptoContext<DCRTPoly> context_cpu;
	FIDESlib::CKKS::Context *context = nullptr;
	int circuit_depth;
	int num_slots;

	bool load_on_the_fly_gpu = false;

	// Rotations per layer.
	const std::vector<int> rots_layer_1 = {1, -1, 2, 32, -32, 33, -33, 31, -31, -1024};
	const std::vector<int> rots_layer_2_down = {1, 2, 4, 8, 64 - 16, -(1024 - 256), (1024 - 256) * 32, -8192, -16384};
	const std::vector<int> rots_layer_2 = {1, -1, 16, -16, -17, 17, -15, 15, -256};
	const std::vector<int> rots_layer_3_down = {1, 2, 4, 32 - 8, -(256 - 64), (256 - 64) * 64, -4096, -4096 - 4096};
	const std::vector<int> rots_layer_3 = {1, -1, 8, -8, -7, 7, -9, 9, -64};
	const std::vector<int> rots_layer_final = {1, 2, 4, 8, 16, 32, -15, 64, 128, 256, 512, 1024, 2048};
	std::vector<int> rots_total = {1,	-1,	  2,	4,	   7,	 -7,	8,	   -8,	  9,	  -9,
								   15,	-15,  16,	-16,   17,	 -17,	24,	   31,	  -31,	  32,
								   -32, 33,	  -33,	48,	   64,	 -64,	128,   -192,  256,	  -256,
								   512, -768, 1024, -1024, 2048, -4096, -8192, 12288, -16348, 24576};

	// Raw data for each layer.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> weights_conv1bn1;
	FIDESlib::CKKS::RawPlainText bias_conv1bn1;
	std::map<std::tuple<int, int>, std::vector<std::vector<FIDESlib::CKKS::RawPlainText>>> weights_layers;
	std::map<std::tuple<int, int>, FIDESlib::CKKS::RawPlainText> bias_layers;
	std::map<std::tuple<int, int>, std::vector<FIDESlib::CKKS::RawPlainText>> weights_layers_dx;
	std::map<std::tuple<int, int>, std::tuple<FIDESlib::CKKS::RawPlainText, FIDESlib::CKKS::RawPlainText>>
			bias_layers_sx;
	std::map<std::tuple<int, int>, std::tuple<FIDESlib::CKKS::RawPlainText, FIDESlib::CKKS::RawPlainText>>
			bias_layers_dx;
	FIDESlib::CKKS::RawPlainText weight_final_layer;
	FIDESlib::CKKS::RawPlainText initial_layer_mask;
	FIDESlib::CKKS::RawPlainText final_layer_mask;
	std::vector<FIDESlib::CKKS::RawPlainText> full_pack_mask;
	FIDESlib::CKKS::RawCipherText zero_ctxt;
	std::vector<FIDESlib::CKKS::RawPlainText> first_and_second_n_masks_l2;
	std::vector<FIDESlib::CKKS::RawPlainText> mask_first_n_mods_l2;
	std::vector<FIDESlib::CKKS::RawPlainText> mask_channel_l2;
	std::vector<FIDESlib::CKKS::RawPlainText> first_and_second_n_masks_l3;
	std::vector<FIDESlib::CKKS::RawPlainText> mask_first_n_mods_l3;
	std::vector<FIDESlib::CKKS::RawPlainText> mask_channel_l3;

	// Data on GPU.
	std::vector<std::vector<Ptxt *>> weights_conv1bn1_gpu;
	Ptxt *bias_conv1bn1_gpu = nullptr;
	std::map<std::tuple<int, int>, std::vector<std::vector<Ptxt *>>> weights_layers_gpu;
	std::map<std::tuple<int, int>, Ptxt *> bias_layers_gpu;
	std::map<std::tuple<int, int>, std::vector<Ptxt *>> weights_layers_dx_gpu;
	std::map<std::tuple<int, int>, std::tuple<Ptxt *, Ptxt *>> bias_layers_sx_gpu;
	std::map<std::tuple<int, int>, std::tuple<Ptxt *, Ptxt *>> bias_layers_dx_gpu;
	Ptxt *weight_final_layer_gpu = nullptr;
	Ptxt *initial_layer_mask_gpu = nullptr;
	Ptxt *final_layer_mask_gpu = nullptr;
	std::vector<Ptxt *> full_pack_mask_gpu;
	Ctxt *zero_ctxt_gpu = nullptr;
	std::vector<Ptxt *> first_and_second_n_masks_l2_gpu;
	std::vector<Ptxt *> mask_first_n_mods_l2_gpu;
	std::vector<Ptxt *> mask_channel_l2_gpu;
	std::vector<Ptxt *> first_and_second_n_masks_l3_gpu;
	std::vector<Ptxt *> mask_first_n_mods_l3_gpu;
	std::vector<Ptxt *> mask_channel_l3_gpu;

	// Scaling factors per layer.
	double prescale;
	double convbn_initial_scale = 0.90;
	double convbn_1_1_scale = 1.00;
	double convbn_1_2_scale = 0.52;
	double convbn_2_1_scale = 0.55;
	double convbn_2_2_scale = 0.36;
	double convbn_3_1_scale = 0.63;
	double convbn_3_2_scale = 0.42;
	double convbn_4_1_sx_scale = 0.57;
	double convbn_4_1_dx_scale = 0.40;
	double convbn_4_2_scale = 0.40;
	double convbn_5_1_scale = 0.76;
	double convbn_5_2_scale = 0.37;
	double convbn_6_1_scale = 0.63;
	double convbn_6_2_scale = 0.25;
	double convbn_7_1_sx_scale = 0.63;
	double convbn_7_1_dx_scale = 0.40;
	double convbn_7_2_scale = 0.40;
	double convbn_8_1_scale = 0.57;
	double convbn_8_2_scale = 0.33;
	double convbn_9_1_scale = 0.69;
	double convbn_9_2_scale = 0.10;

	// Weight pre-loading.
	void load_weights();
	void clear_weights_l1();
	void clear_weights_l2();
	void clear_weights_l3();
	void load_weights_l1();
	void load_weights_l2();
	void load_weights_l3();

	FHEControllerGPU() = default;

	/*
	 * Context generating/loading stuff.
	 */
	void create_gpu_context(bool load_on_the_fly_to_gpu, const vector<int> &devices);
	void generate_context(int log_ring, int log_scale, int log_primes, int digits_hks, int cts_levels, int stc_levels,
						  int relu_deg, bool serialize = false);
	void load_context(bool verbose = true);

	/*
	 * Move things between CPU and GPU.
	 */
	Ctxt move(const CtxtCPU &c);
	void move_back(Ctxt &c, const CtxtCPU &c_cpu) const;
	Ptxt move_ptxt(const PtxtCPU &c);

	/*
	 * Generating bootstrapping and rotation keys stuff.
	 */
	void generate_bootstrapping_keys(const vector<int> &bootstrap_slots) const;
	void generate_rotation_keys(const vector<int> &rotations) const;
	void gen_keys(vector<int> &rotations, const vector<int> &bootstrap_slots) const;
	void load_keys(const vector<int> &rotations, int boostrap_slots);

	/*
	 * CKKS Encoding/Decoding/Encryption/Decryption.
	 */
	[[nodiscard]] PtxtCPU encode(const vector<double> &vec, int level, int plaintext_num_slots) const;
	[[nodiscard]] PtxtCPU encode(double val, int level, int plaintext_num_slots) const;
	[[nodiscard]] CtxtCPU encrypt(const vector<double> &vec, int level = 0, int plaintext_num_slots = 0) const;
	[[nodiscard]] PtxtCPU decrypt(const CtxtCPU &c) const;
	[[nodiscard]] vector<double> decrypt_tovector(const CtxtCPU &c, int slots) const;

	/*
	 * Essential operations.
	 */
	void bootstrap(Ctxt &c, bool prescale) const;
	void relu(Ctxt &c, double scale) const;

	/*
	 * Convolutional Neural Network functions.
	 */
	Ctxt convbn_initial(const Ctxt &in);
	Ctxt convbn(const Ctxt &in, int layer, int n);
	Ctxt convbn2(const Ctxt &in, int layer, int n);
	Ctxt convbn3(const Ctxt &in, int layer, int n);
	array<Ctxt, 2> convbn1632sx(const Ctxt &in, int layer, int n);
	array<Ctxt, 2> convbn1632dx(const Ctxt &in, int layer, int n);
	array<Ctxt, 2> convbn3264sx(const Ctxt &in, int layer, int n);
	array<Ctxt, 2> convbn3264dx(const Ctxt &in, int layer, int n);

	/*
	 * Downsampling.
	 */
	Ctxt downsample1024to256(const Ctxt &c1, const Ctxt &c2);
	Ctxt downsample256to64(const Ctxt &c1, const Ctxt &c2);

	/*
	 * Extras.
	 */
	void rotsum(Ctxt &in, int slots);
	void rotsum_padded(Ctxt &in, int slots);
	void repeat(Ctxt &in, int slots);

	/*
	 * Masking things
	 */
	PtxtCPU gen_mask(int n, int level);
	PtxtCPU mask_first_n(int n, int level);
	PtxtCPU mask_second_n(int n, int level);
	PtxtCPU mask_first_n_mod(int n, int padding, int pos, int level);
	PtxtCPU mask_first_n_mod2(int n, int padding, int pos, int level);
	PtxtCPU mask_channel(int n, int level);
	PtxtCPU mask_channel_2(int n, int level);
	PtxtCPU mask_from_to(int from, int to, int level);
	PtxtCPU mask_mod(int n, int level, double custom_val);

	/*
	 * Boot precision.
	 */
	void bootstrap_precision(const Ctxt &c);

	int relu_degree = 119;
	string parameters_folder = "NO_FOLDER";

private:
	KeyPair<DCRTPoly> key_pair;
	vector<uint32_t> level_budget = {4, 4};
};

#endif
