#include "FHEControllerGPU.cuh"

#include "../../../include/CKKS/AccumulateBroadcast.cuh"
#include "FIDESlib/CKKS/ApproxModEval.cuh"
#include "FIDESlib/CKKS/Bootstrap.cuh"
#include "FIDESlib/CKKS/Context.cuh"
#include "FIDESlib/CKKS/KeySwitchingKey.cuh"

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

FIDESlib::CKKS::Parameters params{.logN = 16, .L = 29, .dnum = 4, .primes = p64, .Sprimes = sp64, .batch = 12};

lbcrypto::Plaintext encodeExt(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, const std::vector<double> &value,
							  size_t noiseScaleDeg, uint32_t L, int slots) {

	uint32_t M = cc->GetCyclotomicOrder();
	const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

	ILDCRTParams<DCRTPoly::Integer> elementParams = *(cryptoParams->GetElementParams());
	uint32_t towersToDrop = 0;
	if (L != 0) {
		towersToDrop = elementParams.GetParams().size() - L - 1;
	}
	for (uint32_t i = 0; i < towersToDrop; i++) {
		elementParams.PopLastParam();
	}
	auto paramsQ = elementParams.GetParams();
	usint sizeQ = paramsQ.size();
	auto paramsP = cryptoParams->GetParamsP()->GetParams();
	usint sizeP = paramsP.size();
	std::vector<NativeInteger> moduli(sizeQ + sizeP);
	std::vector<NativeInteger> roots(sizeQ + sizeP);
	for (size_t i = 0; i < sizeQ; i++) {
		moduli[i] = paramsQ[i]->GetModulus();
		roots[i] = paramsQ[i]->GetRootOfUnity();
	}

	for (size_t i = 0; i < sizeP; i++) {
		moduli[sizeQ + i] = paramsP[i]->GetModulus();
		roots[sizeQ + i] = paramsP[i]->GetRootOfUnity();
	}

	auto elementParamsPtr = std::make_shared<ILDCRTParams<DCRTPoly::Integer>>(M, moduli, roots);

	std::vector<std::complex<double>> v(value.size());
	ranges::transform(value, std::back_inserter(v), [](double r) { return std::complex<double>(r, 0); });

	return std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
			->MakeAuxPlaintext(*cc, elementParamsPtr, v, 1, towersToDrop, slots);
}

std::vector<double> get_chebyshev_coefficients(const std::function<double(double)> &func, const double a,
											   const double b, const uint32_t degree) {
	if (!degree) {
		OPENFHE_THROW("The degree of approximation can not be zero");
	}

	const size_t coeffTotal{degree + 1};
	const double bMinusA = 0.5 * (b - a);
	const double bPlusA = 0.5 * (b + a);
	const double PiByDeg = M_PI / static_cast<double>(coeffTotal);
	std::vector<double> functionPoints(coeffTotal);
	for (size_t i = 0; i < coeffTotal; ++i)
		functionPoints[i] = func(std::cos(PiByDeg * (i + 0.5)) * bMinusA + bPlusA);

	const double multFactor = 2.0 / static_cast<double>(coeffTotal);
	std::vector<double> coefficients(coeffTotal);
	for (size_t i = 0; i < coeffTotal; ++i) {
		for (size_t j = 0; j < coeffTotal; ++j)
			coefficients[i] += functionPoints[j] * std::cos(PiByDeg * i * (j + 0.5));
		coefficients[i] *= multFactor;
	}
	return coefficients;
}

void FHEControllerGPU::create_gpu_context(bool load_on_the_fly_to_gpu, const vector<int> &devices) {
	FIDESlib::CKKS::RawParams raw = FIDESlib::CKKS::GetRawParams(context_cpu);
	const auto adapted = params.adaptTo(raw);
	for (auto &i: devices) {
		std::cout << "Added device: " << i << std::endl;
	}
	context = new FIDESlib::CKKS::Context(adapted, std::vector<int>(devices));
	context->batch = 100;
	auto raw_eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(key_pair);
	FIDESlib::CKKS::KeySwitchingKey eval_key(*context);
	eval_key.Initialize(*context, raw_eval_key);
	FIDESlib::CKKS::Context::AddEvalKey(std::move(eval_key));

	load_on_the_fly_gpu = load_on_the_fly_to_gpu;
	std::cout << "Loading on the fly: " << load_on_the_fly_gpu << std::endl;

	std::cout << "K: " << context->K << std::endl;

	/*
	convbn_initial_scale = 0.90 * scale;
	convbn_1_1_scale = 1.00 * scale;
	convbn_1_2_scale = 0.52 * scale;
	convbn_2_1_scale = 0.55 * scale;
	convbn_2_2_scale = 0.36 * scale;
	convbn_3_1_scale = 0.63 * scale;
	convbn_3_2_scale = 0.42 * scale;
	convbn_4_1_sx_scale = 0.57 * scale;
	convbn_4_1_dx_scale = 0.40 * scale;
	convbn_4_2_scale = 0.40 * scale;
	convbn_5_1_scale = 0.76 * scale;
	convbn_5_2_scale = 0.37 * scale;
	convbn_6_1_scale = 0.63 * scale;
	convbn_6_2_scale = 0.25 * scale;
	convbn_7_1_sx_scale = 0.63 * scale;
	convbn_7_1_dx_scale = 0.40 * scale;
	convbn_7_2_scale = 0.40 * scale;
	convbn_8_1_scale = 0.57 * scale;
	convbn_8_2_scale = 0.33 * scale;
	convbn_9_1_scale = 0.69 * scale;
	convbn_9_2_scale = 0.10 * scale;
	*/
}

Ctxt FHEControllerGPU::move(const CtxtCPU &c) { return {*context, FIDESlib::CKKS::GetRawCipherText(context_cpu, c)}; }

void FHEControllerGPU::move_back(Ctxt &c, const CtxtCPU &c_cpu) const {
	FIDESlib::CKKS::RawCipherText raw;
	c.dropToLevel(1);
	c.store(*context, raw);
	FIDESlib::CKKS::GetOpenFHECipherText(c_cpu, raw);
}

Ptxt FHEControllerGPU::move_ptxt(const PtxtCPU &c) {
	return {*context, FIDESlib::CKKS::GetRawPlainText(context_cpu, c)};
}

void FHEControllerGPU::generate_context(int log_ring, int log_scale, int log_primes, int digits_hks, int cts_levels,
										int stc_levels, int relu_deg, bool serialize) {

	CCParams<CryptoContextCKKSRNS> parameters;

	num_slots = 1 << 14;

	parameters.SetSecretKeyDist(SPARSE_TERNARY);
	// parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
	parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
	parameters.SetNumLargeDigits(digits_hks);
	parameters.SetRingDim(1 << log_ring);
	parameters.SetBatchSize(num_slots);

	level_budget = vector<uint32_t>();

	level_budget.push_back(cts_levels);
	level_budget.push_back(stc_levels);

	int dcrtBits = log_primes;
	int firstMod = log_scale;

	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	parameters.SetFirstModSize(firstMod);
	parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);

	uint32_t approxBootstrapDepth = 4 + 4; // During EvalRaise, Chebyshev, DoubleAngle

	uint32_t levelsUsedBeforeBootstrap = get_relu_depth(relu_deg) + 3;

	//<relu_degree> is at class-level, <relu_deg> is the input of the function
	relu_degree = relu_deg;

	write_to_file("../" + parameters_folder + "/relu_degree.txt", to_string(relu_deg));
	write_to_file("../" + parameters_folder + "/level_budget.txt",
				  to_string(level_budget[0]) + "," + to_string(level_budget[1]));

	circuit_depth = levelsUsedBeforeBootstrap +
					FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

	cout << endl
		 << "Ciphertexts depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2
		 << endl;

	parameters.SetMultiplicativeDepth(circuit_depth);

	context_cpu = GenCryptoContext(parameters);

	cout << "Context built, generating keys..." << endl;

	context_cpu->Enable(PKE);
	context_cpu->Enable(KEYSWITCH);
	context_cpu->Enable(LEVELEDSHE);
	context_cpu->Enable(ADVANCEDSHE);
	context_cpu->Enable(FHE);

	key_pair = context_cpu->KeyGen();

	context_cpu->EvalMultKeyGen(key_pair.secretKey);

	cout << "Generated." << endl;

	if (!serialize) {
		return;
	}

	cout << "Now serializing keys ..." << endl;

	ofstream multKeyFile("../" + parameters_folder + "/mult-keys.txt", ios::out | ios::binary);
	if (multKeyFile.is_open()) {
		if (!context_cpu->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
			cerr << "Error writing EvalMult keys" << std::endl;
			exit(1);
		}
		cout << "EvalMult keys have been serialized" << std::endl;
		multKeyFile.close();
	} else {
		cerr << "Error serializing EvalMult keys in \""
			 << "../" + parameters_folder + "/mult-keys.txt" << "\"" << endl;
		exit(1);
	}

	if (!Serial::SerializeToFile("../" + parameters_folder + "/crypto-context.txt", context_cpu, SerType::BINARY)) {
		cerr << "Error writing serialization of the crypto context to crypto-context.txt" << endl;
	} else {
		cout << "Crypto Context have been serialized" << std::endl;
	}

	if (!Serial::SerializeToFile("../" + parameters_folder + "/public-key.txt", key_pair.publicKey, SerType::BINARY)) {
		cerr << "Error writing serialization of public key to public-key.txt" << endl;
	} else {
		cout << "Public Key has been serialized" << std::endl;
	}

	if (!Serial::SerializeToFile("../" + parameters_folder + "/secret-key.txt", key_pair.secretKey, SerType::BINARY)) {
		cerr << "Error writing serialization of public key to secret-key.txt" << endl;
	} else {
		cout << "Secret Key has been serialized" << std::endl;
	}
}

void FHEControllerGPU::load_context(const bool verbose) {
	context_cpu->ClearEvalMultKeys();
	context_cpu->ClearEvalAutomorphismKeys();

	CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

	if (verbose)
		cout << "Reading serialized context..." << endl;

	if (!Serial::DeserializeFromFile("../" + parameters_folder + "/crypto-context.txt", context_cpu, SerType::BINARY)) {
		cerr << "I cannot read serialized data from: "
			 << "../" + parameters_folder + "/crypto-context.txt" << endl;
		exit(1);
	}

	PublicKey<DCRTPoly> clientPublicKey;
	if (!Serial::DeserializeFromFile("../" + parameters_folder + "/public-key.txt", clientPublicKey, SerType::BINARY)) {
		cerr << "I cannot read serialized data from public-key.txt" << endl;
		exit(1);
	}

	PrivateKey<DCRTPoly> serverSecretKey;
	if (!Serial::DeserializeFromFile("../" + parameters_folder + "/secret-key.txt", serverSecretKey, SerType::BINARY)) {
		cerr << "I cannot read serialized data from public-key.txt" << endl;
		exit(1);
	}

	key_pair.publicKey = clientPublicKey;
	key_pair.secretKey = serverSecretKey;

	std::ifstream multKeyIStream("../" + parameters_folder + "/mult-keys.txt", ios::in | ios::binary);
	if (!multKeyIStream.is_open()) {
		cerr << "Cannot read serialization from "
			 << "mult-keys.txt" << endl;
		exit(1);
	}
	if (!context_cpu->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
		cerr << "Could not deserialize eval mult key file" << endl;
		exit(1);
	}

	relu_degree = stoi(read_from_file("../" + parameters_folder + "/relu_degree.txt"));

	level_budget[0] = read_from_file("../" + parameters_folder + "/level_budget.txt").at(0) - '0';
	level_budget[1] = read_from_file("../" + parameters_folder + "/level_budget.txt").at(2) - '0';

	if (verbose)
		cout << "CtoS: " << level_budget[0] << ", StoC: " << level_budget[1] << endl;

	constexpr uint32_t approxBootstrapDepth = 4 + 4;
	const uint32_t levelsUsedBeforeBootstrap = get_relu_depth(relu_degree) + 4;

	circuit_depth = levelsUsedBeforeBootstrap +
					FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

	if (verbose)
		cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2
			 << endl;


	std::cout << circuit_depth << std::endl;
	num_slots = 1 << 14;
}

void FHEControllerGPU::generate_bootstrapping_keys(const vector<int> &bootstrap_slots) const {
	for (auto i: bootstrap_slots) {
		context_cpu->EvalBootstrapSetup(level_budget, {(uint32_t) dim1, (uint32_t) dim1}, i);
		context_cpu->EvalBootstrapKeyGen(key_pair.secretKey, i);
	}
}

void FHEControllerGPU::generate_rotation_keys(const vector<int> &rotations) const {
	context_cpu->EvalRotateKeyGen(key_pair.secretKey, rotations);
}

void FHEControllerGPU::gen_keys(vector<int> &rotations, const vector<int> &bootstrap_slots) const {
	generate_bootstrapping_keys(bootstrap_slots);

	std::set<int> rots(rotations.begin(), rotations.end());
	auto acc = FIDESlib::CKKS::GetAccumulateRotationIndices(4, 1, 64);
	auto acc2 = FIDESlib::CKKS::GetAccumulateRotationIndices(4, 1, 32);
	auto acc3 = FIDESlib::CKKS::GetAccumulateRotationIndices(4, 1, 16);
	auto acc4 = FIDESlib::CKKS::GetAccumulateRotationIndices(4, 64, 64);
	rots.insert(acc.begin(), acc.end());
	rots.insert(acc2.begin(), acc2.end());
	rots.insert(acc3.begin(), acc3.end());
	rots.insert(acc4.begin(), acc4.end());
	vector<int> indexes(rots.begin(), rots.end());

	rotations = indexes;
	generate_rotation_keys(indexes);
}

void FHEControllerGPU::load_keys(const vector<int> &rotations, int bootstrap_slots) {
	FIDESlib::CKKS::AddBootstrapPrecomputation(context_cpu, key_pair, bootstrap_slots, *context);
	prescale = FIDESlib::CKKS::GetPreScaleFactor(*context, this->num_slots);
	for (auto i: rotations) {
		auto raw = FIDESlib::CKKS::GetRotationKeySwitchKey(key_pair, i, context_cpu);
		FIDESlib::CKKS::KeySwitchingKey rot_ksk(*context);
		rot_ksk.Initialize(*context, raw);
		context->AddRotationKey(i, std::move(rot_ksk));
	}
}

PtxtCPU FHEControllerGPU::encode(const vector<double> &vec, const int level, int plaintext_num_slots) const {
	if (plaintext_num_slots == 0) {
		plaintext_num_slots = num_slots;
	}
	// PtxtCPU p = encodeExt(context_cpu, vec, 1, level, plaintext_num_slots);
	PtxtCPU p = context_cpu->MakeCKKSPackedPlaintext(vec, 1, level, nullptr, plaintext_num_slots);
	p->SetLength(plaintext_num_slots);

	return p;
}

PtxtCPU FHEControllerGPU::encode(const double val, const int level, int plaintext_num_slots) const {
	if (plaintext_num_slots == 0) {
		plaintext_num_slots = num_slots;
	}

	vector<double> vec;
	for (int i = 0; i < plaintext_num_slots; i++) {
		vec.push_back(val);
	}
	// PtxtCPU p = encodeExt(context_cpu, vec, 1, level, plaintext_num_slots);
	PtxtCPU p = context_cpu->MakeCKKSPackedPlaintext(vec, 1, level, nullptr, plaintext_num_slots);
	p->SetLength(plaintext_num_slots);
	return p;
}

CtxtCPU FHEControllerGPU::encrypt(const vector<double> &vec, const int level, int plaintext_num_slots) const {
	if (plaintext_num_slots == 0) {
		plaintext_num_slots = num_slots;
	}

	PtxtCPU p = encode(vec, level, plaintext_num_slots);

	return context_cpu->Encrypt(p, key_pair.publicKey);
}

PtxtCPU FHEControllerGPU::decrypt(const CtxtCPU &c) const {
	PtxtCPU p;
	context_cpu->Decrypt(key_pair.secretKey, c, &p);
	return p;
}

vector<double> FHEControllerGPU::decrypt_tovector(const CtxtCPU &c, int slots) const {
	if (slots == 0) {
		slots = num_slots;
	}

	PtxtCPU p;
	context_cpu->Decrypt(key_pair.secretKey, c, &p);
	p->SetSlots(slots);
	p->SetLength(slots);
	vector<double> vec = p->GetRealPackedValue();
	return vec;
}

void FHEControllerGPU::bootstrap(Ctxt &c, bool prescale) const { FIDESlib::CKKS::Bootstrap(c, 16384, prescale); }

void FHEControllerGPU::relu(Ctxt &c, double scale) const {
	auto coefficients = get_chebyshev_coefficients(
			[this, scale](const double x) -> double {
				if (x < 0)
					return 0;
				return (1.0 / (scale)) * x;
			},
			-1.0, 1.0, relu_degree);
	FIDESlib::CKKS::evalChebyshevSeries(c, FIDESlib::CKKS::Context::GetEvalKey(), coefficients, -1.0, 1.0);
}

void FHEControllerGPU::clear_weights_l3() {
	auto idx = std::make_tuple(7, 1);

	weights_layers.erase(idx);
	bias_layers_sx.erase(idx);

	if (!load_on_the_fly_gpu) {
		for (auto &i: weights_layers_gpu[idx]) {
			for (auto &ii: i) {
				delete ii;
			}
			i.clear();
		}
		weights_layers_gpu.erase(idx);
		delete std::get<0>(bias_layers_sx_gpu[idx]);
		delete std::get<1>(bias_layers_sx_gpu[idx]);
		bias_layers_sx_gpu.erase(idx);
	}
	weights_layers_dx.erase(idx);
	bias_layers_dx.erase(idx);
	if (!load_on_the_fly_gpu) {
		for (auto &i: weights_layers_dx_gpu[idx]) {
			delete i;
		}
		weights_layers_dx_gpu.erase(idx);
		delete std::get<0>(bias_layers_dx_gpu[idx]);
		delete std::get<1>(bias_layers_dx_gpu[idx]);
		bias_layers_dx_gpu.erase(idx);
	}

	auto idxs = {std::make_tuple(7, 2), std::make_tuple(8, 1), std::make_tuple(8, 2), std::make_tuple(9, 1),
				 std::make_tuple(9, 2)};
	for (auto id: idxs) {
		weights_layers.erase(id);
		bias_layers.erase(id);
		if (!load_on_the_fly_gpu) {
			for (auto &i: weights_layers_gpu[id]) {
				for (const auto &ii: i) {
					delete ii;
				}
				i.clear();
			}
			weights_layers_gpu.erase(id);
			delete bias_layers_gpu[id];
			bias_layers_gpu.erase(id);
		}
	}

	first_and_second_n_masks_l3.clear();
	mask_first_n_mods_l3.clear();
	mask_channel_l3.clear();
	if (!load_on_the_fly_gpu) {
		for (const auto &i: first_and_second_n_masks_l3_gpu) {
			delete i;
		}
		first_and_second_n_masks_l3_gpu.clear();
		for (const auto &i: mask_first_n_mods_l3_gpu) {
			delete i;
		}
		mask_first_n_mods_l3_gpu.clear();
		for (const auto &i: mask_channel_l3_gpu) {
			delete i;
		}
		mask_channel_l3_gpu.clear();
		delete weight_final_layer_gpu;
		delete final_layer_mask_gpu;
	}
}

void FHEControllerGPU::clear_weights_l2() {
	auto idx = std::make_tuple(4, 1);

	weights_layers.erase(idx);
	bias_layers_sx.erase(idx);

	if (!load_on_the_fly_gpu) {
		for (auto &i: weights_layers_gpu[idx]) {
			for (auto &ii: i) {
				delete ii;
			}
			i.clear();
		}
		weights_layers_gpu.erase(idx);
		delete std::get<0>(bias_layers_sx_gpu[idx]);
		delete std::get<1>(bias_layers_sx_gpu[idx]);
		bias_layers_sx_gpu.erase(idx);
	}
	weights_layers_dx.erase(idx);
	bias_layers_dx.erase(idx);
	if (!load_on_the_fly_gpu) {
		for (auto &i: weights_layers_dx_gpu[idx]) {
			delete i;
		}
		weights_layers_dx_gpu.erase(idx);
		delete std::get<0>(bias_layers_dx_gpu[idx]);
		delete std::get<1>(bias_layers_dx_gpu[idx]);
		bias_layers_dx_gpu.erase(idx);
	}

	auto idxs = {std::make_tuple(4, 2), std::make_tuple(5, 1), std::make_tuple(5, 2), std::make_tuple(6, 1),
				 std::make_tuple(6, 2)};
	for (auto id: idxs) {
		weights_layers.erase(id);
		bias_layers.erase(id);
		if (!load_on_the_fly_gpu) {
			for (auto &i: weights_layers_gpu[id]) {
				for (const auto &ii: i) {
					delete ii;
				}
				i.clear();
			}
			weights_layers_gpu.erase(id);
			delete bias_layers_gpu[id];
			bias_layers_gpu.erase(id);
		}
	}

	first_and_second_n_masks_l2.clear();
	mask_first_n_mods_l2.clear();
	mask_channel_l2.clear();
	if (!load_on_the_fly_gpu) {
		for (const auto &i: first_and_second_n_masks_l2_gpu) {
			delete i;
		}
		first_and_second_n_masks_l2_gpu.clear();
		for (const auto &i: mask_first_n_mods_l2_gpu) {
			delete i;
		}
		mask_first_n_mods_l2_gpu.clear();
		for (const auto &i: mask_channel_l2_gpu) {
			delete i;
		}
		mask_channel_l2_gpu.clear();
	}
}

void FHEControllerGPU::clear_weights_l1() {
	auto idxs = {std::make_tuple(1, 1), std::make_tuple(1, 2), std::make_tuple(2, 1),
				 std::make_tuple(2, 2), std::make_tuple(3, 1), std::make_tuple(3, 2)};
	for (auto id: idxs) {
		weights_layers.erase(id);
		bias_layers.erase(id);
		if (!load_on_the_fly_gpu) {
			for (auto &i: weights_layers_gpu[id]) {
				for (const auto &ii: i) {
					delete ii;
				}
				i.clear();
			}
			weights_layers_gpu.erase(id);
			delete bias_layers_gpu[id];
			bias_layers_gpu.erase(id);
		}
	}

	weights_conv1bn1.clear();
	if (!load_on_the_fly_gpu) {
		for (auto &i: weights_conv1bn1_gpu) {
			for (const auto &ii: i) {
				delete ii;
			}
			i.clear();
		}
		weights_conv1bn1_gpu.clear();
		delete bias_conv1bn1_gpu;
	}

	// Masks.
	if (!load_on_the_fly_gpu) {
		delete initial_layer_mask_gpu;
	}
}

void FHEControllerGPU::load_weights_l1() {

	int levels = context_cpu->GetElementParams()->GetParams().size();
	// Initial layer.
	const int convbn_initial_levels_weights = levels - 11;
	const int convbn_initial_levels_bias = levels - 9;
	const int convbn_initial_slots = 16384;
	for (auto j = 0; j < 16; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals =
					read_values_from_file("../weights/conv1bn1-ch" + to_string(j) + "-k" + to_string(k + 1) + ".bin",
										  convbn_initial_scale * prescale);
			auto encoded = encode(vals, convbn_initial_levels_weights, convbn_initial_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		weights_conv1bn1.push_back(values);
	}
	bias_conv1bn1 = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/conv1bn1-bias.bin", convbn_initial_scale * prescale),
								convbn_initial_levels_bias, convbn_initial_slots));

	if (!load_on_the_fly_gpu) {
		for (const auto &v: weights_conv1bn1) {
			std::vector<Ptxt *> v_gpu;
			for (const auto &w: v) {
				v_gpu.push_back(new Ptxt(*context, w));
			}
			weights_conv1bn1_gpu.push_back(v_gpu);
		}
		bias_conv1bn1_gpu = new Ptxt(*context, bias_conv1bn1);
	}

	// Multi-block parameters layer 1.
	const int convbn_layer_1_slots = 16384;
	const int convbn_layer1_levels_weights = levels - 3;
	const int convbn_layer1_levels_bias = levels - 2;

	// Layer 1 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_1_1;
	for (auto j = 0; j < 16; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(1) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_1_1_scale * prescale);


			auto encoded = encode(vals, convbn_layer1_levels_weights, convbn_layer_1_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_1_1.push_back(values);
	}
	weights_layers[std::make_tuple(1, 1)] = layer_1_1;
	bias_layers[std::make_tuple(1, 1)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(1) + "-conv" + to_string(1) +
															  "bn" + to_string(1) + "-bias.bin",
													  convbn_1_1_scale * prescale),
								convbn_layer1_levels_bias, convbn_layer_1_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(1, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 1 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_1_2;
	for (auto j = 0; j < 16; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(1) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_1_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer1_levels_weights, convbn_layer_1_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_1_2.push_back(values);
	}
	weights_layers[std::make_tuple(1, 2)] = layer_1_2;
	bias_layers[std::make_tuple(1, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(1) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_1_2_scale * prescale),
								convbn_layer1_levels_bias, convbn_layer_1_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(1, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 2 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_2_1;
	for (auto j = 0; j < 16; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(2) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_2_1_scale * prescale);
			auto encoded = encode(vals, convbn_layer1_levels_weights, convbn_layer_1_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_2_1.push_back(values);
	}
	weights_layers[std::make_tuple(2, 1)] = layer_2_1;
	bias_layers[std::make_tuple(2, 1)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(2) + "-conv" + to_string(1) +
															  "bn" + to_string(1) + "-bias.bin",
													  convbn_2_1_scale * prescale),
								convbn_layer1_levels_bias, convbn_layer_1_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(2, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 2 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_2_2;
	for (auto j = 0; j < 16; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(2) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_2_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer1_levels_weights, convbn_layer_1_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_2_2.push_back(values);
	}
	weights_layers[std::make_tuple(2, 2)] = layer_2_2;
	bias_layers[std::make_tuple(2, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(2) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_2_2_scale * prescale),
								convbn_layer1_levels_bias, convbn_layer_1_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(2, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 3 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_3_1;
	for (auto j = 0; j < 16; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(3) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_3_1_scale * prescale);
			auto encoded = encode(vals, convbn_layer1_levels_weights, convbn_layer_1_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_3_1.push_back(values);
	}
	weights_layers[std::make_tuple(3, 1)] = layer_3_1;
	bias_layers[std::make_tuple(3, 1)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(3) + "-conv" + to_string(1) +
															  "bn" + to_string(1) + "-bias.bin",
													  convbn_3_1_scale * prescale),
								convbn_layer1_levels_bias, convbn_layer_1_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(3, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 3 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_3_2;
	for (auto j = 0; j < 16; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(3) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_3_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer1_levels_weights, convbn_layer_1_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_3_2.push_back(values);
	}
	weights_layers[std::make_tuple(3, 2)] = layer_3_2;
	bias_layers[std::make_tuple(3, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(3) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_3_2_scale * prescale),
								convbn_layer1_levels_bias, convbn_layer_1_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(3, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Masks.
	initial_layer_mask = FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_from_to(0, 1024, 14));
	if (!load_on_the_fly_gpu) {
		initial_layer_mask_gpu = new Ptxt(*context, initial_layer_mask);
	}
}

void FHEControllerGPU::load_weights_l2() {

	int levels = context_cpu->GetElementParams()->GetParams().size();
	// Multi-block parameters layer 2.
	const int convbn_layer_2_slots = 8192;
	const int convbn_layer2_levels_weights = levels - 2;
	const int convbn_layer2_levels_bias = levels - 1;

	// Layer 4 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_4_2;
	for (auto j = 0; j < 32; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(4) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_4_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer2_levels_weights, convbn_layer_2_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_4_2.push_back(values);
	}
	weights_layers[std::make_tuple(4, 2)] = layer_4_2;
	bias_layers[std::make_tuple(4, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(4) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_4_2_scale * prescale),
								convbn_layer2_levels_bias, convbn_layer_2_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(4, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 5 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_5_1;
	for (auto j = 0; j < 32; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(5) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_5_1_scale * prescale);
			auto encoded = encode(vals, convbn_layer2_levels_weights, convbn_layer_2_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_5_1.push_back(values);
	}
	weights_layers[std::make_tuple(5, 1)] = layer_5_1;
	bias_layers[std::make_tuple(5, 1)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(5) + "-conv" + to_string(1) +
															  "bn" + to_string(1) + "-bias.bin",
													  convbn_5_1_scale * prescale),
								convbn_layer2_levels_bias, convbn_layer_2_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(5, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 5 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_5_2;
	for (auto j = 0; j < 32; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(5) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_5_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer2_levels_weights, convbn_layer_2_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_5_2.push_back(values);
	}
	weights_layers[std::make_tuple(5, 2)] = layer_5_2;
	bias_layers[std::make_tuple(5, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(5) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_5_2_scale * prescale),
								convbn_layer2_levels_bias, convbn_layer_2_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(5, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 6 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_6_1;
	for (auto j = 0; j < 32; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(6) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_6_1_scale * prescale);
			auto encoded = encode(vals, convbn_layer2_levels_weights, convbn_layer_2_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_6_1.push_back(values);
	}
	weights_layers[std::make_tuple(6, 1)] = layer_6_1;
	bias_layers[std::make_tuple(6, 1)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(6) + "-conv" + to_string(1) +
															  "bn" + to_string(1) + "-bias.bin",
													  convbn_6_1_scale * prescale),
								convbn_layer2_levels_bias, convbn_layer_2_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(6, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 6 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_6_2;
	for (auto j = 0; j < 32; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(6) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_6_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer2_levels_weights, convbn_layer_2_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_6_2.push_back(values);
	}
	weights_layers[std::make_tuple(6, 2)] = layer_6_2;
	bias_layers[std::make_tuple(6, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(6) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_6_2_scale * prescale),
								convbn_layer2_levels_bias, convbn_layer_2_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(6, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Multi-block parameters layer 2. (4-1).
	const int convbn_layer_2_x_slots = 16384;
	const int convbn_layer2_x_levels_weights = levels - 2;
	const int convbn_layer2_x_levels_bias = levels - 1;

	// Layer 4 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_4_1;
	for (auto j = 0; j < 32; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(4) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_4_1_sx_scale * prescale);
			auto encoded = encode(vals, convbn_layer2_x_levels_weights, convbn_layer_2_x_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_4_1.push_back(values);
	}
	weights_layers[std::make_tuple(4, 1)] = layer_4_1;
	auto bias1_4_1 = encode(read_values_from_file("../weights/layer" + to_string(4) + "-conv" + to_string(1) + "bn" +
														  to_string(1) + "-bias1.bin",
												  convbn_4_1_sx_scale * prescale),
							convbn_layer2_x_levels_bias, convbn_layer_2_x_slots);
	auto bias2_4_1 = encode(read_values_from_file("../weights/layer" + to_string(4) + "-conv" + to_string(1) + "bn" +
														  to_string(1) + "-bias2.bin",
												  convbn_4_1_sx_scale * prescale),
							convbn_layer2_x_levels_bias, convbn_layer_2_x_slots);
	bias_layers_sx[std::make_tuple(4, 1)] = std::make_tuple(FIDESlib::CKKS::GetRawPlainText(context_cpu, bias1_4_1),
															FIDESlib::CKKS::GetRawPlainText(context_cpu, bias2_4_1));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(4, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		Ptxt *bias_1_gpu = new Ptxt(*context, std::get<0>(bias_layers_sx[idx]));
		Ptxt *bias_2_gpu = new Ptxt(*context, std::get<1>(bias_layers_sx[idx]));
		bias_layers_sx_gpu[idx] = std::make_tuple(bias_1_gpu, bias_2_gpu);
	}

	std::vector<FIDESlib::CKKS::RawPlainText> layer_4_1_dx;
	for (auto j = 0; j < 32; ++j) {
		auto vals = read_values_from_file("../weights/layer" + to_string(4) + "dx-conv" + to_string(1) + "bn" +
												  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(1) + ".bin",
										  convbn_4_1_dx_scale * prescale);
		auto encoded = encode(vals, convbn_layer2_x_levels_weights, convbn_layer_2_x_slots);
		layer_4_1_dx.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
	}
	weights_layers_dx[std::make_tuple(4, 1)] = layer_4_1_dx;
	auto bias1_4_1_dx = encode(read_values_from_file("../weights/layer" + to_string(4) + "dx-conv" + to_string(1) +
															 "bn" + to_string(1) + "-bias1.bin",
													 convbn_4_1_dx_scale * prescale),
							   convbn_layer2_x_levels_bias, convbn_layer_2_x_slots);
	auto bias2_4_1_dx = encode(read_values_from_file("../weights/layer" + to_string(4) + "dx-conv" + to_string(1) +
															 "bn" + to_string(1) + "-bias2.bin",
													 convbn_4_1_dx_scale * prescale),
							   convbn_layer2_x_levels_bias, convbn_layer_2_x_slots);
	bias_layers_dx[std::make_tuple(4, 1)] = std::make_tuple(FIDESlib::CKKS::GetRawPlainText(context_cpu, bias1_4_1_dx),
															FIDESlib::CKKS::GetRawPlainText(context_cpu, bias2_4_1_dx));

	if (!load_on_the_fly_gpu) {
		std::vector<Ptxt *> ws_gpu;
		auto idx = std::make_tuple(4, 1);
		for (auto w: weights_layers_dx[idx]) {
			ws_gpu.push_back(new Ptxt(*context, w));
		}
		weights_layers_dx_gpu[idx] = ws_gpu;
		auto bias_1_gpu = new Ptxt(*context, std::get<0>(bias_layers_dx[idx]));
		auto bias_2_gpu = new Ptxt(*context, std::get<1>(bias_layers_dx[idx]));
		bias_layers_dx_gpu[idx] = std::make_tuple(bias_1_gpu, bias_2_gpu);
	}

	// Masks layer 2.
	auto mask_first = FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_first_n(16384, levels - 7));
	auto mask_second = FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_second_n(16384, levels - 7));
	first_and_second_n_masks_l2 = {mask_first, mask_second};

	if (!load_on_the_fly_gpu) {
		first_and_second_n_masks_l2_gpu = {new Ptxt(*context, first_and_second_n_masks_l2[0]),
										   new Ptxt(*context, first_and_second_n_masks_l2[1])};
	}

	auto full_pack_2 = FIDESlib::CKKS::GetRawPlainText(context_cpu, gen_mask(2, levels - 6));
	auto full_pack_4 = FIDESlib::CKKS::GetRawPlainText(context_cpu, gen_mask(4, levels - 5));
	auto full_pack_8 = FIDESlib::CKKS::GetRawPlainText(context_cpu, gen_mask(8, levels - 4));
	full_pack_mask = {full_pack_2, full_pack_4, full_pack_8};

	if (!load_on_the_fly_gpu) {
		auto fp_2_gpu = new Ptxt(*context, full_pack_mask[0]);
		auto fp_4_gpu = new Ptxt(*context, full_pack_mask[1]);
		auto fp_8_gpu = new Ptxt(*context, full_pack_mask[2]);
		full_pack_mask_gpu = {fp_2_gpu, fp_4_gpu, fp_8_gpu};
	}

	zero_ctxt = FIDESlib::CKKS::GetRawCipherText(context_cpu, encrypt({0}));

	if (!load_on_the_fly_gpu) {
		zero_ctxt_gpu = new Ctxt(*context, zero_ctxt);
	}

	for (auto i = 0; i < 16; ++i) {
		mask_first_n_mods_l2.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_first_n_mod(16, 1024, i, 20)));
	}

	if (!load_on_the_fly_gpu) {
		for (const auto &m: mask_first_n_mods_l2) {
			mask_first_n_mods_l2_gpu.push_back(new Ptxt(*context, m));
		}
	}

	for (auto i = 0; i < 32; ++i) {
		mask_channel_l2.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_channel(i, 21)));
	}

	if (!load_on_the_fly_gpu) {
		for (const auto &m: mask_channel_l2) {
			mask_channel_l2_gpu.push_back(new Ptxt(*context, m));
		}
	}
}

void FHEControllerGPU::load_weights_l3() {

	int levels = context_cpu->GetElementParams()->GetParams().size();
	// Multi-block parameters layer 3. (7-1).
	const int convbn_layer_3_x_slots = 8192;
	const int convbn_layer3_x_levels_weights = levels - 2;
	const int convbn_layer3_x_levels_bias = levels - 1;

	// Layer 7 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_7_1;
	for (auto j = 0; j < 64; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(7) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_7_1_sx_scale * prescale);
			auto encoded = encode(vals, convbn_layer3_x_levels_weights, convbn_layer_3_x_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_7_1.push_back(values);
	}
	weights_layers[std::make_tuple(7, 1)] = layer_7_1;
	auto bias1_7_1 = encode(read_values_from_file("../weights/layer" + to_string(7) + "-conv" + to_string(1) + "bn" +
														  to_string(1) + "-bias1.bin",
												  convbn_7_1_sx_scale * prescale),
							convbn_layer3_x_levels_bias, convbn_layer_3_x_slots);
	auto bias2_7_1 = encode(read_values_from_file("../weights/layer" + to_string(7) + "-conv" + to_string(1) + "bn" +
														  to_string(1) + "-bias2.bin",
												  convbn_7_1_sx_scale * prescale),
							convbn_layer3_x_levels_bias, convbn_layer_3_x_slots);
	bias_layers_sx[std::make_tuple(7, 1)] = std::make_tuple(FIDESlib::CKKS::GetRawPlainText(context_cpu, bias1_7_1),
															FIDESlib::CKKS::GetRawPlainText(context_cpu, bias2_7_1));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(7, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		Ptxt *bias_1_gpu = new Ptxt(*context, std::get<0>(bias_layers_sx[idx]));
		Ptxt *bias_2_gpu = new Ptxt(*context, std::get<1>(bias_layers_sx[idx]));
		bias_layers_sx_gpu[idx] = std::make_tuple(bias_1_gpu, bias_2_gpu);
	}

	std::vector<FIDESlib::CKKS::RawPlainText> layer_7_1_dx;
	for (auto j = 0; j < 64; ++j) {
		auto vals = read_values_from_file("../weights/layer" + to_string(7) + "dx-conv" + to_string(1) + "bn" +
												  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(1) + ".bin",
										  convbn_7_1_dx_scale * prescale);
		auto encoded = encode(vals, convbn_layer3_x_levels_weights, convbn_layer_3_x_slots);
		layer_7_1_dx.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
	}
	weights_layers_dx[std::make_tuple(7, 1)] = layer_7_1_dx;
	auto bias1_7_1_dx = encode(read_values_from_file("../weights/layer" + to_string(7) + "dx-conv" + to_string(1) +
															 "bn" + to_string(1) + "-bias1.bin",
													 convbn_7_1_dx_scale * prescale),
							   convbn_layer3_x_levels_bias, convbn_layer_3_x_slots);
	auto bias2_7_1_dx = encode(read_values_from_file("../weights/layer" + to_string(7) + "dx-conv" + to_string(1) +
															 "bn" + to_string(1) + "-bias2.bin",
													 convbn_7_1_dx_scale * prescale),
							   convbn_layer3_x_levels_bias, convbn_layer_3_x_slots);
	bias_layers_dx[std::make_tuple(7, 1)] = std::make_tuple(FIDESlib::CKKS::GetRawPlainText(context_cpu, bias1_7_1_dx),
															FIDESlib::CKKS::GetRawPlainText(context_cpu, bias2_7_1_dx));


	if (!load_on_the_fly_gpu) {
		std::vector<Ptxt *> ws_gpu;
		auto idx = std::make_tuple(7, 1);
		for (const auto &w: weights_layers_dx[idx]) {
			ws_gpu.push_back(new Ptxt(*context, w));
		}
		weights_layers_dx_gpu[idx] = ws_gpu;
		auto bias_1_gpu = new Ptxt(*context, std::get<0>(bias_layers_dx[idx]));
		auto bias_2_gpu = new Ptxt(*context, std::get<1>(bias_layers_dx[idx]));
		bias_layers_dx_gpu[idx] = std::make_tuple(bias_1_gpu, bias_2_gpu);
	}

	// Multi-block parameters layer 2.
	const int convbn_layer_3_slots = 4096;
	const int convbn_layer3_levels_weights = levels - 2;
	const int convbn_layer3_levels_bias = levels - 1;

	// Layer 7 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_7_2;
	for (auto j = 0; j < 64; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(7) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_7_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer3_levels_weights, convbn_layer_3_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_7_2.push_back(values);
	}
	weights_layers[std::make_tuple(7, 2)] = layer_7_2;
	bias_layers[std::make_tuple(7, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(7) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_7_2_scale * prescale),
								convbn_layer3_levels_bias, convbn_layer_3_slots));


	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(7, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 8 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_8_1;
	for (auto j = 0; j < 64; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(8) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_8_1_scale * prescale);
			auto encoded = encode(vals, convbn_layer3_levels_weights, convbn_layer_3_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_8_1.push_back(values);
	}
	weights_layers[std::make_tuple(8, 1)] = layer_8_1;
	bias_layers[std::make_tuple(8, 1)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(8) + "-conv" + to_string(1) +
															  "bn" + to_string(1) + "-bias.bin",
													  convbn_8_1_scale * prescale),
								convbn_layer3_levels_bias, convbn_layer_3_slots));


	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(8, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 8 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_8_2;
	for (auto j = 0; j < 64; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(8) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_8_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer3_levels_weights, convbn_layer_3_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_8_2.push_back(values);
	}
	weights_layers[std::make_tuple(8, 2)] = layer_8_2;
	bias_layers[std::make_tuple(8, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(8) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_8_2_scale * prescale),
								convbn_layer3_levels_bias, convbn_layer_3_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(8, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 9 - 1.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_9_1;
	for (auto j = 0; j < 64; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(9) + "-conv" + to_string(1) + "bn" +
													  to_string(1) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_9_1_scale * prescale);
			auto encoded = encode(vals, convbn_layer3_levels_weights, convbn_layer_3_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_9_1.push_back(values);
	}
	weights_layers[std::make_tuple(9, 1)] = layer_9_1;
	bias_layers[std::make_tuple(9, 1)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(9) + "-conv" + to_string(1) +
															  "bn" + to_string(1) + "-bias.bin",
													  convbn_9_1_scale * prescale),
								convbn_layer3_levels_bias, convbn_layer_3_slots));


	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(9, 1);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Layer 9 - 2.
	std::vector<std::vector<FIDESlib::CKKS::RawPlainText>> layer_9_2;
	for (auto j = 0; j < 64; ++j) {
		std::vector<FIDESlib::CKKS::RawPlainText> values;
		for (auto k = 0; k < 9; ++k) {
			auto vals = read_values_from_file("../weights/layer" + to_string(9) + "-conv" + to_string(2) + "bn" +
													  to_string(2) + "-ch" + to_string(j) + "-k" + to_string(k + 1) +
													  ".bin",
											  convbn_9_2_scale * prescale);
			auto encoded = encode(vals, convbn_layer3_levels_weights, convbn_layer_3_slots);
			values.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, encoded));
		}
		layer_9_2.push_back(values);
	}
	weights_layers[std::make_tuple(9, 2)] = layer_9_2;
	bias_layers[std::make_tuple(9, 2)] = FIDESlib::CKKS::GetRawPlainText(
			context_cpu, encode(read_values_from_file("../weights/layer" + to_string(9) + "-conv" + to_string(2) +
															  "bn" + to_string(2) + "-bias.bin",
													  convbn_9_2_scale * prescale),
								convbn_layer3_levels_bias, convbn_layer_3_slots));

	if (!load_on_the_fly_gpu) {
		std::vector<std::vector<Ptxt *>> gpu_w_layer;
		auto idx = std::make_tuple(9, 2);
		for (const auto &ws_raw: weights_layers[idx]) {
			std::vector<Ptxt *> gpu_w_row;
			for (const auto &w_raw: ws_raw) {
				gpu_w_row.push_back(new Ptxt(*context, w_raw));
			}
			gpu_w_layer.push_back(gpu_w_row);
		}
		weights_layers_gpu[idx] = gpu_w_layer;
		bias_layers_gpu[idx] = new Ptxt(*context, bias_layers[idx]);
	}

	// Masks layer 3.
	auto mask_first_l3 = FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_first_n(8192, levels - 7));
	auto mask_second_l3 = FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_second_n(8192, levels - 7));
	first_and_second_n_masks_l3 = {mask_first_l3, mask_second_l3};

	if (!load_on_the_fly_gpu) {
		first_and_second_n_masks_l3_gpu = {new Ptxt(*context, mask_first_l3), new Ptxt(*context, mask_second_l3)};
	}

	for (auto i = 0; i < 32; ++i) {
		mask_first_n_mods_l3.push_back(
				FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_first_n_mod2(8, 256, i, levels - 5)));
	}

	if (!load_on_the_fly_gpu) {
		for (const auto &w: mask_first_n_mods_l3) {
			mask_first_n_mods_l3_gpu.push_back(new Ptxt(*context, w));
		}
	}

	for (auto i = 0; i < 64; ++i) {
		mask_channel_l3.push_back(FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_channel_2(i, levels - 4)));
	}

	if (!load_on_the_fly_gpu) {
		for (const auto &w: mask_channel_l3) {
			mask_channel_l3_gpu.push_back(new Ptxt(*context, w));
		}
	}

	// Data and mask final layer.
	weight_final_layer =
			FIDESlib::CKKS::GetRawPlainText(context_cpu, encode(read_fc_weight("../weights/fc.bin"), 14, 4096));
	final_layer_mask = FIDESlib::CKKS::GetRawPlainText(context_cpu, mask_mod(64, 14, 1.0 / 64.0));

	if (!load_on_the_fly_gpu) {
		weight_final_layer_gpu = new Ptxt(*context, weight_final_layer);
		final_layer_mask_gpu = new Ptxt(*context, final_layer_mask);
	}
}

void FHEControllerGPU::load_weights() {
	load_weights_l1();
	load_weights_l2();
	load_weights_l3();
}

Ctxt FHEControllerGPU::convbn_initial(const Ctxt &in) {

	/// Hoisted rotations of the input.
	int img_width = 32;
	int padding = 1;
	array<Ctxt, 8> aux = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};
	vector<Ctxt *> c_rotations;
	for (auto i = 0; i < 8; ++i) {
		c_rotations.push_back(&aux[i]);
	}
	std::vector<int> indexes;
	indexes.push_back(-padding - img_width);
	indexes.push_back(-img_width);
	indexes.push_back(padding - img_width);
	indexes.push_back(-padding);
	indexes.push_back(padding);
	indexes.push_back(-padding + img_width);
	indexes.push_back(img_width);
	indexes.push_back(padding + img_width);
	std::vector<FIDESlib::CKKS::KeySwitchingKey *> keys;
	for (auto i: indexes) {
		keys.push_back(&(context->GetRotationKey(i)));
	}
	Ctxt tmp(*context);
	tmp.copy(in);
	tmp.rotate_hoisted(keys, indexes, c_rotations, false);

	// K_rows array.
	array<Ctxt, 9> k_rows = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};

	// Row computation data.
	Ptxt *encodedGPU = nullptr;
	// Final sum storing.
	Ctxt final_sum(*context);

	for (int j = 0; j < 16; j++) {
		if (load_on_the_fly_gpu) {
			delete encodedGPU;
			encodedGPU = new Ptxt(*context);
		}
		// Compute rows.
		for (int k = 0; k < 4; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_conv1bn1_gpu[j][k];
			} else {
				encodedGPU->load(weights_conv1bn1[j][k]);
			}
			k_rows[k].copy(*c_rotations[k]);
			k_rows[k].multPt(*encodedGPU);
		}
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_conv1bn1_gpu[j][4];
		} else {
			encodedGPU->load(weights_conv1bn1[j][4]);
		}
		k_rows[4].copy(in);
		k_rows[4].multPt(*encodedGPU);
		for (int k = 5; k < 9; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_conv1bn1_gpu[j][k];
			} else {
				encodedGPU->load(weights_conv1bn1[j][k]);
			}
			k_rows[k].copy(*c_rotations[k - 1]);
			k_rows[k].multPt(*encodedGPU);
		}

		// Sum rows on binary tree manner.
		k_rows[0].add(k_rows[1]);
		k_rows[2].add(k_rows[3]);
		k_rows[4].add(k_rows[5]);
		k_rows[6].add(k_rows[7]);
		k_rows[0].add(k_rows[2]);
		k_rows[4].add(k_rows[6]);
		k_rows[0].add(k_rows[4]);
		k_rows[0].add(k_rows[8]);

		// Result for this iteration.
		Ctxt res(*context);
		res.copy(k_rows[0]);
		k_rows[0].rotate(1024, context->GetRotationKey(1024));
		res.add(k_rows[0]);
		k_rows[0].rotate(1024, context->GetRotationKey(1024));
		res.add(k_rows[0]);
		if (!load_on_the_fly_gpu) {
			encodedGPU = initial_layer_mask_gpu;
		} else {
			delete encodedGPU;
			encodedGPU = new Ptxt(*context, initial_layer_mask);
		}
		res.multPt(*encodedGPU);
		if (j == 0) {
			final_sum.copy(res);
			final_sum.rotate(1024, context->GetRotationKey(1024));
		} else {
			final_sum.add(res);
			final_sum.rotate(1024, context->GetRotationKey(1024));
		}
	}

	/// Bias!
	if (final_sum.NoiseLevel == 2)
		final_sum.rescale();
	if (!load_on_the_fly_gpu) {
		encodedGPU = bias_conv1bn1_gpu;
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, bias_conv1bn1);
	}
	final_sum.addPt(*encodedGPU);

	if (load_on_the_fly_gpu)
		delete encodedGPU;

	return final_sum;
}

Ctxt FHEControllerGPU::convbn(const Ctxt &in, int layer, int n) {


	/// Hoisted rotations of the input.
	int img_width = 32;
	int padding = 1;
	array<Ctxt, 8> aux = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};
	vector<Ctxt *> c_rotations;
	for (auto i = 0; i < 8; ++i) {
		c_rotations.push_back(&aux[i]);
	}
	std::vector<int> indexes;
	indexes.push_back(-padding - img_width);
	indexes.push_back(-img_width);
	indexes.push_back(padding - img_width);
	indexes.push_back(-padding);
	indexes.push_back(padding);
	indexes.push_back(-padding + img_width);
	indexes.push_back(img_width);
	indexes.push_back(padding + img_width);
	std::vector<FIDESlib::CKKS::KeySwitchingKey *> keys;
	for (auto i: indexes) {
		keys.push_back(&context->GetRotationKey(i));
	}
	Ctxt tmp(*context);
	tmp.copy(in);
	tmp.dropToLevel(1);
	tmp.rotate_hoisted(keys, indexes, c_rotations, false);

	// Rows array.
	array<Ctxt, 9> k_rows = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};

	// Final sum.
	Ctxt final_sum(*context);

	// Row computation data.
	Ptxt *encodedGPU;
	if (load_on_the_fly_gpu)
		encodedGPU = new Ptxt(*context);

	auto idx = std::make_tuple(layer, n);

	for (int j = 0; j < 16; j++) {

		// Row computation.
		for (int k = 0; k < 4; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows[k].copy(*c_rotations[k]);
			k_rows[k].multPt(*encodedGPU);
		}
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_gpu[idx][j][4];
		} else {
			encodedGPU->load(weights_layers[idx][j][4]);
		}
		k_rows[4].copy(in);
		k_rows[4].multPt(*encodedGPU);
		for (int k = 5; k < 9; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows[k].copy(*c_rotations[k - 1]);
			k_rows[k].multPt(*encodedGPU);
		}

		// Sum rows on binary tree manner.
		k_rows[0].add(k_rows[1]);
		k_rows[2].add(k_rows[3]);
		k_rows[4].add(k_rows[5]);
		k_rows[6].add(k_rows[7]);
		k_rows[0].add(k_rows[2]);
		k_rows[4].add(k_rows[6]);
		k_rows[0].add(k_rows[4]);
		k_rows[0].add(k_rows[8]);

		if (j == 0) {
			final_sum.copy(k_rows[0]);
			final_sum.rotate(-1024, context->GetRotationKey(-1024));
		} else {
			final_sum.add(k_rows[0]);
			final_sum.rotate(-1024, context->GetRotationKey(-1024));
		}
	}

	if (!load_on_the_fly_gpu) {
		encodedGPU = bias_layers_gpu[idx];
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, bias_layers[idx]);
	}
	if (final_sum.NoiseLevel == 2)
		final_sum.rescale();
	final_sum.addPt(*encodedGPU);

	// Bias!

	if (load_on_the_fly_gpu)
		delete encodedGPU;

	return final_sum;
}

Ctxt FHEControllerGPU::convbn2(const Ctxt &in, int layer, int n) {

	auto start = start_time();

	/// Hoisted rotations of the input.
	int img_width = 16;
	int padding = 1;

	array<Ctxt, 8> aux = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};
	vector<Ctxt *> c_rotations;
	for (auto i = 0; i < 8; ++i) {
		c_rotations.push_back(&aux[i]);
	}
	std::vector<int> indexes;
	indexes.push_back(-padding - img_width);
	indexes.push_back(-img_width);
	indexes.push_back(padding - img_width);
	indexes.push_back(-padding);
	indexes.push_back(padding);
	indexes.push_back(-padding + img_width);
	indexes.push_back(img_width);
	indexes.push_back(padding + img_width);
	std::vector<FIDESlib::CKKS::KeySwitchingKey *> keys;
	for (auto i: indexes) {
		keys.push_back(&context->GetRotationKey(i));
	}
	Ctxt tmp(*context);
	tmp.copy(in);
	tmp.rotate_hoisted(keys, indexes, c_rotations, false);

	// Rows array.
	array<Ctxt, 9> k_rows = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};

	// Final sum.
	Ctxt final_sum(*context);

	// Row computation data.
	Ptxt *encodedGPU;
	if (load_on_the_fly_gpu)
		encodedGPU = new Ptxt(*context);

	auto idx = std::make_tuple(layer, n);

	for (int j = 0; j < 32; j++) {

		for (int k = 0; k < 4; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows[k].copy(*c_rotations[k]);
			k_rows[k].multPt(*encodedGPU);
		}
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_gpu[idx][j][4];
		} else {
			encodedGPU->load(weights_layers[idx][j][4]);
		}
		k_rows[4].copy(in);
		k_rows[4].multPt(*encodedGPU);
		for (int k = 5; k < 9; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows[k].copy(*c_rotations[k - 1]);
			k_rows[k].multPt(*encodedGPU);
		}

		// Sum rows on binary tree manner.
		k_rows[0].add(k_rows[1]);
		k_rows[2].add(k_rows[3]);
		k_rows[4].add(k_rows[5]);
		k_rows[6].add(k_rows[7]);
		k_rows[0].add(k_rows[2]);
		k_rows[4].add(k_rows[6]);
		k_rows[0].add(k_rows[4]);
		k_rows[0].add(k_rows[8]);

		if (j == 0) {
			final_sum.copy(k_rows[0]);
			final_sum.rotate(-256, context->GetRotationKey(-256));
		} else {
			final_sum.add(k_rows[0]);
			final_sum.rotate(-256, context->GetRotationKey(-256));
		}
	}

	// Bias!.
	if (final_sum.NoiseLevel == 2)
		final_sum.rescale();
	if (!load_on_the_fly_gpu) {
		encodedGPU = bias_layers_gpu[idx];
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, bias_layers[idx]);
	}
	final_sum.addPt(*encodedGPU);

	if (load_on_the_fly_gpu)
		delete encodedGPU;

	return final_sum;
}

Ctxt FHEControllerGPU::convbn3(const Ctxt &in, int layer, int n) {

	/// Hoisted rotations of the input.
	int img_width = 8;
	int padding = 1;
	array<Ctxt, 8> aux = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};
	vector<Ctxt *> c_rotations;
	for (auto i = 0; i < 8; ++i) {
		c_rotations.push_back(&aux[i]);
	}
	std::vector<int> indexes;
	indexes.push_back(-padding - img_width);
	indexes.push_back(-img_width);
	indexes.push_back(padding - img_width);
	indexes.push_back(-padding);
	indexes.push_back(padding);
	indexes.push_back(-padding + img_width);
	indexes.push_back(img_width);
	indexes.push_back(padding + img_width);
	std::vector<FIDESlib::CKKS::KeySwitchingKey *> keys;
	for (auto i: indexes) {
		keys.push_back(&context->GetRotationKey(i));
	}
	Ctxt tmp(*context);
	tmp.copy(in);
	tmp.rotate_hoisted(keys, indexes, c_rotations, false);

	// Rows array.
	array<Ctxt, 9> k_rows = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};

	// Final sum.
	Ctxt final_sum(*context);

	// Row computation data.
	Ptxt *encodedGPU;
	if (load_on_the_fly_gpu)
		encodedGPU = new Ptxt(*context);

	auto idx = std::make_tuple(layer, n);

	for (int j = 0; j < 64; j++) {

		for (int k = 0; k < 4; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows[k].copy(*c_rotations[k]);
			k_rows[k].multPt(*encodedGPU);
		}
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_gpu[idx][j][4];
		} else {
			encodedGPU->load(weights_layers[idx][j][4]);
		}
		k_rows[4].copy(in);
		k_rows[4].multPt(*encodedGPU);
		for (int k = 5; k < 9; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows[k].copy(*c_rotations[k - 1]);
			k_rows[k].multPt(*encodedGPU);
		}

		// Sum rows on binary tree manner.
		k_rows[0].add(k_rows[1]);
		k_rows[2].add(k_rows[3]);
		k_rows[4].add(k_rows[5]);
		k_rows[6].add(k_rows[7]);
		k_rows[0].add(k_rows[2]);
		k_rows[4].add(k_rows[6]);
		k_rows[0].add(k_rows[4]);
		k_rows[0].add(k_rows[8]);

		if (j == 0) {
			final_sum.copy(k_rows[0]);
			final_sum.rotate(-64, context->GetRotationKey(-64));
		} else {
			final_sum.add(k_rows[0]);
			final_sum.rotate(-64, context->GetRotationKey(-64));
		}
	}

	// Bias!
	if (final_sum.NoiseLevel == 2)
		final_sum.rescale();
	if (!load_on_the_fly_gpu) {
		encodedGPU = bias_layers_gpu[idx];
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, bias_layers[idx]);
	}
	final_sum.addPt(*encodedGPU);

	if (load_on_the_fly_gpu) {
		delete encodedGPU;
	}

	return final_sum;
}

array<Ctxt, 2> FHEControllerGPU::convbn1632sx(const Ctxt &in, int layer, int n) {

	/// Hoisted rotations of the input.
	int img_width = 32;
	int padding = 1;
	array<Ctxt, 8> aux = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};
	vector<Ctxt *> c_rotations;
	for (auto i = 0; i < 8; ++i) {
		c_rotations.push_back(&aux[i]);
	}
	std::vector<int> indexes;
	indexes.push_back(-padding - img_width);
	indexes.push_back(-img_width);
	indexes.push_back(padding - img_width);
	indexes.push_back(-padding);
	indexes.push_back(padding);
	indexes.push_back(-padding + img_width);
	indexes.push_back(img_width);
	indexes.push_back(padding + img_width);
	std::vector<FIDESlib::CKKS::KeySwitchingKey *> keys;
	for (auto i: indexes) {
		keys.push_back(&context->GetRotationKey(i));
	}
	Ctxt tmp(*context);
	tmp.copy(in);
	tmp.rotate_hoisted(keys, indexes, c_rotations, false);

	// Final sum.
	Ctxt final_sum_16(*context);
	Ctxt final_sum_1632(*context);

	// Row computation data.
	Ptxt *encodedGPU;
	if (load_on_the_fly_gpu)
		encodedGPU = new Ptxt(*context);

	// Row data.
	array<Ctxt, 9> k_rows_16 = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};

	array<Ctxt, 9> k_rows_1632 = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};

	auto idx = std::make_tuple(layer, n);

	for (int j = 0; j < 16; j++) {

		for (int k = 0; k < 4; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows_16[k].copy(*c_rotations[k]);
			k_rows_16[k].multPt(*encodedGPU);
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j + 16][k];
			} else {
				encodedGPU->load(weights_layers[idx][j + 16][k]);
			}
			k_rows_1632[k].copy(*c_rotations[k]);
			k_rows_1632[k].multPt(*encodedGPU);
		}
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_gpu[idx][j][4];
		} else {
			encodedGPU->load(weights_layers[idx][j][4]);
		}
		k_rows_16[4].copy(in);
		k_rows_16[4].multPt(*encodedGPU);
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_gpu[idx][j + 16][4];
		} else {
			encodedGPU->load(weights_layers[idx][j + 16][4]);
		}
		k_rows_1632[4].copy(in);
		k_rows_1632[4].multPt(*encodedGPU);
		for (int k = 5; k < 9; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows_16[k].copy(*c_rotations[k - 1]);
			k_rows_16[k].multPt(*encodedGPU);
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j + 16][k];
			} else {
				encodedGPU->load(weights_layers[idx][j + 16][k]);
			}
			k_rows_1632[k].copy(*c_rotations[k - 1]);
			k_rows_1632[k].multPt(*encodedGPU);
		}

		// Sum rows on binary tree manner.
		k_rows_16[0].add(k_rows_16[1]);
		k_rows_16[2].add(k_rows_16[3]);
		k_rows_16[4].add(k_rows_16[5]);
		k_rows_16[6].add(k_rows_16[7]);
		k_rows_16[0].add(k_rows_16[2]);
		k_rows_16[4].add(k_rows_16[6]);
		k_rows_16[0].add(k_rows_16[4]);
		k_rows_16[0].add(k_rows_16[8]);
		k_rows_1632[0].add(k_rows_1632[1]);
		k_rows_1632[2].add(k_rows_1632[3]);
		k_rows_1632[4].add(k_rows_1632[5]);
		k_rows_1632[6].add(k_rows_1632[7]);
		k_rows_1632[0].add(k_rows_1632[2]);
		k_rows_1632[4].add(k_rows_1632[6]);
		k_rows_1632[0].add(k_rows_1632[4]);
		k_rows_1632[0].add(k_rows_1632[8]);

		if (j == 0) {
			final_sum_16.copy(k_rows_16[0]);
			final_sum_16.rotate(-1024, context->GetRotationKey(-1024));
			final_sum_1632.copy(k_rows_1632[0]);
			final_sum_1632.rotate(-1024, context->GetRotationKey(-1024));
		} else {
			final_sum_16.add(k_rows_16[0]);
			final_sum_16.rotate(-1024, context->GetRotationKey(-1024));
			final_sum_1632.add(k_rows_1632[0]);
			final_sum_1632.rotate(-1024, context->GetRotationKey(-1024));
		}
	}

	// Bias!
	if (final_sum_16.NoiseLevel == 2)
		final_sum_16.rescale();
	if (final_sum_1632.NoiseLevel == 2)
		final_sum_1632.rescale();
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<0>(bias_layers_sx_gpu[idx]); // TODO scale
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, std::get<0>(bias_layers_sx[idx]));
	}
	final_sum_16.addPt(*encodedGPU);
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<1>(bias_layers_sx_gpu[idx]);
	} else {
		encodedGPU->load(std::get<1>(bias_layers_sx[idx]));
	}
	final_sum_1632.addPt(*encodedGPU);

	if (load_on_the_fly_gpu) {
		delete encodedGPU;
	}

	array<Ctxt, 2> res{Ctxt(std::move(final_sum_16)), Ctxt(std::move(final_sum_1632))};

	return res;
}

array<Ctxt, 2> FHEControllerGPU::convbn1632dx(const Ctxt &in, int layer, int n) {

	// Results.
	Ctxt finalSum016(*context);
	Ctxt finalSum1632(*context);
	Ctxt k_row_16(*context);
	Ctxt k_row_1632(*context);

	// Data computation needs.
	Ptxt *encodedGPU;
	if (load_on_the_fly_gpu) {
		encodedGPU = new Ptxt(*context);
	}

	auto idx = std::make_tuple(layer, n);

	for (int j = 0; j < 16; j++) {

		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_dx_gpu[idx][j];
		} else {
			encodedGPU->load(weights_layers_dx[idx][j]);
		}
		k_row_16.copy(in);
		k_row_16.multPt(*encodedGPU);
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_dx_gpu[idx][j + 16];
		} else {
			encodedGPU->load(weights_layers_dx[idx][j + 16]);
		}
		k_row_1632.copy(in);
		k_row_1632.multPt(*encodedGPU);

		if (j == 0) {
			finalSum016.copy(k_row_16);
			finalSum016.rotate(-1024, context->GetRotationKey(-1024));
			finalSum1632.copy(k_row_1632);
			finalSum1632.rotate(-1024, context->GetRotationKey(-1024));
		} else {
			finalSum016.add(k_row_16);
			finalSum016.rotate(-1024, context->GetRotationKey(-1024));
			finalSum1632.add(k_row_1632);
			finalSum1632.rotate(-1024, context->GetRotationKey(-1024));
		}
	}

	// Bias!
	if (finalSum016.NoiseLevel == 2)
		finalSum016.rescale();
	if (finalSum1632.NoiseLevel == 2)
		finalSum1632.rescale();
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<0>(bias_layers_dx_gpu[idx]);
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, std::get<0>(bias_layers_dx[idx]));
	}
	finalSum016.addPt(*encodedGPU);
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<1>(bias_layers_dx_gpu[idx]);
	} else {
		encodedGPU->load(std::get<1>(bias_layers_dx[idx]));
	}
	finalSum1632.addPt(*encodedGPU);

	if (load_on_the_fly_gpu)
		delete encodedGPU;

	array<Ctxt, 2> res{Ctxt(std::move(finalSum016)), Ctxt(std::move(finalSum1632))};

	return res;
}

array<Ctxt, 2> FHEControllerGPU::convbn3264sx(const Ctxt &in, int layer, int n) {

	/// Hoisted rotations of the input.
	int img_width = 16;
	int padding = 1;
	array<Ctxt, 8> aux = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};
	vector<Ctxt *> c_rotations;
	for (auto i = 0; i < 8; ++i) {
		c_rotations.push_back(&aux[i]);
	}
	std::vector<int> indexes;
	indexes.push_back(-padding - img_width);
	indexes.push_back(-img_width);
	indexes.push_back(padding - img_width);
	indexes.push_back(-padding);
	indexes.push_back(padding);
	indexes.push_back(-padding + img_width);
	indexes.push_back(img_width);
	indexes.push_back(padding + img_width);
	std::vector<FIDESlib::CKKS::KeySwitchingKey *> keys;
	for (auto i: indexes) {
		keys.push_back(&context->GetRotationKey(i));
	}
	Ctxt tmp(*context);
	tmp.copy(in);
	tmp.rotate_hoisted(keys, indexes, c_rotations, false);

	// Final results.
	Ctxt final_sum_32(*context);
	Ctxt final_sum_3264(*context);

	// Arrays of rows.
	array<Ctxt, 9> k_rows_32 = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};
	array<Ctxt, 9> k_rows_3264 = {
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
			Ctxt(*context), Ctxt(*context), Ctxt(*context), Ctxt(*context),
	};

	// Tmp values for computation.
	Ptxt *encodedGPU;
	if (load_on_the_fly_gpu) {
		encodedGPU = new Ptxt(*context);
	}

	auto idx = std::make_tuple(layer, n);

	for (int j = 0; j < 32; j++) {

		for (int k = 0; k < 4; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows_32[k].copy(*c_rotations[k]);
			k_rows_32[k].multPt(*encodedGPU);
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j + 32][k];
			} else {
				encodedGPU->load(weights_layers[idx][j + 32][k]);
			}
			k_rows_3264[k].copy(*c_rotations[k]);
			k_rows_3264[k].multPt(*encodedGPU);
		}
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_gpu[idx][j][4];
		} else {
			encodedGPU->load(weights_layers[idx][j][4]);
		}
		k_rows_32[4].copy(in);
		k_rows_32[4].multPt(*encodedGPU);
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_gpu[idx][j + 32][4];
		} else {
			encodedGPU->load(weights_layers[idx][j + 32][4]);
		}
		k_rows_3264[4].copy(in);
		k_rows_3264[4].multPt(*encodedGPU);
		for (int k = 5; k < 9; k++) {
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j][k];
			} else {
				encodedGPU->load(weights_layers[idx][j][k]);
			}
			k_rows_32[k].copy(*c_rotations[k - 1]);
			k_rows_32[k].multPt(*encodedGPU);
			if (!load_on_the_fly_gpu) {
				encodedGPU = weights_layers_gpu[idx][j + 32][k];
			} else {
				encodedGPU->load(weights_layers[idx][j + 32][k]);
			}
			k_rows_3264[k].copy(*c_rotations[k - 1]);
			k_rows_3264[k].multPt(*encodedGPU);
		}

		// Sum rows on binary tree manner.
		k_rows_32[0].add(k_rows_32[1]);
		k_rows_32[2].add(k_rows_32[3]);
		k_rows_32[4].add(k_rows_32[5]);
		k_rows_32[6].add(k_rows_32[7]);
		k_rows_32[0].add(k_rows_32[2]);
		k_rows_32[4].add(k_rows_32[6]);
		k_rows_32[0].add(k_rows_32[4]);
		k_rows_32[0].add(k_rows_32[8]);
		k_rows_3264[0].add(k_rows_3264[1]);
		k_rows_3264[2].add(k_rows_3264[3]);
		k_rows_3264[4].add(k_rows_3264[5]);
		k_rows_3264[6].add(k_rows_3264[7]);
		k_rows_3264[0].add(k_rows_3264[2]);
		k_rows_3264[4].add(k_rows_3264[6]);
		k_rows_3264[0].add(k_rows_3264[4]);
		k_rows_3264[0].add(k_rows_3264[8]);

		if (j == 0) {
			final_sum_32.copy(k_rows_32[0]);
			final_sum_32.rotate(-256, context->GetRotationKey(-256));
			final_sum_3264.copy(k_rows_3264[0]);
			final_sum_3264.rotate(-256, context->GetRotationKey(-256));
		} else {
			final_sum_32.add(k_rows_32[0]);
			final_sum_32.rotate(-256, context->GetRotationKey(-256));
			final_sum_3264.add(k_rows_3264[0]);
			final_sum_3264.rotate(-256, context->GetRotationKey(-256));
		}
	}

	// Bias!
	if (final_sum_32.NoiseLevel == 2)
		final_sum_32.rescale();
	if (final_sum_3264.NoiseLevel == 2)
		final_sum_3264.rescale();
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<0>(bias_layers_sx_gpu[idx]);
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, std::get<0>(bias_layers_sx[idx]));
	}
	final_sum_32.addPt(*encodedGPU);
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<1>(bias_layers_sx_gpu[idx]);
	} else {
		encodedGPU->load(std::get<1>(bias_layers_sx[idx]));
	}
	final_sum_3264.addPt(*encodedGPU);

	if (load_on_the_fly_gpu) {
		delete encodedGPU;
	}

	array<Ctxt, 2> res{Ctxt(std::move(final_sum_32)), Ctxt(std::move(final_sum_3264))};

	return res;
}

array<Ctxt, 2> FHEControllerGPU::convbn3264dx(const Ctxt &in, int layer, int n) {

	// Results.
	Ctxt finalSum032(*context);
	Ctxt finalSum3264(*context);
	Ctxt k_rows_32(*context);
	Ctxt k_rows_3264(*context);

	// Data computation needs.
	Ptxt *encodedGPU;
	if (load_on_the_fly_gpu) {
		encodedGPU = new Ptxt(*context);
	}

	auto idx = std::make_tuple(layer, n);

	for (int j = 0; j < 32; j++) {
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_dx_gpu[idx][j];
		} else {
			encodedGPU->load(weights_layers_dx[idx][j]);
		}
		k_rows_32.copy(in);
		k_rows_32.multPt(*encodedGPU);
		if (!load_on_the_fly_gpu) {
			encodedGPU = weights_layers_dx_gpu[idx][j + 32];
		} else {
			encodedGPU->load(weights_layers_dx[idx][j + 32]);
		}
		k_rows_3264.copy(in);
		k_rows_3264.multPt(*encodedGPU);

		if (j == 0) {
			finalSum032.copy(k_rows_32);
			finalSum032.rotate(-256, context->GetRotationKey(-256));
			finalSum3264.copy(k_rows_3264);
			finalSum3264.rotate(-256, context->GetRotationKey(-256));
		} else {
			finalSum032.add(k_rows_32);
			finalSum032.rotate(-256, context->GetRotationKey(-256));
			finalSum3264.add(k_rows_3264);
			finalSum3264.rotate(-256, context->GetRotationKey(-256));
		}
	}

	// Bias!
	if (finalSum032.NoiseLevel == 2)
		finalSum032.rescale();
	if (finalSum3264.NoiseLevel == 2)
		finalSum3264.rescale();
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<0>(bias_layers_dx_gpu[idx]);
	} else {
		delete encodedGPU;
		encodedGPU = new Ptxt(*context, std::get<0>(bias_layers_dx[idx]));
	}
	finalSum032.addPt(*encodedGPU);
	if (!load_on_the_fly_gpu) {
		encodedGPU = std::get<1>(bias_layers_dx_gpu[idx]);
	} else {
		encodedGPU->load(std::get<1>(bias_layers_dx[idx]));
	}
	finalSum3264.addPt(*encodedGPU);

	if (load_on_the_fly_gpu)
		delete encodedGPU;

	array<Ctxt, 2> res{Ctxt(std::move(finalSum032)), Ctxt(std::move(finalSum3264))};

	return res;
}

Ctxt FHEControllerGPU::downsample1024to256(const Ctxt &c1, const Ctxt &c2) {

	num_slots = 16384 * 2;

	Ptxt *aux;

	// Pack in a single ciphertext.
	if (load_on_the_fly_gpu) {
		aux = new Ptxt(*context, first_and_second_n_masks_l2[0]);
	} else {
		aux = first_and_second_n_masks_l2_gpu[0];
	}
	Ctxt full_pack(*context);
	full_pack.copy(c1);
	full_pack.multPt(*aux);
	if (load_on_the_fly_gpu) {
		aux->load(first_and_second_n_masks_l2[1]);
	} else {
		aux = first_and_second_n_masks_l2_gpu[1];
	}
	Ctxt masked(*context);
	masked.copy(c2);
	masked.multPt(*aux);
	full_pack.add(masked);

	if (load_on_the_fly_gpu) {
		aux->load(full_pack_mask[0]);
	} else {
		aux = full_pack_mask_gpu[0];
	}
	Ctxt rot(*context);
	rot.copy(full_pack);
	rot.rotate(1, context->GetRotationKey(1));
	full_pack.add(rot);
	full_pack.multPt(*aux);

	if (load_on_the_fly_gpu) {
		delete aux;
		aux = new Ptxt(*context, full_pack_mask[1]);
	} else {
		aux = full_pack_mask_gpu[1];
	}
	rot.copy(full_pack);
	rot.rotate(2, context->GetRotationKey(2));
	full_pack.add(rot);
	full_pack.multPt(*aux);

	if (load_on_the_fly_gpu) {
		delete aux;
		aux = new Ptxt(*context, full_pack_mask[2]);
	} else {
		aux = full_pack_mask_gpu[2];
	}
	rot.copy(full_pack);
	rot.rotate(4, context->GetRotationKey(4));
	full_pack.add(rot);
	full_pack.multPt(*aux);

	rot.copy(full_pack);
	rot.rotate(8, context->GetRotationKey(8));
	full_pack.add(rot);

	if (load_on_the_fly_gpu) {
		delete aux;
		aux = new Ptxt(*context);
	}
	Ctxt down_sampled_rows(*context);
	if (load_on_the_fly_gpu) {
		down_sampled_rows.load(zero_ctxt);
	} else {
		down_sampled_rows.copy(*zero_ctxt_gpu);
	}
	for (int i = 0; i < 16; i++) {
		if (load_on_the_fly_gpu) {
			aux->load(mask_first_n_mods_l2[i]);
		} else {
			aux = mask_first_n_mods_l2_gpu[i];
		}
		masked.copy(full_pack);
		masked.multPt(*aux);
		down_sampled_rows.add(masked);
		if (i < 15) {
			full_pack.rotate(64 - 16, context->GetRotationKey(64 - 16));
		}
	}

	down_sampled_rows.rescale();

	if (load_on_the_fly_gpu) {
		delete aux;
		aux = new Ptxt(*context);
	}
	Ctxt down_sampled_channels(*context);
	if (load_on_the_fly_gpu) {
		down_sampled_channels.load(zero_ctxt);
	} else {
		down_sampled_channels.copy(*zero_ctxt_gpu);
	}
	for (int i = 0; i < 32; i++) {
		if (load_on_the_fly_gpu) {
			aux->load(mask_channel_l2[i]);
		} else {
			aux = mask_channel_l2_gpu[i]; // TODO prescale
		}
		masked.copy(down_sampled_rows);
		masked.multPt(*aux);
		down_sampled_channels.add(masked);
		down_sampled_channels.rotate(-(1024 - 256), context->GetRotationKey(-(1024 - 256)));
	}

	if (load_on_the_fly_gpu)
		delete aux;

	down_sampled_channels.rotate((1024 - 256) * 32, context->GetRotationKey((1024 - 256) * 32));
	masked.copy(down_sampled_channels);
	masked.rotate(-8192, context->GetRotationKey(-8192));
	down_sampled_channels.add(masked);
	masked.copy(down_sampled_channels);
	masked.rotate(-16384, context->GetRotationKey(-16384));
	down_sampled_channels.add(masked);

	return down_sampled_channels;
}

Ctxt FHEControllerGPU::downsample256to64(const Ctxt &c1, const Ctxt &c2) {
	num_slots = 8192 * 2;

	Ptxt *aux;
	if (load_on_the_fly_gpu) {
		aux = new Ptxt(*context, first_and_second_n_masks_l3[0]);
	} else {
		aux = first_and_second_n_masks_l3_gpu[0];
	}

	Ctxt full_pack(*context);
	full_pack.copy(c1);
	Ctxt masked(*context);
	masked.copy(c2);
	full_pack.multPt(*aux);
	if (load_on_the_fly_gpu) {
		aux->load(first_and_second_n_masks_l3[1]);
	} else {
		aux = first_and_second_n_masks_l3_gpu[1];
	}
	masked.multPt(*aux);
	full_pack.add(masked);

	if (load_on_the_fly_gpu) {
		aux->load(full_pack_mask[0]);
	} else {
		aux = full_pack_mask_gpu[0];
	}
	Ctxt rot(*context);
	rot.copy(full_pack);
	rot.rotate(1, context->GetRotationKey(1));
	full_pack.add(rot);
	full_pack.multPt(*aux);

	if (load_on_the_fly_gpu) {
		delete aux;
		aux = new Ptxt(*context, full_pack_mask[1]);
	} else {
		aux = full_pack_mask_gpu[1];
	}
	rot.copy(full_pack);
	rot.rotate(2, context->GetRotationKey(2));
	full_pack.add(rot);
	full_pack.multPt(*aux);

	rot.copy(full_pack);
	rot.rotate(4, context->GetRotationKey(4));
	full_pack.add(rot);

	Ctxt down_sampled_rows(*context);
	if (load_on_the_fly_gpu) {
		down_sampled_rows.load(zero_ctxt);
		delete aux;
		aux = new Ptxt(*context);
	} else {
		down_sampled_rows.copy(*zero_ctxt_gpu);
	}

	for (int i = 0; i < 32; i++) {
		if (load_on_the_fly_gpu) {
			aux->load(mask_first_n_mods_l3[i]);
		} else {
			aux = mask_first_n_mods_l3_gpu[i];
		}
		masked.copy(full_pack);
		masked.multPt(*aux);
		down_sampled_rows.add(masked);
		if (i < 31) {
			full_pack.rotate(32 - 8, context->GetRotationKey(32 - 8));
		}
	}

	Ctxt down_sampled_channels(*context);
	if (load_on_the_fly_gpu) {
		down_sampled_channels.load(zero_ctxt);
		delete aux;
		aux = new Ptxt(*context);
	} else {
		down_sampled_channels.copy(*zero_ctxt_gpu);
	}
	for (int i = 0; i < 64; i++) {
		if (load_on_the_fly_gpu) {
			aux->load(mask_channel_l3[i]);
		} else {
			aux = mask_channel_l3_gpu[i]; // TODO prescale
		}
		masked.copy(down_sampled_rows);
		masked.multPt(*aux);
		down_sampled_channels.add(masked);
		down_sampled_channels.rotate(-(256 - 64), context->GetRotationKey(-(256 - 64)));
	}
	if (load_on_the_fly_gpu)
		delete aux;

	down_sampled_channels.rotate((256 - 64) * 64, context->GetRotationKey((256 - 64) * 64));
	masked.copy(down_sampled_channels);
	masked.rotate(-4096, context->GetRotationKey(-4096));
	down_sampled_channels.add(masked);
	masked.copy(down_sampled_channels);
	masked.rotate(-4096 - 4096, context->GetRotationKey(-4096 - 4096));
	down_sampled_channels.add(masked);

	return down_sampled_channels;
}

void FHEControllerGPU::rotsum(Ctxt &in, const int slots) {
	Ctxt result(*context);
	Ctxt tmp(*context);
	if (1) {
		FIDESlib::CKKS::Accumulate(in, 4, 1, slots);
	} else {
		result.copy(in);
		for (int i = 0; i < log2(slots); i++) {
			tmp.copy(result);
			auto rot_idx = pow(2, i);
			tmp.rotate(rot_idx, context->GetRotationKey(rot_idx));
			result.add(tmp);
		}
		in.copy(result);
	}
}

void FHEControllerGPU::rotsum_padded(Ctxt &in, const int slots) {
	Ctxt result(*context);
	Ctxt tmp(*context);
	result.copy(in);

	if (1) {
		FIDESlib::CKKS::Accumulate(in, 4, slots, slots);
	} else {
		for (int i = 0; i < log2(slots); i++) {
			tmp.copy(result);
			auto rot_idx = slots * pow(2, i);
			tmp.rotate(rot_idx, context->GetRotationKey(rot_idx));
			result.add(tmp);
		}
		in.copy(result);
	}
}

void FHEControllerGPU::repeat(Ctxt &in, const int slots) {
	rotsum(in, slots);
	in.rotate(-slots + 1, context->GetRotationKey(-slots + 1));
}

PtxtCPU FHEControllerGPU::gen_mask(const int n, const int level) {
	vector<double> mask;

	int copy_interval = n;

	for (int i = 0; i < num_slots; i++) {
		if (copy_interval > 0) {
			mask.push_back(1);
		} else {
			mask.push_back(0);
		}

		copy_interval--;

		if (copy_interval <= -n) {
			copy_interval = n;
		}
	}

	return encode(mask, level, num_slots);
}

PtxtCPU FHEControllerGPU::mask_first_n(const int n, const int level) {
	vector<double> mask;

	for (int i = 0; i < num_slots; i++) {
		if (i < n) {
			mask.push_back(1);
		} else {
			mask.push_back(0);
		}
	}

	return encode(mask, level, num_slots);
}

PtxtCPU FHEControllerGPU::mask_second_n(const int n, const int level) {
	vector<double> mask;

	for (int i = 0; i < num_slots; i++) {
		if (i >= n) {
			mask.push_back(1);
		} else {
			mask.push_back(0);
		}
	}

	return encode(mask, level, num_slots);
}

PtxtCPU FHEControllerGPU::mask_first_n_mod(const int n, const int padding, const int pos, const int level) {
	vector<double> mask;
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < (pos * n); j++) {
			mask.push_back(0);
		}
		for (int j = 0; j < n; j++) {
			mask.push_back(1);
		}
		for (int j = 0; j < (padding - n - (pos * n)); j++) {
			mask.push_back(0);
		}
	}

	return encode(mask, level, 16384 * 2);
}

PtxtCPU FHEControllerGPU::mask_first_n_mod2(const int n, const int padding, const int pos, const int level) {
	vector<double> mask;
	for (int i = 0; i < 64; i++) {
		for (int j = 0; j < (pos * n); j++) {
			mask.push_back(0);
		}
		for (int j = 0; j < n; j++) {
			mask.push_back(1);
		}
		for (int j = 0; j < (padding - n - (pos * n)); j++) {
			mask.push_back(0);
		}
	}

	return encode(mask, level, 8192 * 2);
}

PtxtCPU FHEControllerGPU::mask_channel(const int n, const int level) {
	vector<double> mask;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 1024; j++) {
			mask.push_back(0);
		}
	}

	for (int i = 0; i < 256; i++) {
		mask.push_back(1);
	}

	for (int i = 0; i < 1024 - 256; i++) {
		mask.push_back(0);
	}

	for (int i = 0; i < 31 - n; i++) {
		for (int j = 0; j < 1024; j++) {
			mask.push_back(0);
		}
	}

	return encode(mask, level, 16384 * 2);
}

PtxtCPU FHEControllerGPU::mask_channel_2(const int n, const int level) {
	vector<double> mask;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 256; j++) {
			mask.push_back(0);
		}
	}

	for (int i = 0; i < 64; i++) {
		mask.push_back(prescale);
	}

	for (int i = 0; i < 256 - 64; i++) {
		mask.push_back(0);
	}

	for (int i = 0; i < 63 - n; i++) {
		for (int j = 0; j < 256; j++) {
			mask.push_back(0);
		}
	}

	return encode(mask, level, 8192 * 2);
}

PtxtCPU FHEControllerGPU::mask_mod(const int n, const int level, const double custom_val) {
	vector<double> vec;

	for (int i = 0; i < num_slots; i++) {
		if (i % n == 0) {
			vec.push_back(custom_val);
		} else {
			vec.push_back(0);
		}
	}

	return encode(vec, level, num_slots);
}

PtxtCPU FHEControllerGPU::mask_from_to(const int from, const int to, const int level) {
	vector<double> vec;

	for (int i = 0; i < num_slots; i++) {
		if (i >= from && i < to) {
			vec.push_back(1);
		} else {
			vec.push_back(0);
		}
	}

	return encode(vec, level, num_slots);
}

void FHEControllerGPU::bootstrap_precision(const Ctxt &c) {
	cout << "Computing boostrap precision..." << endl;

	Ctxt tmp(*context);
	tmp.copy(c);

	CtxtCPU c_cpu = encrypt({0}, circuit_depth - c.getLevel(), num_slots);
	move_back(tmp, c_cpu);
	FIDESlib::CKKS::RawCipherText raw_c;

	tmp.store(*context, raw_c);
	FIDESlib::CKKS::GetOpenFHECipherText(c_cpu, raw_c);

	PtxtCPU a = decrypt(c_cpu);

	bootstrap(tmp, false);
	tmp.store(*context, raw_c);
	FIDESlib::CKKS::GetOpenFHECipherText(c_cpu, raw_c);

	PtxtCPU b = decrypt(c_cpu);

	cout << "Precision: " << to_string(utils::compute_approx_error(a, b)) << endl;
}
