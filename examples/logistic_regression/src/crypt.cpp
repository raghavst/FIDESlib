#include "crypt.hpp"
#include <execution>
#include "fhe.hpp"

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypt_data(const std::vector<std::vector<double>> &data,
																   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk) {
	std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypted_data(data.size());

	std::map<size_t, std::vector<double>> data_map_plain;

	for (size_t i = 0; i < data.size(); i++) {
		data_map_plain.insert(std::make_pair(i, data[i]));
	}

	std::for_each_n(std::execution::par_unseq, data_map_plain.begin(), data_map_plain.size(), [&](const auto &v) {
		const lbcrypto::Plaintext plaintext = cc_cpu->MakeCKKSPackedPlaintext(v.second);
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext = cc_cpu->Encrypt(pk, plaintext);
		encrypted_data[v.first] = ciphertext;
	});

	return encrypted_data;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypt_data(const std::vector<double> &data,
													  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
													  const int scaleDegree, const int level) {
	const lbcrypto::Plaintext plaintext = cc_cpu->MakeCKKSPackedPlaintext(data, scaleDegree, level);
	const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext = cc_cpu->Encrypt(pk, plaintext);
	return ciphertext;
}

std::vector<double> decrypt_data(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct,
								 const lbcrypto::PrivateKey<lbcrypto::DCRTPoly> &sk) {
	std::vector<double> data;
	data.reserve(num_slots);
	lbcrypto::Plaintext pt;
	cc_cpu->Decrypt(ct, sk, &pt);
	// std::cout << pt << std::endl;
	for (const auto vec = pt->GetCKKSPackedValue(); const auto &v: vec) {
		data.push_back(v.real());
	}
	return data;
}

std::vector<std::vector<double>> decrypt_data(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &ct,
											  const lbcrypto::PrivateKey<lbcrypto::DCRTPoly> &sk) {
	std::vector<std::vector<double>> data;

	data.reserve(ct.size());
	for (const auto &v: ct) {
		data.emplace_back(decrypt_data(v, sk));
	}

	return data;
}
