#ifndef CRYPT_HPP
#define CRYPT_HPP

#include <vector>

#include <openfhe.h>

/**
 * Encrypt a matrix of data.
 * @param data Data to be encrypted.
 * @param pk Public encryption key.
 * @return Equivalent vector of ciphertext that encrypts the matrix by row.
 */
std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> encrypt_data(const std::vector<std::vector<double>> &data,
																   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk);

/**
 * Encrypt a vector of data.
 * @param data Vector to be encrypted.
 * @param pk Public encryption key.
 * @return Ciphertext with the encrypted vector.
 */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypt_data(const std::vector<double> &data,
													  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk,
													  const int scaleDegree, const int level);

/**
 * Decrypt the given ciphertext.
 * @param ct Ciphertext to be decrypted.
 * @param sk Secret key.
 * @return Vector of decrypted data.
 */
std::vector<double> decrypt_data(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct,
								 const lbcrypto::PrivateKey<lbcrypto::DCRTPoly> &sk);

/**
 * Decrypt a matrix of data.
 * @param ct Data to be decrypted.
 * @param sk Secret key.
 * @return Equivalent vector of ciphertext that decrypts the matrix by row.
 */
std::vector<std::vector<double>> decrypt_data(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &ct,
											  const lbcrypto::PrivateKey<lbcrypto::DCRTPoly> &sk);

#endif // CRYPT_HPP
