#ifndef LOWMEMORYFHERESNET20_UTILS_H
#define LOWMEMORYFHERESNET20_UTILS_H

#include <iostream>
#include <openfhe.h>

using namespace std;
using namespace std::chrono;
using namespace lbcrypto;

#define YELLOW_TEXT "\033[1;33m"
#define RESET_COLOR "\033[0m"

namespace utils {

	static inline chrono::time_point<steady_clock, nanoseconds> start_time() { return steady_clock::now(); }

	static inline string get_class(int max_index) {
		switch (max_index) {
			case 0:
				return "Airplane";
			case 1:
				return "Automobile";
			case 2:
				return "Bird";
			case 3:
				return "Cat";
			case 4:
				return "Deer";
			case 5:
				return "Dog";
			case 6:
				return "Frog";
			case 7:
				return "Horse";
			case 8:
				return "Ship";
			case 9:
				return "Truck";
		}

		return "?";
	}

	static inline void print_duration(chrono::time_point<steady_clock, nanoseconds> &start,
									  chrono::time_point<steady_clock, nanoseconds> &end, string &title) {
		auto ms = duration_cast<milliseconds>(end - start);
		cout << "⌛(" << title << "): " << ms.count() << endl;
	}

	static inline vector<double> read_values_from_file(const string &filename, double scale = 1) {
		vector<double> values;

		// TODO FIX, JUST TO RUN ON SERVER
		return std::vector<double>(8, 0.0);

		ifstream file(filename);

		if (!file.is_open()) {
			std::cerr << "Can not open " << filename << std::endl;
			return values; // Restituisce un vettore vuoto in caso di errore
		}

		string row;
		while (std::getline(file, row)) {
			istringstream stream(row);
			string value;
			while (std::getline(stream, value, ',')) {
				try {
					double num = stod(value);
					values.push_back(num * scale);
				} catch (const invalid_argument &e) {
					cerr << "Can not convert: " << value << endl;
					values.push_back(0.0);
				}
			}
		}

		file.close();


		return values;
	}

	static inline vector<double> read_fc_weight(const string &filename) {
		vector<double> weight = read_values_from_file("../weights/fc.bin");
		vector<double> weight_corrected;

		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 10; j++) {
				weight_corrected.push_back(weight[(10 * i) + j]);
			}
			for (int j = 0; j < 64 - 10; j++) {
				weight_corrected.push_back(0);
			}
		}

		return weight_corrected;
	}

	static inline double compute_approx_error(Plaintext expected, Plaintext bootstrapped) {
		vector<complex<double>> result;
		vector<complex<double>> expectedResult;

		result = bootstrapped->GetCKKSPackedValue();
		expectedResult = expected->GetCKKSPackedValue();


		if (result.size() != expectedResult.size())
			OPENFHE_THROW(config_error, "Cannot compare vectors with different numbers of elements");

		double maxError = 0;
		for (size_t i = 0; i < result.size(); ++i) {
			double error = std::abs(result[i].real() - expectedResult[i].real());
			if (maxError < error)
				maxError = error;
		}

		return std::abs(std::log2(maxError));
	}

	static inline int get_relu_depth(int degree) {
		switch (degree) {
			case 5:
				return 3;
			case 13:
				return 4;
			case 27:
				return 5;
			case 59:
				return 6;
			case 119:
				return 7;
			case 200:
			case 247:
				return 8;
			case 495:
				return 9;
			case 1007:
				return 10;
			case 2031:
				return 11;
		}

		cerr << "Set a valid degree for ReLU" << endl;
		exit(1);
	}

	static inline void write_to_file(string filename, string content) {
		ofstream file;
		file.open(filename);
		file << content.c_str();
		file.close();
	}

	static inline string read_from_file(string filename) {
		// It reads only the first line!!
		string line;
		ifstream myfile(filename);
		if (myfile.is_open()) {
			if (getline(myfile, line)) {
				myfile.close();
				return line;
			} else {
				cerr << "Could not open " << filename << "." << endl;
				exit(1);
			}
		} else {
			cerr << "Could not open " << filename << "." << endl;
			exit(1);
		}
	}
} // namespace utils

#endif
