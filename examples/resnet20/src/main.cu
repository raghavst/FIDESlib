#include <iostream>
#include <sys/stat.h>

#include "FHEControllerGPU.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../include/CKKS/Bootstrap.cuh"
#include "stb_image.hpp"

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"
#define RESET_COLOR "\033[0m"

void check_arguments(int argc, char *argv[]);
vector<double> read_image(const char *filename);

void executeResNet20();

Ctxt initial_layer(Ctxt &&in);
Ctxt layer1(Ctxt &&in);
Ctxt layer2(Ctxt &&in);
Ctxt layer3(Ctxt &&in);
Ctxt final_layer(Ctxt &&in);

FHEControllerGPU controller;

int dim1 = 4;
int generate_context;
string input_filename;
int verbose = true;
int logringdim = 16;
bool load_on_the_fly_disk = false;
bool load_on_the_fly_gpu = false;

std::vector<int> devices{0};

int main(int argc, char *argv[]) {

	check_arguments(argc, argv);

	if (generate_context == -1) {
		cerr << "You either have to use void clear_weights();the argument \"generate_keys\" or \"load_keys\"!\nIf it "
				"is your first time, you could try "
				"with \"./LowMemoryFHEResNet20 generate_keys 1\"\nCheck the README.md.\nAborting. :-("
			 << endl;
		exit(1);
	}

	if (generate_context > 0) {
		switch (generate_context) {
			case 1:
				controller.generate_context(logringdim, 52, 48, 2, 3, 3, /*27*/ 27, true);
				break;
			default:
				std::cerr << "Use a defined experiment" << std::endl;
				exit(1);
		}

	} else {
		controller.load_context(verbose > 1);
	}

	controller.gen_keys(controller.rots_total, {16384});
	{
		char *res = getenv("FIDESLIB_USE_LOAD_ON_THE_FLY");
		if (res && !(0 == std::strcmp(res, ""))) {
			int num_dev = atoi(res);
			load_on_the_fly_gpu = num_dev;

			std::cout << "loadontheflygpu: " << num_dev << std::endl;
		}
	}
	std::cout << "loadontheflygpu: " << load_on_the_fly_gpu << std::endl;
	controller.create_gpu_context(load_on_the_fly_gpu, devices);
	executeResNet20();
}


void executeResNet20() {
	if (verbose >= 0)
		cout << "Encrypted ResNet20 classification started." << endl;

	if (input_filename.empty()) {
		input_filename = "../inputs/luis.png";
		if (verbose >= 0)
			cout << "You did not set any input, I use " << GREEN_TEXT << "../inputs/luis.png" << RESET_COLOR << "."
				 << endl;
	} else {
		if (verbose >= 0)
			cout << "I am going to encrypt and classify " << GREEN_TEXT << input_filename << RESET_COLOR << "." << endl;
	}

	vector<double> input_image = read_image(input_filename.c_str());

	CtxtCPU in_cpu =
			controller.encrypt(input_image, controller.circuit_depth - 4 - get_relu_depth(controller.relu_degree));
	Ctxt in = controller.move(in_cpu);

	controller.load_keys(controller.rots_total, 16384);

	if (verbose > 0) {
		CtxtCPU tmp = controller.encrypt(input_image, controller.circuit_depth - 2);
		Ctxt tmp_gpu = controller.move(tmp);
		controller.bootstrap_precision(tmp_gpu);
	}

	controller.load_weights();

	cudaDeviceSynchronize();
	auto start = start_time();

	cudaDeviceSynchronize();
	auto first_layer = start_time();
	Ctxt firstLayer = initial_layer(std::move(in));
	cudaDeviceSynchronize();
	auto layer_1 = start_time();
	Ctxt resLayer1 = layer1(std::move(firstLayer));
	cudaDeviceSynchronize();
	auto load_2 = start_time();

	cudaDeviceSynchronize();
	auto layer_2 = start_time();
	Ctxt resLayer2 = layer2(std::move(resLayer1));
	cudaDeviceSynchronize();
	auto load_3 = start_time();
	cudaDeviceSynchronize();
	auto layer_3 = start_time();
	Ctxt resLayer3 = layer3(std::move(resLayer2));
	cudaDeviceSynchronize();
	auto finallayer = start_time();
	Ctxt finalRes = final_layer(std::move(resLayer3));
	cudaDeviceSynchronize();
	auto end = start_time();

	auto str = std::string("Total inference time");
	print_duration(start, end, str);
	str = "Load L1 time";
	print_duration(start, first_layer, str);
	str = "Initial layer time";
	print_duration(first_layer, layer_1, str);
	str = "Layer 1 time";
	print_duration(layer_1, load_2, str);
	str = "Load L2 time";
	print_duration(load_2, layer_2, str);
	str = "Layer 2 time";
	print_duration(layer_2, load_3, str);
	str = "Load L3 time";
	print_duration(load_3, layer_3, str);
	str = "Layer 3 time";
	print_duration(layer_3, finallayer, str);
	str = "Final layer time";
	print_duration(finallayer, end, str);

	CtxtCPU res_cpu = controller.encrypt({0}, controller.circuit_depth - finalRes.getLevel(), controller.num_slots);
	controller.move_back(finalRes, res_cpu);
	/*
		vector<double> clear_result = controller.decrypt_tovector(res_cpu, 10);

		// Index of the max element.
		const auto max_element_iterator = ranges::max_element(clear_result);
		const int index_max = distance(clear_result.begin(), max_element_iterator);

		if (verbose >= 0) {
			cout << "The input image is classified as " << YELLOW_TEXT << utils::get_class(index_max) << RESET_COLOR <<
	   ""
				 << endl;
			cout << "The index of max element is " << YELLOW_TEXT << index_max << RESET_COLOR << "" << endl;
		}
		*/
}
Ctxt initial_layer(Ctxt &&in) {

	if (in.NoiseLevel == 2)
		in.rescale();
	in.dropToLevel(2);

	Ctxt res = controller.convbn_initial(in);

	FIDESlib::CKKS::Bootstrap(res, controller.num_slots, true);

	controller.relu(res, controller.convbn_initial_scale);

	return res;
}

Ctxt final_layer(Ctxt &&in) {

	controller.num_slots = 4096;

	const Ptxt weight(*controller.context, controller.weight_final_layer);

	controller.rotsum(in, 64);

	const Ptxt mask(*controller.context, controller.final_layer_mask);
	in.multPt(mask);


	controller.repeat(in, 16);

	in.multPt(weight);

	controller.rotsum_padded(in, 64);


	return in;
}

Ctxt layer3(Ctxt &&in) {

	// controller.bootstrap(in, true);
	if (in.NoiseLevel == 2)
		in.rescale();
	in.dropToLevel(1);
	array<Ctxt, 2> res1sx = controller.convbn3264sx(in, 7, 1);
	array<Ctxt, 2> res1dx = controller.convbn3264dx(in, 7, 1);
	controller.bootstrap(res1sx[0], true);
	controller.bootstrap(res1sx[1], true);
	controller.bootstrap(res1dx[0], true);
	controller.bootstrap(res1dx[1], true);
	res1sx[0].dropToLevel(6);
	res1sx[1].dropToLevel(6);
	res1dx[0].dropToLevel(6);
	res1dx[1].dropToLevel(6);
	Ctxt full_pack_Sx = controller.downsample256to64(res1sx[0], res1sx[1]);
	Ctxt full_pack_Dx = controller.downsample256to64(res1dx[0], res1dx[1]);
	controller.num_slots = 4096;
	controller.bootstrap(full_pack_Sx, true);
	controller.relu(full_pack_Sx, controller.convbn_7_1_sx_scale);
	full_pack_Sx.copy(controller.convbn3(full_pack_Sx, 7, 2));
	full_pack_Sx.add(full_pack_Dx);
	controller.bootstrap(full_pack_Sx, true);
	controller.relu(full_pack_Sx, controller.convbn_7_2_scale);

	Ctxt res2 = controller.convbn3(full_pack_Sx, 8, 1);
	controller.bootstrap(res2, true);
	controller.relu(res2, controller.convbn_8_1_scale);
	res2.copy(controller.convbn3(res2, 8, 2));
	full_pack_Sx.multScalar(controller.convbn_8_2_scale);
	res2.add(full_pack_Sx);
	controller.bootstrap(res2, true);
	controller.relu(res2, controller.convbn_8_2_scale);

	Ctxt res3 = controller.convbn3(res2, 9, 1);
	controller.bootstrap(res3, true);
	controller.relu(res3, controller.convbn_9_1_scale);
	res3.copy(controller.convbn3(res3, 9, 2));
	res2.multScalar(controller.convbn_9_2_scale * controller.prescale);
	res3.add(res2);
	controller.bootstrap(res3, true);
	controller.relu(res3, controller.convbn_9_2_scale * controller.prescale);
	if (res3.NoiseLevel == 2)
		res3.rescale();
	res3.dropToLevel(0);
	controller.bootstrap(res3, true);
	return res3;
}

Ctxt layer2(Ctxt &&in) {

	// controller.bootstrap(in, true);

	if (in.NoiseLevel == 2)
		in.rescale();
	in.dropToLevel(1);
	array<Ctxt, 2> res1sx = controller.convbn1632sx(in, 4, 1);
	array<Ctxt, 2> res1dx = controller.convbn1632dx(in, 4, 1);

	controller.bootstrap(res1sx[0], true);
	controller.bootstrap(res1sx[1], true);
	controller.bootstrap(res1dx[0], true);
	controller.bootstrap(res1dx[1], true);
	res1sx[0].dropToLevel(7);
	res1sx[1].dropToLevel(7);
	res1dx[0].dropToLevel(7);
	res1dx[1].dropToLevel(7);
	Ctxt fullpackSx = controller.downsample1024to256(res1sx[0], res1sx[1]);

	Ctxt fullpackDx = controller.downsample1024to256(res1dx[0], res1dx[1]);

	controller.num_slots = 8192;

	controller.bootstrap(fullpackSx, true);

	controller.relu(fullpackSx, controller.convbn_4_1_sx_scale);

	if (fullpackSx.NoiseLevel == 2)
		fullpackSx.rescale();
	fullpackSx.dropToLevel(1);
	fullpackSx.copy(controller.convbn2(fullpackSx, 4, 2));
	fullpackSx.add(fullpackDx);
	controller.bootstrap(fullpackSx, true);
	controller.relu(fullpackSx, controller.convbn_4_2_scale);

	if (fullpackSx.NoiseLevel == 2)
		fullpackSx.rescale();
	fullpackSx.dropToLevel(1);
	Ctxt res2 = controller.convbn2(fullpackSx, 5, 1);

	controller.bootstrap(res2, true);

	controller.relu(res2, controller.convbn_5_1_scale);

	if (res2.NoiseLevel == 2)
		res2.rescale();
	res2.dropToLevel(1);
	res2.copy(controller.convbn2(res2, 5, 2));

	fullpackSx.multScalar(controller.convbn_5_2_scale * controller.prescale);

	res2.add(fullpackSx);

	controller.bootstrap(res2, true);

	controller.relu(res2, controller.convbn_5_2_scale);

	if (res2.NoiseLevel == 2)
		res2.rescale();
	res2.dropToLevel(1);
	Ctxt res3 = controller.convbn2(res2, 6, 1);

	controller.bootstrap(res3, true);

	controller.relu(res3, controller.convbn_6_1_scale);

	if (res3.NoiseLevel == 2)
		res3.rescale();
	res3.dropToLevel(1);
	res3.copy(controller.convbn2(res3, 6, 2));

	res2.multScalar(controller.convbn_6_2_scale * controller.prescale);

	res3.add(res2);

	controller.bootstrap(res3, true);

	controller.relu(res3, controller.convbn_6_2_scale);


	return res3;
}

Ctxt layer1(Ctxt &&in) {

	if (in.NoiseLevel == 2)
		in.rescale();
	in.dropToLevel(1);
	Ctxt res1 = controller.convbn(in, 1, 1);

	controller.bootstrap(res1, true);

	controller.relu(res1, controller.convbn_1_1_scale);

	if (res1.NoiseLevel == 2)
		res1.rescale();
	res1.dropToLevel(1);
	res1.copy(controller.convbn(res1, 1, 2));

	Ctxt scaled(in.cc);
	scaled.copy(in);
	scaled.multScalar(controller.convbn_1_2_scale * controller.prescale);

	res1.add(scaled);

	controller.bootstrap(res1, true);

	controller.relu(res1, controller.convbn_1_2_scale);
	if (res1.NoiseLevel == 2)
		res1.rescale();
	res1.dropToLevel(1);
	Ctxt res2 = controller.convbn(res1, 2, 1);

	controller.bootstrap(res2, true);

	controller.relu(res2, controller.convbn_2_1_scale);


	if (res2.NoiseLevel == 2)
		res2.rescale();
	res2.dropToLevel(1);
	res2.copy(controller.convbn(res2, 2, 2));

	res1.multScalar(controller.convbn_2_2_scale * controller.prescale);

	res2.add(res1);

	controller.bootstrap(res2, true);

	controller.relu(res2, controller.convbn_2_2_scale);


	if (res2.NoiseLevel == 2)
		res2.rescale();
	res2.dropToLevel(1);
	Ctxt res3 = controller.convbn(res2, 3, 1);

	controller.bootstrap(res3, true);

	controller.relu(res3, controller.convbn_3_1_scale);

	if (res3.NoiseLevel == 2)
		res3.rescale();
	res3.dropToLevel(1);
	res3.copy(controller.convbn(res3, 3, 2));

	res2.multScalar(controller.convbn_3_2_scale * controller.prescale);

	res3.add(res2);

	controller.bootstrap(res3, true);

	controller.relu(res3, controller.convbn_3_2_scale);


	return res3;
}


void check_arguments(int argc, char *argv[]) {
	generate_context = -1;
	verbose = 0;

	{
		char *res = getenv("FIDESLIB_USE_NUM_GPUS");
		if (res && !(0 == std::strcmp(res, ""))) {
			int num_dev = atoi(res);
			if (num_dev > 0) {
				std::vector<int> dev;
				for (int i = 0; i < num_dev; ++i) {
					dev.push_back(i);
				}
				devices = dev;
			}
			std::cout << "Devices: " << num_dev << std::endl;
		}
	}

	{
		char *res = getenv("FIDESLIB_USE_LOG_RINGDIM");
		if (res && !(0 == std::strcmp(res, ""))) {
			int num_dev = atoi(res);
			if (num_dev > 0) {
				logringdim = num_dev;
			}
			std::cout << "logRingDim: " << num_dev << std::endl;
		}
	}

	{
		char *res = getenv("FIDESLIB_USE_DIM1");
		if (res && !(0 == std::strcmp(res, ""))) {
			int num_dev = atoi(res);
			if (num_dev > 0) {
				dim1 = num_dev;
			}
			std::cout << "dim1: " << num_dev << std::endl;
		}
	}


	for (int i = 1; i < argc; ++i) {
		// I first check the "verbose" command.
		if (string(argv[i]) == "verbose") {
			if (i + 1 < argc) {
				verbose = atoi(argv[i + 1]);
			}
		}
	}

	for (int i = 1; i < argc; ++i) {
		if (string(argv[i]) == "load_keys") {
			if (i + 1 < argc) {
				controller.parameters_folder = "keys_exp" + string(argv[i + 1]);
				if (verbose > 1)
					cout << "Context folder set to: \"" << controller.parameters_folder << "\"." << endl;
				generate_context = 0;
			}
		}

		if (string(argv[i]) == "load_on_the_fly_disk") {
			load_on_the_fly_disk = true;
		}

		if (string(argv[i]) == "load_on_the_fly_gpu") {
			load_on_the_fly_gpu = true;
		}

		if (string(argv[i]) == "generate_keys") {
			if (i + 1 < argc) {
				string folder;
				if (string(argv[i + 1]) == "1") {
					folder = "keys_exp1";
					generate_context = 1;
				} else if (string(argv[i + 1]) == "2") {
					folder = "keys_exp2";
					generate_context = 2;
				} else {
					cerr << "Set a proper value for 'generate_keys'. For instance, use '1'. Check the README.md"
						 << endl;
					exit(1);
				}

				struct stat sb{};
				if (stat(("../" + folder).c_str(), &sb) == 0) {
					cerr << "The keys folder \"" << folder << "\" already exists, I will abort.";
					exit(1);
				} else {
					mkdir(("../" + folder).c_str(), 0777);
				}

				controller.parameters_folder = folder;
				if (verbose > 1)
					cout << "Context folder set to: \"" << controller.parameters_folder << "\"." << endl;
			}
		}
		if (string(argv[i]) == "input") {
			if (i + 1 < argc) {
				input_filename = "../" + string(argv[i + 1]);
				if (verbose > 1)
					cout << "Input image set to: \"" << input_filename << "\"." << endl;
			}
		}
	}
}

vector<double> read_image(const char *filename) {
	int width = 32;
	int height = 32;
	int channels = 3;
	unsigned char *image_data = stbi_load(filename, &width, &height, &channels, 0);

	if (!image_data) {
		cerr << "Could not load the image in " << filename << endl;
		return {};
	}

	vector<double> imageVector;
	imageVector.reserve(width * height * channels);

	for (int i = 0; i < width * height; ++i) {
		// Channel R
		imageVector.push_back(static_cast<double>(image_data[3 * i]) / 255.0f);
	}
	for (int i = 0; i < width * height; ++i) {
		// Channel G
		imageVector.push_back(static_cast<double>(image_data[1 + 3 * i]) / 255.0f);
	}
	for (int i = 0; i < width * height; ++i) {
		// Channel B
		imageVector.push_back(static_cast<double>(image_data[2 + 3 * i]) / 255.0f);
	}

	stbi_image_free(image_data);

	return imageVector;
}
