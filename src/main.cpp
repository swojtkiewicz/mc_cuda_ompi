/**
* Name        : mc_cuda_ompi
* Author      : Stanislaw Wojtkiewicz
* Version     : 1.1
* Description : OpenMPI with CUDA
*/

using namespace std;

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <climits>
#include <chrono>
#include "SimulationEngine.cuh"

using namespace std;


bool checkGPUDevice(cudaError_t *error, int MPI_ID, int ind_worker, std::vector<ClusterComputer> *workers, std::stringstream &message_stream){


	//printf("GPU requested: %d.\n",requested_GPU);

	//helps index MPI_ID for a given worker (GPU)
	int GPUs_so_far_per_workers = 0;
	for (int current_worker = 0; current_worker < ind_worker; current_worker++){
		//go up to a current worker and sum up all GPUs present for all workers
		GPUs_so_far_per_workers += (*workers)[current_worker].no_of_GPUs * ind_worker;
	}

	//check number of available CUDA devices
	int deviceCount = 0;
	*error = cudaGetDeviceCount(&deviceCount);
	if ((deviceCount == 0) || (*error != cudaSuccess)){
		*error = cudaErrorNoDevice;
		message_stream << "It seems like you have no NVidia GPUs. CUDA error code: " << error << std::endl;
		message_stream << WHERE_AM_I << std::endl;
		return false;
	}
	//printf("Device count: %d\n", deviceCount);

	// check if requested correct GPU number
	if ((*workers)[ind_worker].no_of_GPUs != deviceCount) {
		message_stream << "Bad GPU number from file: " << (*workers)[ind_worker].no_of_GPUs << std::endl
				<< "Available GPUs on " << (*workers)[ind_worker].name.c_str() << ": " << deviceCount << std::endl;
		message_stream << WHERE_AM_I << std::endl;
		*error = cudaErrorInvalidDevice;
		return false;
	}

	//check the compute capabilities and choose card with the highest one, within the defined values for which the code was compiled
	cudaDeviceProp deviceProp;
	float deviceCompCap_current = 0.0f;


	for (int device = 0; device < deviceCount; ++device) {
		*error = cudaGetDeviceProperties(&deviceProp, device);

		if (*error == cudaSuccess) {
			deviceCompCap_current = (float)deviceProp.major + 0.1f * (float)deviceProp.minor;

			// add 1 for the master worker, add the GPU index and number of GPUs for all cluster nodes up to now
			(*workers)[ind_worker].MPI_worker_ID.push_back(1 + device + GPUs_so_far_per_workers);
			(*workers)[ind_worker].GPU_ID.push_back(device);
			(*workers)[ind_worker].totalGLobalMemory.push_back(deviceProp.totalGlobalMem);
			(*workers)[ind_worker].computeCapability.push_back(deviceCompCap_current);


			//calculate CUDA cores (single precision threads that can be run simultaneously)
			int CUDA_cores = 0;

			switch (deviceProp.major){
				case 3:
					CUDA_cores = 192 * deviceProp.multiProcessorCount;
					break;
			    case 5:
					CUDA_cores = 128 * deviceProp.multiProcessorCount;
					break;
			    case 6:
			    	if ((deviceProp.minor == 1) || (deviceProp.minor == 2))
			    		CUDA_cores = 128 * deviceProp.multiProcessorCount;
			    	else if (deviceProp.minor == 0)
			    		CUDA_cores = 64 * deviceProp.multiProcessorCount;
			    	else{
			    		message_stream << "Unknown device type. Detected CUDA compute capability: " << deviceCompCap_current << std::endl;
						message_stream << "Assuming 64 CUDA cores per multiprocessor." << std::endl;
						message_stream << WHERE_AM_I << std::endl;
						CUDA_cores = 64 * deviceProp.multiProcessorCount;
			    	}
			    	break;
			    case 7:
			    	CUDA_cores = 64 * deviceProp.multiProcessorCount;
			    	break;
			    case 8:
			    	if (deviceProp.minor == 0)
			    		CUDA_cores = 64 * deviceProp.multiProcessorCount;
			    	else if (deviceProp.minor == 6)
			    		CUDA_cores = 128 * deviceProp.multiProcessorCount;
			    	else{
			    		message_stream << "Unknown device type. Detected CUDA compute capability: " << deviceCompCap_current << std::endl;
						message_stream << "Assuming 64 CUDA cores per multiprocessor." << std::endl;
						message_stream << WHERE_AM_I << std::endl;
						CUDA_cores = 64 * deviceProp.multiProcessorCount;
			    	}
			    	break;
			    default:
			    	if (deviceProp.major > 8){
			    		message_stream << "Unknown NEW device type. Detected CUDA compute capability: " << deviceCompCap_current << std::endl;
			    		message_stream << "Assuming 128 CUDA cores per multiprocessor." << std::endl;
						message_stream << WHERE_AM_I << std::endl;

						CUDA_cores = 128 * deviceProp.multiProcessorCount;
			    		break;
			    	}else{
						message_stream << "Unknown device type. CUDA compute capability: " << deviceCompCap_current << std::endl;
						message_stream << WHERE_AM_I << std::endl;
						*error = cudaErrorInvalidDevice;
						return false;
			    	}
			}

			//calculate theoretical gigaflops - computational power
			float gigaflops = (float)deviceProp.clockRate / 1000000.0f; // the clock rate conversion from kHz to GHz
			gigaflops *= (float)CUDA_cores;

			(*workers)[ind_worker].CUDA_cores.push_back(CUDA_cores);
			(*workers)[ind_worker].GFLOPS.push_back(gigaflops);

		}else{
			message_stream << "Error while reading GPU properties. Device ID: " << device << ". Worker ID: " << ind_worker << "." << std::endl;
			message_stream << WHERE_AM_I << std::endl;
			*error = cudaErrorInvalidDevice;
			return false;
		}
	}

	return true;
};


/**
 * \brief Shows the progress bar at the console
 * \param float progress_all actual simulated photons
 * \param total_photons total photons to simulate
 */
inline void progressBar(float progress_all, float total_photons)
{
	//calculation parameter
	float x = 0.0f;
	// number of signs of the progress bar in the console window
	float signs = 40.0f;
	// Calculate the ratio of complete-to-incomplete.
	float ratio = progress_all/total_photons;
	//complete
	float c = roundf(ratio * signs);

	// Show the percentage complete. setw(3) - sets the 3 signs width to show the percentage (avoid different widths of percentage)
	cout << setw(3) << (int)(ratio*100.0f) << "% [";

	// Show the complete signs
	for (x = 0.0f; x < c; x++)
		cout << "=";

	//show the incomplete white signs
	for (x = c; x < signs; x++)
		cout << " ";

	//print the end signs, print immediately and go to the first sign in the line
	cout << "]\r" << flush;
}

/**
 * \brief The main function
 * \param argc number of input arguments - not used
 * \param argv input arguments - not used
 */
int main(int argc, char *argv[])
{

	// Initialise the Message Passing Interface
	MPI::Init(argc,argv);
	//set error handler
	MPI::COMM_WORLD.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);

	//*************************************** NOW TRY TO EXECUTE WITHIN THE CLUSTER
	try {

		/**
		//check aeguments number
		if (argc < 2) {
			showHelp(argv[0]);
			return 1;
		}

		//default values
		char* mesh_core_name = NULL;
		char* femdata_file_name = NULL;
		int requested_GPU = -1;
		bool isCPU = false;
		double absolute_tolerance = 1e-12;
		double relative_tolerance = 1e-12;

		//loop through args
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if ((arg == "-h") || (arg == "--help")) {
				showHelp(argv[0]);
				return 0;
			}
			else if ((arg == "-o") || (arg == "--out_file_name")) {
				if (i + 1 < argc) {
					femdata_file_name = argv[++i];
				}
				else {
					std::cerr << "--out_file_name option requires one argument." << std::endl;
					return 1;
				}
			}
			else if (arg == "--gpu") {
				if (i + 1 < argc) {
					requested_GPU = atoi(argv[++i]);
				}
				else {
					std::cerr << "--gpu option requires one argument." << std::endl;
					return 1;
				}
			}
			else if (arg == "--cpu") {
				isCPU = true;
			}
			else if ((arg == "-abs") || (arg == "--absolute_tolerance")) {
				if (i + 1 < argc) {
					absolute_tolerance = atof(argv[++i]);
				}
				else {
					std::cerr << "--absolute_tolerance requires one argument." << std::endl;
					return 1;
				}
			}
			else if ((arg == "-rel") || (arg == "--relative_tolerance")) {
				if (i + 1 < argc) {
					relative_tolerance = atof(argv[++i]);
				}
				else {
					std::cerr << "--relative_tolerance requires one argument." << std::endl;
					return 1;
				}
			}
			else {
				mesh_core_name = argv[i];
			}
		}

		 * */


		//holds length of the actual MPI processor name and number of GPU workers within the cluster
		int namelen = 0, num_GPU_workers = 0;
		//holds the actual MPI processor name
		char processor_name[MPI::MAX_PROCESSOR_NAME];
		//number of workers within cluster
		int numprocs = MPI::COMM_WORLD.Get_size();
		//id number of the current MPI processor
		int myid = MPI::COMM_WORLD.Get_rank();
		//current worker name
		MPI::Get_processor_name(processor_name,namelen);
		//max number of blocks times threads per blocks per single run, max for all workers within the cluster, used to set the correct offset for parallel random number generators
		int max_GPU_size = 0;

		//Distributions of time of flight of photons (DTOF) for the current MPI processor
		float *DTOFs = NULL;
		//Distributions of time of flight of photons (DTOF) for all MPI processor
		float *DTOFs_all = NULL;

		//helpers for sensitivity factors calculation
		float *DTOFs_MPP = NULL;
		float *DTOFs_MTSF = NULL;
		float *DTOFs_VSF = NULL;

		//only for the root MPI process
		float *DTOFs_em_det_pairs = NULL;
		float *result = NULL;

		//helper
		int DTOFs_cut_index = 0;

		//simulation settings
		Settings settings;

		//cluster computers parameters
		std::vector<ClusterComputer> cluster_computers;
		//number of cluster computers as requested in the cluster settings file
		int number_of_cluster_computers = 0;

		//calculation arrays index parameters
		int i = 0, j = 0, ind = 0;
		//holds size of DTOF and voxels structure
		int size_DTOFs = 0, size_voxels = 0;

		// progress helpers
		float progress = 0.0f;
		float progress_all = 0.0f;
		float total_photons = 0.0f;

		//sources coordinates
		int *source_x_coordinates = NULL;
		int *source_y_coordinates = NULL;
		int *source_z_coordinates = NULL;

		//time-based seed
		unsigned long long int seed = 0;
		if (myid == 0){
			//Initialise at the master only, it will be broadcasted to other workers
			seed = time(NULL);
		}

		/// holds formatted error message
		stringstream message_stream;
		message_stream.str(string());


		//***************************************load settings file
		//variables for filenames
		string voxels_filename;
		string em_det_filename;
		string opt_prop_changes_filename;
		string cluster_filename;
		//optical properties within structures
		float *optical_properties = NULL;


		settings = loadSettingsFile("settings.set",&voxels_filename, &optical_properties, &total_photons,
									&em_det_filename, &opt_prop_changes_filename, &cluster_filename);


		//make room for reflectance and fluorescence DTOFs
		settings.numDTOF *= 2;

		//if we will calculate the sensitivity factors, force necessary parameters
		if (settings.sensitivity_factors > 0)
		{
			//voxels will be updated and the sens. factors calculated
			settings.voxels_update = 0;
			settings.sensitivity_factors = 1;
		}


		//***************************************load cluster parameters file

		if (!loadClusterParametersFile(&cluster_computers, cluster_filename, &number_of_cluster_computers, message_stream)){
			std::cout << std::endl << "Something went wrong:(" << std::endl;
			std::cout << message_stream.str() << std::endl << std::endl << std::flush;
			return 1;
		}

		//cout << "Number of cluster computers: " << number_of_cluster_computers << endl << flush;
		//for (int idx = 0; idx < number_of_cluster_computers; idx++){
		//	cout << cluster_computers[idx].toString() << endl << flush;
		//}


		//check for workers, if we have some
		num_GPU_workers = calculateWorkers(&cluster_computers, message_stream);
		if (num_GPU_workers < 1){
			std::cout << std::endl << "No workers read from the cluster setting file (" << num_GPU_workers << ")." << std::endl;
			std::cout << message_stream.str() << std::endl << std::endl << std::flush;
			return 1;
	   }

		//get the max number of blocks times threads per blocks per single run, max for all workers within the cluster, used to set the correct offset for parallel random number generators
		max_GPU_size = calculateMaxGPUSize(&cluster_computers, message_stream);
		//std::cout << "max GPU : " << max_GPU_size << endl;
		if (max_GPU_size < 1){
			std::cout << std::endl << "Bad GPU run size (" << max_GPU_size << ")." << std::endl;
			std::cout << message_stream.str() << std::endl << std::endl << std::flush;
			return 1;
	   }


		//***************************************load voxelized model
		// the model as loaded
		unsigned int *voxels = NULL;
		// the model with sources and detectors
		unsigned int *voxels_em_det_pairs = NULL;

		// set number of source voxels to 0, this is to check if a source is loaded
		settings.source_coordinates_number = 0;
		voxels = loadVoxelsFile(voxels_filename, &settings, &source_x_coordinates, &source_y_coordinates, &source_z_coordinates);

		//this can be 0 if sources defined in a separate file
		if ((settings.source_coordinates_number == 0) && (settings.em_det_in_separate_file == 0)){
			std::cout << std::endl << "No source voxels in the voxelized model." << std::endl << std::flush;
			return 1;
		}

		// number of voxels
		size_voxels = settings.x_dim * settings.y_dim * settings.z_dim;

		// voxels with sources and detectors
		voxels_em_det_pairs = new unsigned int[size_voxels];
		//memcpy(voxels_em_det_pairs, voxels, sizeof(unsigned int) * size_voxels);


		//***************************************load source-detector pairs coordinates (if needed)
		//this will hold the source-det pairs from separate file if needed
		int *em_det_pairs = NULL;
		//assume a single source-detector pair for all voxels of source (2) and detector (3) indexes as defined within the voxelized structure.
		int no_of_em_det_pairs = 1;
		//now, if sources and detectors provided in the separate file
		if (settings.em_det_in_separate_file == 1)
		{

			if(!loadEmDetFile(em_det_filename, &em_det_pairs, &no_of_em_det_pairs, message_stream)){
				std::cout << std::endl << "Something went wrong:(" << std::endl;
				std::cout << message_stream.str() << std::endl << std::endl << std::flush;
				return 1;
			}

		}

		//***************************************load optical properties changes file (if needed)
		float *opt_prop_course = NULL;
		int no_of_opt_prop_changes = 1;
		if (settings.opt_prop_change == 1)
		{
			loadOpticalPropertiesCourse(opt_prop_changes_filename, &opt_prop_course, &no_of_opt_prop_changes);
		}


		//***************************************set other parameters and helpers after reading files



		// helpers to save results for sensitivity factors (MPP, MTSF and VSF)
		float *voxels_MTSF = NULL;
		float *voxels_VSF = NULL;

		//allocate output
		// local results for a given worker
		float *voxels_float = NULL;
		// accumulated results
		float *voxels_all = NULL;


		if ((settings.voxels_update > 0) || (settings.sensitivity_factors > 0))
		{
			voxels_all = new float[size_voxels];
			voxels_float = new float[size_voxels];


			thrust::fill(thrust::omp::par,voxels_all, voxels_all + size_voxels, 0.0f);
			thrust::fill(thrust::omp::par,voxels_float, voxels_float + size_voxels, 0.0f);
		}

		if (settings.sensitivity_factors > 0)
		{
			voxels_MTSF = new float[size_voxels];
			voxels_VSF = new float[size_voxels];

			thrust::fill(thrust::omp::par,voxels_MTSF, voxels_MTSF + size_voxels, 0.0f);
			thrust::fill(thrust::omp::par,voxels_VSF, voxels_VSF + size_voxels, 0.0f);
		}

		//max scattering coefficient within the model, in mm^-1 unit
		float max_scattering = 0.0f;

		for (i = 0; i < MAX_STRUCTURES; i++)
		{
			// look for the maximum mus within the model in mm^-1
			if (optical_properties[i * NUM_OPTICAL_PROPERTIES + IDX_MUS] > max_scattering) {
				max_scattering = optical_properties[i * NUM_OPTICAL_PROPERTIES + IDX_MUS];
			}

			//convert optical properties to voxel units
			for (j = 0; j < NUM_OPTICAL_PROPERTIES; j++)
			{
				//only for optical properties having the per length units
				if ((j != IDX_N) && (j != IDX_G) && (j != IDX_Q) && (j != IDX_TAU))
					optical_properties[i*NUM_OPTICAL_PROPERTIES + j] *= settings.vox_size;

				//cout << optical_properties[i*NUM_OPTICAL_PROPERTIES + j] << " " << flush;
			}
			//cout << endl << flush;
		}


		//check for voxel size as in Wojtkiewicz_2021 paper
		if (max_scattering != 0) {
			//check the condition as in Wojtkiewicz_2021 paper
			if (settings.vox_size > 1 / (3 * max_scattering)) {
				message_stream.str(string()); message_stream.clear();
				message_stream << endl << "WARNING" << endl << "Too big voxel size compared to the scattering:" << endl
					<< "voxel size : " << settings.vox_size << "mm" << endl
					<< "max scattering : " << max_scattering << "mm^-1" << endl
					<< "Results might be inaccurate. Consider reducing the voxel size below 1/(3*mus)" << endl << endl;

				cout << message_stream.str() << flush;
				message_stream.str(string()); message_stream.clear();
			}
		}

		// DTOFs
		size_DTOFs = settings.numDTOF;
		if (settings.sensitivity_factors > 0)
		{
			DTOFs = new float[size_DTOFs];
			DTOFs_all = new float[size_DTOFs];
			DTOFs_MPP = new float[size_DTOFs];
			DTOFs_MTSF = new float[size_DTOFs];
			DTOFs_VSF = new float[size_DTOFs];

			thrust::fill(thrust::omp::par,DTOFs, DTOFs + size_DTOFs, 0.0f);
			thrust::fill(thrust::omp::par,DTOFs_all, DTOFs_all + size_DTOFs, 0.0f);
			thrust::fill(thrust::omp::par,DTOFs_MPP, DTOFs_MPP + size_DTOFs, 0.0f);
			thrust::fill(thrust::omp::par,DTOFs_MTSF, DTOFs_MTSF + size_DTOFs, 0.0f);
			thrust::fill(thrust::omp::par,DTOFs_VSF, DTOFs_VSF + size_DTOFs, 0.0f);
		}
		else
		{
			DTOFs = new float[size_DTOFs];
			DTOFs_all = new float[size_DTOFs];

			thrust::fill(thrust::omp::par,DTOFs, DTOFs + size_DTOFs, 0.0f);
			thrust::fill(thrust::omp::par,DTOFs_all, DTOFs_all + size_DTOFs, 0.0f);
		}

		// workers execution time
		time_t *t_start_workers = new time_t[numprocs];
		time_t *t_end_workers = new time_t[numprocs];


		//requests declarations for non-blocking send/recv within the cluster
		MPI_Request request_DTOFs;
		MPI_Request request_voxels;
		MPI_Request request_seed;
		MPI_Request *request_progress_send = new MPI_Request[numprocs-1];
		MPI_Request *request_progress_recv = new MPI_Request[numprocs-1];


		//ID 0 is always the MPI master
		if (myid == 0)
		{
			//initialise results
			DTOFs_em_det_pairs = new float[size_DTOFs * no_of_em_det_pairs];
			thrust::fill(thrust::omp::par,DTOFs_em_det_pairs, DTOFs_em_det_pairs + size_DTOFs * no_of_em_det_pairs, 0.0f);

			result = new float[(size_DTOFs + DTOF_HEADER_SIZE) * no_of_em_det_pairs];
			thrust::fill(thrust::omp::par,result, result + (size_DTOFs + DTOF_HEADER_SIZE) * no_of_em_det_pairs, 0.0f);

		}

		// chrono timers to get exec times
		auto t_start_main = std::chrono::high_resolution_clock::now();

		auto t_start = std::chrono::high_resolution_clock::now();
		auto t_stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_start).count();

		//used to calculate if we have enough GPU memory resources
		size_t free_GPU_global_memory = 0;


		//***********************************************gather GPUs information across the cluster

		//this worker GPU calculation power
		float worker_gflops = 0.0f;
		//all workers GPUs calculation power
		float *workers_gflops = new float[numprocs];
		memset(workers_gflops, 0, sizeof(float) * numprocs);

		// CUDA error code
		cudaError_t error = cudaSuccess;

		// if not the master
		if (myid > 0){
			//loop through the workers loaded from file
			for (size_t ind_worker = 0; ind_worker < cluster_computers.size(); ind_worker++){
				//if the entry is for the current worker
				if (strcmp(processor_name, cluster_computers[ind_worker].name.c_str()) == 0){
					// check GPUs on the current worker
					if(!checkGPUDevice(&error, myid, ind_worker, &cluster_computers, message_stream)){
						std::cout << std::endl << "Something went wrong getting GPUs info:(" << std::endl;
						std::cout << message_stream.str() << std::endl << std::endl << std::flush;
						return 1;
					}

					//display any message if needed
					if (!message_stream.str().empty()) {
						std::cout << std::endl << message_stream.str() << std::endl << std::endl << std::flush;
						message_stream.str(string()); message_stream.clear();
					}
				}

				//get gflops performance for current worker
				for(size_t ind_GPU = 0; ind_GPU < cluster_computers[ind_worker].GPU_ID.size(); ind_GPU++){
					if (cluster_computers[ind_worker].MPI_worker_ID[ind_GPU] == myid){
						//get the GFLOPs
						worker_gflops = cluster_computers[ind_worker].GFLOPS[ind_GPU];
						//set the GPU id for the current worker
						settings.GPU_ID = cluster_computers[ind_worker].GPU_ID[ind_GPU];
					}
				}
			}
		}

		//cout << "GPU ID : " << settings.GPU_ID << " @ " << myid << endl;

		//distribute the current worker performance across the cluster, this is blockin operation
		MPI_Allgather(&worker_gflops, 1, MPI_FLOAT, workers_gflops, 1, MPI_FLOAT, MPI_COMM_WORLD);

		//for (int ind_worker = 0; ind_worker < numprocs; ind_worker++){
		//	cout << workers_gflops[ind_worker] << " : ";
		//}
		//cout << endl << flush;


		//check if we have enough global memory

		// if not the master
		if (myid > 0)
		{
			//loop through the workers loaded from file
			for (size_t ind_worker = 0; ind_worker < cluster_computers.size(); ind_worker++)
			{
				//calculate the memory requirements for current worker
				for(size_t ind_GPU = 0; ind_GPU < cluster_computers[ind_worker].GPU_ID.size(); ind_GPU++)
				{
					if (cluster_computers[ind_worker].MPI_worker_ID[ind_GPU] == myid)
					{
						//worker_gflops = cluster_computers[ind_worker].GFLOPS[ind_GPU];

						//subtract the needed global memory from total available
						free_GPU_global_memory = cluster_computers[ind_worker].totalGLobalMemory[ind_GPU] -
								(sizeof(settings) +
										sizeof(curandStateMRG32k3a) * (size_t)cluster_computers[ind_worker].no_of_blocks[ind_GPU] * (size_t)cluster_computers[ind_worker].threads_per_block[ind_GPU] +
										sizeof(int) +
										sizeof(float) * settings.numDTOF +
										sizeof(unsigned int) * settings.x_dim * settings.y_dim * settings.z_dim +
										DIMENSIONS * sizeof(int) * settings.source_coordinates_number +
										sizeof(float) * MAX_STRUCTURES * NUM_OPTICAL_PROPERTIES);

						//assume we can use up to 80% of the memory left to store the scattering coordinates
						size_t max_allowed_coordinate_events = (size_t)roundf(0.8f*(float)(free_GPU_global_memory) / (float)(DIMENSIONS * sizeof(int) * cluster_computers[ind_worker].no_of_blocks[ind_GPU] * cluster_computers[ind_worker].threads_per_block[ind_GPU]));

						//relate number of max scattering event to the mus
						//what we do here is to set the needed number of scattering events to save/tracks following a linear function that gives
						//up to 1500 scattering event for the scattering coefficient of 1 by mm and up to 15000 events for the scattering coefficient of 10 by a single mm
						//this linear function was checked empirically, adding more events is under the noise floor
						float slope = (float)(10.0f*MAX_EVENTS_AT_MUS_1 - MAX_EVENTS_AT_MUS_1) / (10.0f - 1.0f);
						float bias = (float)MAX_EVENTS_AT_MUS_1 - slope;
						settings.scatering_coordinates_number = (optical_properties[IDX_MUS] / settings.vox_size) * slope + bias; //we go back to mm unit here. Thus, dividing by the voxel size.

						//now check if we have enough memory to run it properly
						if (((size_t)settings.scatering_coordinates_number <= max_allowed_coordinate_events) &&
								(max_allowed_coordinate_events >= MAX_EVENTS_AT_MUS_1))
						{
							//make it at least 1500
							if (settings.scatering_coordinates_number < MAX_EVENTS_AT_MUS_1) {
								settings.scatering_coordinates_number = MAX_EVENTS_AT_MUS_1;
							}
						}
						else
						{
							std::cout << std::endl << "Not enough GPU global memory : " << processor_name << "(" << myid << ")" << std::endl
								<< "Try to reduce: size of the source, voxel size or the model size. " << std:: endl
								<< (float)(MAX_EVENTS_AT_MUS_1 * DIMENSIONS * sizeof(int) * (size_t)cluster_computers[ind_worker].no_of_blocks[ind_GPU] * (size_t)cluster_computers[ind_worker].threads_per_block[ind_GPU]) - 0.8f*(float)free_GPU_global_memory
								<< " Bytes are needed." << std::endl << std::endl << std::flush;
							//error_cpu = 1;
							return 1;
						}

						//std::cout << "MAX SCATTER EVENTS: " << settings.scatering_coordinates_number << " : " << processor_name << "(" << myid << ")" << std::endl << std::flush;

						settings.blocks = cluster_computers[ind_worker].no_of_blocks[ind_GPU];
						settings.threads_per_block = cluster_computers[ind_worker].threads_per_block[ind_GPU];
					}
				}
			}
		}


		//calculate the workload balance across cluster
		float total_gflops = 0.0f;
		for (int ind_worker = 1; ind_worker < numprocs; ind_worker++){
			total_gflops += workers_gflops[ind_worker];
		}


		//computational power fraction for the current worker
		float cores_balance = 0.0f;
		//calculate the execution power balance and show the theoretical computational power
		if (myid > 0)
		{
			//std::cout << "TFLOPS " << processor_name << "(" << myid << ")" << ": " << workers_gflops[myid]/1000.0f << std::endl << std::flush;
			//std::cout << "cores factor: " << workers_gflops[myid]/total_gflops << std::endl << std::flush;
			cores_balance = workers_gflops[myid]/total_gflops;
		}

		//max runs per machine within cluster, used to service the progress bar
		float max_runs = 0;
		//how often the execution progress is reported to the master, used to service the progress bar
		float percentage_step = 0.05f;
		//helper counters to increase the progress bar percentage
		float recv_progress = 1.0f;
		float sent_progress = 1.0f;


		//number of kernels execution per the current worker GPU
		float runs_per_GPU = 0.0f;
		if (myid > 0){
			runs_per_GPU = ceilf( (total_photons * cores_balance) / (float)settings.blocks / (float)settings.threads_per_block );
		}

		//std::cout << "RUNS PER GPU: " << runs_per_GPU << std::endl << std::flush;


		//update the total simulated photons. Number of the simulated photons should be a reciprocal of number of blocks and threads per block
		//clear the number of photons to simulate
		total_photons = 0.0f;
		//this is number of photons to simulate for a given worker
		float photons_per_worker = 0.0f;
		if (myid > 0){
			photons_per_worker = runs_per_GPU * (float)settings.blocks * (float)settings.threads_per_block;
		}

		//now, integrate number of simulated photons over the cluster
		MPI_Reduce(&photons_per_worker, &total_photons, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

		//if (myid ==0)
		//	std::cout << "Total photons gathered: " << total_photons << std::endl << std::flush;

		//get the maximum runs for a worker within the cluster
		MPI_Reduce(&runs_per_GPU, &max_runs, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

		//if (myid ==0)
		//	std::cout << "Max runs within the cluster: " << max_runs << std::endl;

		// create the simulation engine
		SimulationEngine engine = SimulationEngine();
		//and initialise for workers with GPUs
		if (myid > 0){
			engine.initialize(&settings, &voxels, &optical_properties);
		}

		//display general info
		if (myid == 0)
		{
			std::cout << "\n   Using \"root + " << numprocs-1 << "\" processes" << std::endl;
			std::cout << "   Root master is - " << processor_name << std::endl;
			std::cout << "   TFLOPS Total: " << total_gflops/1000.0f << std::endl << std::endl << std::flush;

			std::cout << "+ total photons = " << total_photons << std::endl;

			if (settings.sensitivity_factors == 0)
			{
				cout << "\nCalculating...\n" << std::endl;
			}
			else
			{
				cout << "\nCalculating sensitivity factors...\n" << std::endl;
			}
		}else{
			std::cout << "+ TFLOPS " << processor_name << "(" << myid << ")" << ": " << workers_gflops[myid]/1000.0f << std::endl;
		}


		//broadcast the see across the cluster, all workers knows the same seed base to start with
		MPI_Ibcast(&seed, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD, &request_seed);
		MPI_Wait(&request_seed, MPI_STATUSES_IGNORE);

		//std::cout << "seed : " << processor_name << "(" << myid << ")" << " : " << seed << std::endl;


		///repeat the whole simulation for each optical property change
		for (int ind_opt_prop_change = 0; ind_opt_prop_change < no_of_opt_prop_changes; ind_opt_prop_change++)
		{
			//set the current optical properties to the next one from file
			if (settings.opt_prop_change == 1)
			{
				for (i = 0; i< MAX_STRUCTURES; i++){
					for (j = 0; j< NUM_OPTICAL_PROPERTIES; j++){

						optical_properties[i*NUM_OPTICAL_PROPERTIES + j] = opt_prop_course[ind_opt_prop_change*MAX_STRUCTURES*NUM_OPTICAL_PROPERTIES + j*MAX_STRUCTURES + i];

						//only for optical properties having the per length units
						if ((j != IDX_N) && (j != IDX_G) && (j != IDX_Q) && (j != IDX_TAU))
							optical_properties[i*NUM_OPTICAL_PROPERTIES + j] *= settings.vox_size;

						//std::cout << optical_properties[i*NUM_OPTICAL_PROPERTIES + j] << " " << std::flush;
					}
					//std::cout << std::endl;
				}
			}

			///repeat the whole simulation for each source-detector pair!
			for (int ind_em_det_pair = 0; ind_em_det_pair < no_of_em_det_pairs; ind_em_det_pair++)
			{
				//make a unique seed for a given simulation (optical property and sour-det pair)
				seed += ind_opt_prop_change*no_of_em_det_pairs + ind_em_det_pair;
				//std::cout << "seed : " << processor_name << "(" << myid << ")" << " : " << seed << std::endl;

				//load fresh model
				thrust::copy(thrust::omp::par, voxels, voxels + size_voxels, voxels_em_det_pairs);

				//set fresh sensitivity factors results
				if (settings.sensitivity_factors > 0)
				{
					thrust::fill(thrust::omp::par,voxels_all, voxels_all + size_voxels, 0.0f);
					thrust::fill(thrust::omp::par,voxels_MTSF, voxels_MTSF + size_voxels, 0.0f);
					thrust::fill(thrust::omp::par,voxels_VSF, voxels_VSF + size_voxels, 0.0f);

					thrust::fill(thrust::omp::par,DTOFs_MPP, DTOFs_MPP + size_DTOFs, 0.0f);
					thrust::fill(thrust::omp::par,DTOFs_MTSF, DTOFs_MTSF + size_DTOFs, 0.0f);
					thrust::fill(thrust::omp::par,DTOFs_VSF, DTOFs_VSF + size_DTOFs, 0.0f);
				}

				//set fresh DTOF results
				thrust::fill(thrust::omp::par,DTOFs_all, DTOFs_all + size_DTOFs, 0.0f);

				//now modify the model based on the source-detector info loaded from file
				if (settings.em_det_in_separate_file == 1)
				{
					///////////////////////////////ADD CURRENT SOURCE//////////////////////////////////////////

					// start and stop of the x coordinate
					int ind_x_start = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_X] - (int)roundf(1 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_R]); //sour_x - (sour_r + 1)
					if (ind_x_start < 1) //there is a void boundary around the model, 1 voxel-thick, zero valued, reserved
						ind_x_start = 1;
					int ind_x_stop = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_X] + (int)roundf(1 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_R]); //sour_x + (sour_r + 1)
					if (ind_x_stop > (int)settings.x_dim - 1) //there is a void boundary around the model, 1 voxel-thick, zero valued, reserved
						ind_x_stop = settings.x_dim - 1;
					// start and stop of the y coordinate
					int ind_y_start = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Y] - (int)roundf(1 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_R]); //sour_y - (sour_r + 1)
					if (ind_y_start < 1)
						ind_y_start = 1;
					int ind_y_stop = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Y] + (int)roundf(1 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_R]); //sour_y + (sour_r + 1)
					if (ind_y_stop > (int)settings.y_dim - 1)
						ind_y_stop = settings.y_dim - 1;
					// start and stop of the z coordinate
					int ind_z_start = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Z] - (int)roundf(1 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_R]); //sour_z - (sour_r + 1)
					if (ind_z_start < 1)
						ind_z_start = 1;
					int ind_z_stop = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Z] + (int)roundf(1 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_R]); //sour_z + (sour_r + 1)
					if (ind_z_stop > (int)settings.z_dim - 1)
						ind_z_stop = settings.z_dim - 1;

					//holds number of boundary voxels touching the source voxel
					int zero_voxels_counter = 0;
					//how many source voxels we have
					settings.source_coordinates_number = 0;
					//calculate number of adjacent boundary voxels
					//this is needed for automatic normal to the surface vector calculation
					//TO_DO - add definition of normal/direction vectors into the sour-det file
					//TO_DO - add option to put sources within the model
					for (int ind_x = ind_x_start; ind_x < ind_x_stop; ind_x++){
						for (int ind_y = ind_y_start; ind_y < ind_y_stop; ind_y++){
							for (int ind_z = ind_z_start; ind_z < ind_z_stop; ind_z++){
								//check if within the sphere of source radius
								if (sqrtf((float)((ind_x - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_X])*(ind_x - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_X]) +
										  (ind_y - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Y])*(ind_y - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Y]) +
										  (ind_z - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Z])*(ind_z - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_Z]))) <= (float)em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_SOUR_R])
								{
									//reset boundary voxels counter
									zero_voxels_counter = 0;
									//check within the surrounding cube
									for (i = ind_x - 1; i <= ind_x + 1; i++){
										for (j = ind_y - 1; j <= ind_y + 1; j++){
											for (ind = ind_z - 1; ind <= ind_z + 1; ind++){
												//if the cube centre is the model (not source, not detector or source) and the adjacent cube element is a boundary voxel
												if ((voxels[ind_x + settings.x_dim*ind_y + settings.x_dim*settings.y_dim*ind_z] >= ID_STRUCT_FIRST) &&
														(voxels[i + settings.x_dim*j + settings.x_dim*settings.y_dim*ind] == ID_STRUCT_BOUNDARY))
												{
													//increase the counter of adjacent boundary voxels
													zero_voxels_counter++;
												}
											}
										}
									}

									// if there source voxels touches the boundary
									if (zero_voxels_counter > 0)
									{
										//increase the number of source voxels
										settings.source_coordinates_number++;
										//set the current voxel ID to source
										voxels_em_det_pairs[ind_x + settings.x_dim*ind_y + settings.x_dim*settings.y_dim*ind_z] = ID_STRUCT_SOURCE;
									}
								}
							}
						}
					}

					//TO_DO - add definition of normal/direction vectors into the sour-det file, as someone might want to add source inside the model
					//for now, just end the program if no adjacent boundary to the source
					if(zero_voxels_counter == 0){
						message_stream.str(string()); message_stream.clear();
						message_stream << "\nThe specified source voxel does not touch the void boundary. Please change the source voxel location.\n";
						message_stream << "source number: " << ind_em_det_pair << "\n";
						for (int ind_sour_element = 0; ind_sour_element < EM_DET_PAIR_PARAMS; ind_sour_element++)
							message_stream<< em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + ind_sour_element] << " ";

						std::cout << message_stream.str() << "\n\n" << std::endl;
						return 1;
					}

					//allocate memory for source voxels coordinates
					source_x_coordinates = new int[settings.source_coordinates_number];
					source_y_coordinates = new int[settings.source_coordinates_number];
					source_z_coordinates = new int[settings.source_coordinates_number];


					//loop through the model and gather source voxels coordinates
					int counter_source_voxels = 0;
					for (i = 0; i < (int)settings.x_dim; i++){
						for (j = 0; j < (int)settings.y_dim; j++){
							for (ind = 0; ind < (int)settings.z_dim; ind++){

								if (voxels_em_det_pairs[i + settings.x_dim*j + settings.x_dim*settings.y_dim*ind] == ID_STRUCT_SOURCE)
								{
									source_x_coordinates[counter_source_voxels] = i;
									source_y_coordinates[counter_source_voxels] = j;
									source_z_coordinates[counter_source_voxels] = ind;
									counter_source_voxels++;
								}
							}
						}
					}

					///////////////////////////////ADD CURRENT DETECTOR//////////////////////////////////////////
					/// set in the voxelized model the detector voxels based on coordinates from the source-detector pair file
					// start and stop of the x coordinate
					ind_x_start = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_X] - (int)roundf(2 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_R]); //det_x - (det_r + 2)
					if (ind_x_start < 1)
						ind_x_start = 1;
					ind_x_stop = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_X] + (int)roundf(2 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_R]); //det_x + (det_r + 2)
					if (ind_x_stop > (int)settings.x_dim - 1)
						ind_x_stop = settings.x_dim - 1;
					// start and stop of the y coordinate
					ind_y_start = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Y] - (int)roundf(2 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_R]); //det_y - (det_r + 2)
					if (ind_y_start < 1)
						ind_y_start = 1;
					ind_y_stop = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Y] + (int)roundf(2 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_R]); //det_y + (det_r + 2)
					if (ind_y_stop > (int)settings.y_dim - 1)
						ind_y_stop = settings.y_dim - 1;
					// start and stop of the z coordinate
					ind_z_start = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Z] - (int)roundf(2 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_R]); //det_z - (det_r + 2)
					if (ind_z_start < 1)
						ind_z_start = 1;
					ind_z_stop = em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Z] + (int)roundf(2 + em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_R]); //det_z + (det_r + 2)
					if (ind_z_stop > (int)settings.z_dim - 1)
						ind_z_stop = settings.z_dim - 1;

					for (int ind_x = ind_x_start; ind_x < ind_x_stop; ind_x++){
						for (int ind_y = ind_y_start; ind_y < ind_y_stop; ind_y++){
							for (int ind_z = ind_z_start; ind_z < ind_z_stop; ind_z++)
							{
								if (sqrtf((float)((ind_x - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_X])*(ind_x - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_X]) +
										  (ind_y - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Y])*(ind_y - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Y]) +
										  (ind_z - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Z])*(ind_z - em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_Z]))) <= (float)em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + IDX_DET_R])
								{
									zero_voxels_counter = 0;
									for (i = ind_x - 1; i <= ind_x + 1; i++){
										for (j = ind_y - 1; j <= ind_y + 1; j++){
											for (ind = ind_z - 1; ind <= ind_z + 1; ind++)
											{

												if ((voxels[ind_x + settings.x_dim*ind_y + settings.x_dim*settings.y_dim*ind_z] >= ID_STRUCT_FIRST) &&
														(voxels[i + settings.x_dim*j + settings.x_dim*settings.y_dim*ind] == ID_STRUCT_BOUNDARY))
												{
													zero_voxels_counter++;
												}
											}
										}
									}

									if (zero_voxels_counter > 0)
									{
										voxels_em_det_pairs[ind_x + settings.x_dim*ind_y + settings.x_dim*settings.y_dim*ind_z] = ID_STRUCT_DETECTOR;
									}
								}
							}
						}
					}
				}



				///set the progress bar service
				if (myid == 0)
				{
					//clear the counters
					progress = 0.0f;
					progress_all = 0.0f;

					//put more info as needed
					//if no sensitivity factors calculation
					if (settings.sensitivity_factors == 0)
					{
						//if the optical properties are changed as well
						if (no_of_opt_prop_changes > 1)
						{
							std::cout << "opt_prop change: " << ind_opt_prop_change + 1 << "/" << no_of_opt_prop_changes << ";\tsour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << std::endl;
						}
						else
						{
							//just the source-detector information
							std::cout << "sour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << std::endl;
						}
					}
					//if sensitivity factors calculation
					else
					{
						//if the optical properties are changed as well
						if (no_of_opt_prop_changes > 1)
						{
							cout << "opt_prop change: " << ind_opt_prop_change + 1 << "/" << no_of_opt_prop_changes << ";\tsour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << ";\tsens_factors (1/3)" << endl;
						}
						else
						{
							//just the source-detector information
							cout << "sour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << ";\tsens_factors (1/3)" << endl;
						}
					}

					//reset the progress factor
					recv_progress = 1.0f;

					// initiate listeners to all workers except master
					for (i = 1; i < numprocs; i++)
						MPI_Recv_init(&progress, 1, MPI_FLOAT, i, 11, MPI_COMM_WORLD, &request_progress_recv[i - 1]);

					//now start receiving in intervals of 5%
					//loop up to maximum runs for a single worker
					for (float ind_f = 0; ind_f < max_runs; ind_f++)
					{
						//if we have reached the intervals, 5%, 10%, 15%, etc.
						if (ind_f / max_runs > recv_progress * percentage_step)
						{
							//start listening to workers
							for (i = 1; i < numprocs; i++)
							{
								MPI_Start(&request_progress_recv[i - 1]);
								MPI_Wait(&request_progress_recv[i - 1], MPI_STATUS_IGNORE);
								//accumulate
								progress_all += progress;
								//display the progress bar where 100% is total photons to run
								progressBar(progress_all, total_photons);
							}
							//increment the progress by 5%
							recv_progress++;
						}
					}
					//show 100% if finished
					progressBar(total_photons, total_photons);
					std::cout << std::endl;
				} // end (myid == 0)



				//now service the workers
				if (myid > 0)
				{
					//allocate GPUs memory and copy stuff to GPUs
					if ((ind_opt_prop_change + ind_em_det_pair) == 0)
					{
						//if the very first run, allocate and copy all data
						engine.cudaAllocateCopy2Device(settings.blocks, settings.threads_per_block, settings.GPU_ID, &settings, &optical_properties, &voxels_em_det_pairs, &source_x_coordinates, &source_y_coordinates, &source_z_coordinates);
					}
					else
					{
						//if the consecutive run, just copy the necessary stuff that changed
						engine.cudaCopy2Device_EmDetPairs(&settings, &optical_properties, &voxels_em_det_pairs, &source_x_coordinates, &source_y_coordinates, &source_z_coordinates);
					}


					//reset the progress factor
					sent_progress = 1.0f;
					//initialise sending to the master process
					MPI_Send_init(&progress, 1, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, &request_progress_send[myid - 1]);

					//t_start_work = time(0);
					t_start_workers[myid-1] = time(0);

					//this wiil be the CUDA kernel execution time
					duration = 0;

					//run as many times as precalculated
					for (float ind_f = 0; ind_f < runs_per_GPU; ind_f++)
					{
						//start counting kernel execution time
						t_start = std::chrono::high_resolution_clock::now();
						//run the kernel
						engine.run(seed, (int)ind_f, myid, max_GPU_size);
						//the stop execution time marker
						t_stop = std::chrono::high_resolution_clock::now();
						//calculate the kernel execution time
						duration += std::chrono::duration_cast<std::chrono::milliseconds>(t_stop - t_start).count();
						//now, handle sending the progress every 5% interval
						if (ind_f / runs_per_GPU > sent_progress*percentage_step)
						{
							//calculate the progress
							progress = (percentage_step * runs_per_GPU) * (float)settings.blocks * (float)settings.threads_per_block;
							//start sending
							MPI_Start(&request_progress_send[myid - 1]);
							//increase by 5%
							sent_progress++;
							//no need to wait for the communication to finish, just go on with the next run
							//MPI_Wait(&request_progress_send[myid - 1], MPI_STATUS_IGNORE);
							//std::cout << "send (" << myid << ") - " << ind+myid+myid-1 << std::endl;
						}
					}

					//worker finished entire
					t_end_workers[myid-1] = time(0);

					//display the time just once at the end
					if ((ind_opt_prop_change == no_of_opt_prop_changes - 1) && (ind_em_det_pair == no_of_em_det_pairs - 1)){
						//show total exec time
						std::cout << "\n\t+ It took " << processor_name << "(" << myid-1 << "): " << difftime(t_end_workers[myid-1], t_start_workers[myid-1]) << " second(s)." << std::endl;
						//show GPU exec time
						std::cout << "\t+ Kernel run @ " << processor_name << "(" << myid-1 << "): " << (double)duration / 1000.0 << " second(s)." << std::endl;
					}

					//decide what to do with the memory, if we will run for a new pair or optical properties
					if ((settings.sensitivity_factors == 0) && (ind_opt_prop_change == no_of_opt_prop_changes - 1) && (ind_em_det_pair == no_of_em_det_pairs - 1))
					{
						//if all is done, copy and free GPU memory
						engine.cudaFreeCopy2Host();
					}
					else
					{
						//if not finished, just copy to FPU memory to host
						engine.cudaCopy2Host();
					}

					//copy results from the GPU enfine to this worker process
					engine.getDTOFs(&DTOFs);

					if ((settings.voxels_update > 0) || (settings.sensitivity_factors > 0))
					{
						engine.getVoxels(&voxels_float);
					}

					/*
					if (myid == 1)
					{
						int *x_coordinates = new int[settings.scatering_coordinates_number*blocks_zeus*threads_per_block_zeus];
						int *y_coordinates = new int[settings.scatering_coordinates_number*blocks_zeus*threads_per_block_zeus];
						int *z_coordinates = new int[settings.scatering_coordinates_number*blocks_zeus*threads_per_block_zeus];

						cout << "\nTrying to copy coordinates\n";

						engine.getPathes(&x_coordinates,&y_coordinates,&z_coordinates);

						cout << "\nCoordinates transfered\n";

						saveResults<int>(x_coordinates, "x_coordinates.txt", 2, settings.scatering_coordinates_number, blocks_zeus * threads_per_block_zeus, 0);
						saveResults<int>(y_coordinates, "y_coordinates.txt", 2, settings.scatering_coordinates_number, blocks_zeus * threads_per_block_zeus, 0);
						saveResults<int>(z_coordinates, "z_coordinates.txt", 2, settings.scatering_coordinates_number, blocks_zeus * threads_per_block_zeus, 0);

					}
					*/
				} //end worker job

				//if 3D results are requested, accumulate over the cluster
				if ((settings.voxels_update > 0) || (settings.sensitivity_factors > 0))
				{
					//accumulate voxel results
					MPI_Ireduce(voxels_float, voxels_all, size_voxels, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &request_voxels);
					MPI_Wait(&request_voxels, MPI_STATUS_IGNORE);
				}

				//if we update voxels only
				if ((settings.voxels_update > 0) && (settings.sensitivity_factors == 0)){
					if (myid == 0){
						//on the master, copy the result for the current source-det pair as we will be added to the results from the next source
						thrust::copy(thrust::omp::par, voxels_all, voxels_all + size_voxels, voxels_float);
					}else{
						//reset this worker results (not master) as we will need it for the next pair or opt properties calculation
						thrust::fill(thrust::omp::par, voxels_float, voxels_float + size_voxels, 0.0f);
					}
				}

				//if we calculate sensitivity factors
				if (settings.sensitivity_factors > 0)
				{
					//reset all worker results (including the master) as we will need it clean for the next pair calculation
					thrust::fill(thrust::omp::par, voxels_float, voxels_float + size_voxels, 0.0f);

					//accumulate DTOFs as this is needed to calculate the consecutive sens factors
					MPI_Ireduce(DTOFs, DTOFs_MPP, size_DTOFs, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &request_DTOFs);
					MPI_Wait(&request_DTOFs, MPI_STATUS_IGNORE);
					//now reset current worker results we will need later
					thrust::fill(thrust::omp::par, DTOFs, DTOFs + size_DTOFs, 0.0f);

					//now set the flag, that we will calculate the 2nd sensitivity factor
					settings.sensitivity_factors = 2;
				}
				else
				{
					//accumulate DTOFs
					MPI_Ireduce(DTOFs, DTOFs_all, size_DTOFs, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &request_DTOFs);
					MPI_Wait(&request_DTOFs, MPI_STATUS_IGNORE);
					//now reset current worker results we might need later
					thrust::fill(thrust::omp::par, DTOFs, DTOFs + size_DTOFs, 0.0f);
				}

				//for master only, save the results for a given source-detector pair
				//results are saved to a disk every source-detector pair is finished
				if (myid == 0)
				{
					//std::cout << "(" << ind_em_det_pair << ") " << "RESULT" << &DTOFs << "(" << DTOFs[25] << ")" << " : " << &DTOFs_all << "(" << DTOFs_all[25] << ")" << "\t" << processor_name << "(" << myid << ")" << std::endl;

					//save the DTOFs to a file, always
					//set the DTOF header first
					result[0 + ind_em_det_pair*(size_DTOFs + DTOF_HEADER_SIZE)] = settings.DTOFmax/(float)(settings.numDTOF/2);
					result[1 + ind_em_det_pair*(size_DTOFs + DTOF_HEADER_SIZE)] = (float)settings.numDTOF;

					for (int ind_param = 0; ind_param < EM_DET_PAIR_PARAMS; ind_param++){
						//if we had em-det pairs from separate file, include the geometry read from the file as a header
						result[ind_param + 2 + ind_em_det_pair*(size_DTOFs + DTOF_HEADER_SIZE)] =
								settings.em_det_in_separate_file == 1 ? (float)em_det_pairs[ind_em_det_pair*EM_DET_PAIR_PARAMS + ind_param] : 0.0;
					}
					//now set the DTOF data
					memcpy(result + ind_em_det_pair*(size_DTOFs + DTOF_HEADER_SIZE) + DTOF_HEADER_SIZE, settings.sensitivity_factors > 0 ? DTOFs_MPP : DTOFs_all, sizeof(float)*size_DTOFs);


					//save DTOF results to a file
					stringstream fname;
					fname << "DTOFs_" << ind_opt_prop_change + 1 << ".txt" << std::flush;
					saveResults<float>(result, fname.str().c_str(), 2, size_DTOFs + DTOF_HEADER_SIZE, no_of_em_det_pairs, 0);



					//if no sens factors and if voxels results present
					if ((settings.sensitivity_factors == 0) && (settings.voxels_update > 0))
					{
						stringstream fname_vox;
						fname_vox << "voxels_out_" << ind_opt_prop_change + 1 << ".vox";
						saveVoxels(fname_vox.str().c_str(),voxels_all, &settings);
					}

				}

				////////////////////SENSITIVITY FACTORS///////////////

				if (settings.sensitivity_factors > 0)
				{
					////////////////////////////////////MTSF//////////////////////////////////////

					if (myid == 0)
					{
						//reset the cumulative progress
						progress = 0.0f;
						progress_all = 0.0f;

						//if the optical properties are changed as well
						if (no_of_opt_prop_changes > 1)
						{
							cout << "opt_prop change: " << ind_opt_prop_change + 1 << "/" << no_of_opt_prop_changes << ";\tsour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << ";\tsens_factors (2/3)" << endl;
						}
						else
						{
							//just the source-detector information
							cout << "sour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << ";\tsens_factors (2/3)" << endl;
						}

						//reset the progress factor
						recv_progress = 1.0f;

						// initiate listeners to all workers except master
						for (i = 1; i < numprocs; i++)
							MPI_Recv_init(&progress, 1, MPI_FLOAT, i, 11, MPI_COMM_WORLD, &request_progress_recv[i - 1]);

						//now start receiving in intervals of 5%
						//loop up to maximum runs for a single worker
						for (float ind_f = 0; ind_f < max_runs; ind_f++)
						{
							//if we have reached the intervals, 5%, 10%, 15%, etc.
							if (ind_f / max_runs > recv_progress * percentage_step)
							{
								//start listening to workers
								for (i = 1; i < numprocs; i++)
								{
									MPI_Start(&request_progress_recv[i - 1]);
									MPI_Wait(&request_progress_recv[i - 1], MPI_STATUS_IGNORE);
									//accumulate
									progress_all += progress;
									//display the progress bar where 100% is total photons to run
									progressBar(progress_all, total_photons);
								}
								//increment the progress by 5%
								recv_progress++;
							}
						}
						//show 100% if finished
						progressBar(total_photons, total_photons);
						std::cout << std::endl;
						//printf("\n");

					}//end (myid == 0)


					//if for workers with a GPU
					if (myid > 0)
					{
						//update the GPU variables needed to calculate the sensitivity factors
						engine.cudaCopy2Device_SensFactors(&settings);

						//reset the progress factor
						sent_progress = 1.0f;
						//initialise sending to the master process
						MPI_Send_init(&progress, 1, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, &request_progress_send[myid - 1]);

						//run as many times as precalculated
						for (float ind_f = 0; ind_f < runs_per_GPU; ind_f++)
						{
							//run the kernel
							engine.run(seed, (int)ind_f, myid, max_GPU_size);
							//now, handle sending the progress every 5% interval
							if (ind_f / runs_per_GPU > sent_progress*percentage_step)
							{
								//calculate the progress
								progress = (percentage_step * runs_per_GPU) * (float)settings.blocks * (float)settings.threads_per_block;
								//start sending
								MPI_Start(&request_progress_send[myid - 1]);
								//increase by 5%
								sent_progress++;
								//no need to wait for the communication to finish, just go on with the next run
								//MPI_Wait(&request_progress_send[myid - 1], MPI_STATUS_IGNORE);
								//std::cout << "send (" << myid << ") - " << ind+myid+myid-1 << std::endl;
							}
						}

						//copy results to the PC meory
						engine.cudaCopy2Host();

						//copy results to MPI process
						engine.getDTOFs(&DTOFs);
						engine.getVoxels(&voxels_float);

					}


					//accumulate results
					MPI_Ireduce(DTOFs, DTOFs_MTSF, size_DTOFs, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &request_DTOFs);
					MPI_Ireduce(voxels_float, voxels_MTSF, size_voxels, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &request_voxels);

					//wait for the DTOFs to transfer across the cluster
					MPI_Wait(&request_DTOFs, MPI_STATUS_IGNORE);
					//now reset current worker results we will need later
					thrust::fill(thrust::omp::par, DTOFs, DTOFs + size_DTOFs, 0.0f);

					//wait for the voxels to transfer across the cluster
					MPI_Wait(&request_voxels, MPI_STATUS_IGNORE);
					//reset all worker results (including the master) as we will need it clean for the next pair calculation
					thrust::fill(thrust::omp::par, voxels_float, voxels_float + size_voxels, 0.0f);


					//now set the flag, that we will calculate the 3rd sensitivity factor
					settings.sensitivity_factors = 3;


					///////////////////////////////////////VSF////////////////////////////////////////////


					if (myid == 0)
					{

						//reset the cumulative progress
						progress = 0.0f;
						progress_all = 0.0f;


						if (no_of_opt_prop_changes > 1)
						{
							cout << "opt_prop change: " << ind_opt_prop_change + 1 << "/" << no_of_opt_prop_changes << ";\tsour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << ";\tsens_factors (3/3)" << endl;
						}
						else
						{
							//just the source-detector information
							cout << "sour-det pair: " << ind_em_det_pair + 1 << "/" << no_of_em_det_pairs << ";\tsens_factors (3/3)" << endl;
						}

						//reset the progress factor
						recv_progress = 1.0f;

						// initiate listeners to all workers except master
						for (i = 1; i < numprocs; i++)
							MPI_Recv_init(&progress, 1, MPI_FLOAT, i, 11, MPI_COMM_WORLD, &request_progress_recv[i - 1]);

						//now start receiving in intervals of 5%
						//loop up to maximum runs for a single worker
						for (float ind_f = 0; ind_f < max_runs; ind_f++)
						{
							//if we have reached the intervals, 5%, 10%, 15%, etc.
							if (ind_f / max_runs > recv_progress * percentage_step)
							{
								//start listening to workers
								for (i = 1; i < numprocs; i++)
								{
									MPI_Start(&request_progress_recv[i - 1]);
									MPI_Wait(&request_progress_recv[i - 1], MPI_STATUS_IGNORE);
									//accumulate
									progress_all += progress;
									//display the progress bar where 100% is total photons to run
									progressBar(progress_all, total_photons);
								}
								//increment the progress by 5%
								recv_progress++;
							}
						}
						//show 100% if finished
						progressBar(total_photons, total_photons);
						std::cout << std::endl;
						//printf("\n");

					}//end (myid == 0)



					if (myid > 0)
					{
						//update the GPU variables needed to calculate the sensitivity factors
						engine.cudaCopy2Device_SensFactors(&settings);

						//reset the progress factor
						sent_progress = 1.0f;
						//initialise sending to the master process
						MPI_Send_init(&progress, 1, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, &request_progress_send[myid - 1]);

						//run as many times as precalculated
						for (float ind_f = 0; ind_f < runs_per_GPU; ind_f++)
						{
							//run the kernel
							engine.run(seed, (int)ind_f, myid, max_GPU_size);
							//now, handle sending the progress every 5% interval
							if (ind_f / runs_per_GPU > sent_progress*percentage_step)
							{
								//calculate the progress
								progress = (percentage_step * runs_per_GPU) * (float)settings.blocks * (float)settings.threads_per_block;
								//start sending
								MPI_Start(&request_progress_send[myid - 1]);
								//increase by 5%
								sent_progress++;
								//no need to wait for the communication to finish, just go on with the next run
								//MPI_Wait(&request_progress_send[myid - 1], MPI_STATUS_IGNORE);
								//std::cout << "send (" << myid << ") - " << ind+myid+myid-1 << std::endl;
							}
						}

						//if the last run
						if ((ind_opt_prop_change == (no_of_opt_prop_changes - 1)) && (ind_em_det_pair == (no_of_em_det_pairs - 1)))
						{
							//copy to host and free CUDA memory resources
							engine.cudaFreeCopy2Host();
						}
						else
						{
							//copy results to the PC memory
							engine.cudaCopy2Host();
						}

						//copy results to MPI process
						engine.getDTOFs(&DTOFs);
						engine.getVoxels(&voxels_float);

					}



					//accumulate results
					MPI_Ireduce(DTOFs, DTOFs_VSF, size_DTOFs, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &request_DTOFs);
					MPI_Ireduce(voxels_float, voxels_VSF, size_voxels, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD, &request_voxels);

					//wait for the DTOFs to transfer across the cluster
					MPI_Wait(&request_DTOFs, MPI_STATUS_IGNORE);
					//now reset current worker results we will need later
					thrust::fill(thrust::omp::par, DTOFs, DTOFs + size_DTOFs, 0.0f);

					//wait for the voxels to transfer across the cluster
					MPI_Wait(&request_voxels, MPI_STATUS_IGNORE);
					//reset all worker results (including the master) as we will need it clean for the next pair calculation
					thrust::fill(thrust::omp::par, voxels_float, voxels_float + size_voxels, 0.0f);


					//now set the flag, that we will go back to calculate the 1st sensitivity factor
					settings.sensitivity_factors = 1;




					if (myid == 0)
					{
						//calculate the sensitivity factor from what we have calculated so far. Refer to Liebert_2004 publication on how to do it
						calculateSensitivityFactors(&settings, &DTOFs_MPP, &DTOFs_MTSF, &DTOFs_VSF, &voxels_all, &voxels_MTSF, &voxels_VSF, &DTOFs_cut_index, ind_em_det_pair, 0);

						stringstream fname;

						fname << "voxels_MPP_em_det_pair_" << ind_em_det_pair + 1 << "_opt_prop_" << ind_opt_prop_change + 1 << ".vox";
						saveVoxels(fname.str().c_str(),voxels_all, &settings);
						fname.str(std::string());

						fname << "voxels_MTSF_em_det_pair_" << ind_em_det_pair + 1 << "_opt_prop_" << ind_opt_prop_change + 1 << ".vox";
						saveVoxels(fname.str().c_str(),voxels_MTSF, &settings);
						fname.str(std::string());

						fname << "voxels_VSF_em_det_pair_" << ind_em_det_pair + 1 << "_opt_prop_" << ind_opt_prop_change + 1 << ".vox";
						saveVoxels(fname.str().c_str(),voxels_VSF, &settings);
						fname.str(std::string());

						//progressBar(total_photons, total_photons);
						//printf("\n");

					}

				}
			}
		}

		//reset GPUs for workers
		if (myid > 0){
			engine.resetDevice();
		}

		//finish the MPI communication
		MPI_Finalize();

		//calculate the overall execution time
		auto t_stop_main = std::chrono::high_resolution_clock::now();
		auto duration_main = std::chrono::duration_cast<std::chrono::milliseconds>(t_stop_main - t_start_main).count();


		std::cout << "Total time: " << (double)duration_main / 1000.0 << " second(s) : " << processor_name << "(" << myid << ")" << std::endl;

	}
	catch (MPI::Exception &e)
	{
		cout << "MPI ERROR: " << e.Get_error_code() \
			 << " - " << e.Get_error_string() \
			 << endl;
		return 1;
	}
	return 0;
}
