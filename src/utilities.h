/**
 * @author Stanislaw Wojtkiewicz
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <algorithm>
//#include <cmath>
//#include <cfloat>

/// for debugging and error reporting
#define WHERE_AM_I "@ line " << __LINE__ << " : " __FILE__


/// max number of model structures with different optical properties
#define MAX_STRUCTURES 12
/// ID of boundary structure within the voxelized model
#define ID_STRUCT_BOUNDARY 0
/// ID of source structure within the voxelized model
#define ID_STRUCT_SOURCE 2
/// ID of detector structure within the voxelized model
#define ID_STRUCT_DETECTOR 3
/// ID of first structure within the voxelized model
#define ID_STRUCT_FIRST 4

/// number of optical properties defined
#define NUM_OPTICAL_PROPERTIES 7
//optical properties are [n g mus muax muam muafx muafm q tau]
/// index of refractive index 'n' within the optical properties
#define IDX_N 0
/// index of anisotropy factor 'g' within the optical properties
#define IDX_G 1
/// index of scattering coefficient 'mus' within the optical properties
#define IDX_MUS 2
/// index of medium absorption coefficient at excitation wavelength 'muax' within the optical properties
#define IDX_MUAX 3
/// index of medium absorption coefficient at emission wavelength 'muam' within the optical properties
#define IDX_MUAM 4
/// index of fluorophore absorption coefficient at excitation wavelength 'muafx' within the optical properties
#define IDX_MUAFX 5
/// index of fluorophore absorption coefficient at emission wavelength 'muafm' within the optical properties
#define IDX_MUAFM 6
/// index of quantum yield 'q' within the optical properties
#define IDX_Q 7
/// index of lifetime 'tau' within the optical properties
#define IDX_TAU 8


/// number of parameters per sour-det pair saved in file, e.g. 8 for entry of [sour_x sour_y sour_z sour_r det_x det_y det_z det_r]
#define EM_DET_PAIR_PARAMS 8
/// index of source x coordinate within the em_det_pairs
#define IDX_SOUR_X 0
/// index of source y coordinate within the em_det_pairs
#define IDX_SOUR_Y 1
/// index of source z coordinate within the em_det_pairs
#define IDX_SOUR_Z 2
/// index of source radius coordinate within the em_det_pairs
#define IDX_SOUR_R 3
/// index of detector x coordinate within the em_det_pairs
#define IDX_DET_X 4
/// index of detector y coordinate within the em_det_pairs
#define IDX_DET_Y 5
/// index of detector z coordinate within the em_det_pairs
#define IDX_DET_Z 6
/// index of detector radius coordinate within the em_det_pairs
#define IDX_DET_R 7

using namespace std;

/**
 * \brief Holds hardware parameters of the Cluster computer/node.
 *
 * Please see the "cluster_parameters.txt" file for more details.
 */

struct ClusterComputer{
	///Name of the computer
	string name;
	///Number of the graphical processing units on the computer. Not only cards but the graphical processing units (one card can have more than one GPU).
	int no_of_GPUs;
	///Number of CUDA cores for each GPU
	std::vector<int> CUDA_cores;

	std::vector<float> GFLOPS;
	///Number of threads per block for each GPU
	std::vector<int> threads_per_block;
	///Number of blocks for each GPU when running kernel function
	std::vector<int> no_of_blocks;

	std::vector<int> MPI_worker_ID;
	std::vector<int> GPU_ID;
	std::vector<size_t> totalGLobalMemory;
	std::vector<float> computeCapability;

	ClusterComputer() {
		name = string("");
		no_of_GPUs = 0;
	}

	/**
	 * \brief Assignment overload.
	 * \details Required to make object copy
	 * @param rhs An object to assign from.
	 * @return Reference to the new object
	 */
	ClusterComputer& operator=(const ClusterComputer& rhs) {
		if (this != &rhs) {
			this->name = rhs.name;
			this->no_of_GPUs = rhs.no_of_GPUs;
			this->CUDA_cores = rhs.CUDA_cores;
			this->GFLOPS = rhs.GFLOPS;
			this->threads_per_block = rhs.threads_per_block;
			this->no_of_blocks = rhs.no_of_blocks;
			this->MPI_worker_ID = rhs.MPI_worker_ID;
			this->GPU_ID = rhs.GPU_ID;
			this->totalGLobalMemory = rhs.totalGLobalMemory;
			this->computeCapability = rhs.computeCapability;

		}
		return (*this);
	};

	/**
	 * \brief Copy constructor
	 *
	 * \param sour an object to copy
	 */
	ClusterComputer(const ClusterComputer& sour) {
		this->name = sour.name;// std::string(rhs.settings_filename);
		this->no_of_GPUs = sour.no_of_GPUs;
		this->CUDA_cores = sour.CUDA_cores;
		this->GFLOPS = sour.GFLOPS;
		this->threads_per_block = sour.threads_per_block;
		this->no_of_blocks = sour.no_of_blocks;
		this->MPI_worker_ID = sour.MPI_worker_ID;
		this->GPU_ID = sour.GPU_ID;
		this->totalGLobalMemory = sour.totalGLobalMemory;
		this->computeCapability = sour.computeCapability;
	}

	/**
	 * \brief Prints out the cluster computer details
	 *
	 * \return The string of the cluster computer details
	 */

	std::string toString() {

		std::stringstream message_stream;
		message_stream.str(std::string());

		message_stream << name << std::endl << "GPUs: " << no_of_GPUs;

		message_stream << std::endl << "CUDA cores: ";
		for (size_t i = 0; i < CUDA_cores.size(); i++){
			message_stream << CUDA_cores[i] << " : ";
		}

		message_stream << std::endl << "GFLOPS: ";
		for (size_t i = 0; i < GFLOPS.size(); i++){
			message_stream << GFLOPS[i] << " : ";
		}

		message_stream << std::endl << "Threads per block: ";
		for (size_t i = 0; i < threads_per_block.size(); i++){
			message_stream << threads_per_block[i] << " : ";
		}

		message_stream << std::endl << "Number of block: ";
		for (size_t i = 0; i < no_of_blocks.size(); i++){
			message_stream << no_of_blocks[i] << " : ";
		}

		message_stream << std::endl << "Worker ID: ";
		for (size_t i = 0; i < MPI_worker_ID.size(); i++){
			message_stream << MPI_worker_ID[i] << " : ";
		}

		message_stream << std::endl << "GPU ID: ";
		for (size_t i = 0; i < GPU_ID.size(); i++){
			message_stream << GPU_ID[i] << " : ";
		}

		message_stream << std::endl << "Total global memory: ";
		for (size_t i = 0; i < totalGLobalMemory.size(); i++){
			message_stream << totalGLobalMemory[i] << " : ";
		}

		message_stream << std::endl << "Compute capability: ";
		for (size_t i = 0; i < computeCapability.size(); i++){
			message_stream << computeCapability[i] << " : ";
		}


		message_stream << std::endl << std::flush;
		return message_stream.str();

	}
};


/**
 * \brief Holds simulation parameters
 *
 * Please see the "settings.set" file for more details.
 */
struct Settings{
	///voxels structure dimensions in x direction
	unsigned int x_dim;
	///voxels structure dimensions in y direction
	unsigned int y_dim;
	///voxels structure dimensions in z direction
	unsigned int z_dim;
	///directional cosine of source normal vector in x direction (vector oriented inside the structure)
	//float source_x_cos;
	///directional cosine of source normal vector in y direction (vector oriented inside the structure)
	//float source_y_cos;
	///directional cosine of source normal vector in z direction (vector oriented inside the structure)
	//float source_z_cos;
	///maximum time in DTOF
	float DTOFmax;
	///number of DTOF samples
	int numDTOF;
	///voxel size in mm
	float vox_size;
	///type of voxels update (no update, reflectance, fluorescence generation probability, fluorescence visiting probability)
	unsigned int voxels_update;
	///sensitivity factors calculation switch
	unsigned int sensitivity_factors;
	///number of source voxels
	unsigned int source_coordinates_number;
	///number of remembered scattering events
	int scatering_coordinates_number;
	///number of blocks
	int blocks;
	///number of threads per block
	int threads_per_block;

	///if sources and detectors in a separate file (not in the voxels file)
	int em_det_in_separate_file;
	///min active (observed) source-detector distance
	float em_det_distance_min;
	///max active (observed) source-detector distance
	float em_det_distance_max;

	///if the optical properties change has place
	int opt_prop_change;

	///current GPU id on the BUS, ussed to 'address' multiple GPUs within a cluster worker
	int GPU_ID;

	///to speed up GPU memory access (padding to 128-byte structure)
	///see CUDA manual
	int padding[15];
};


/**
 * \brief Calculates number of cluster workers
 *
 *  A single cluster machine can have more than one GPU.
 *
 * \param cluster_computers Vector with cluster computers as loaded from the file.
 * \param message_stream Stream to hold a message, including errors and exceptions
 * \return The number of workers. In general, number of GPUs within the cluster.
 */
int calculateWorkers(std::vector<ClusterComputer> *cluster_computers, std::stringstream &message_stream);

/**
 * \brief Calculates maximum number of photons threads for a single run. Maximum within the cluster workers.
 *
 * Number of thread per GPU run is number of block time numbers of thread per block. Please see any NVidia CUDA handbook for meening.
 *
 * \param cluster_computers Vector with cluster computers as loaded from the file.
 * \param message_stream Stream to hold a message, including errors and exceptions
 * \return The maximum number within the cluster GPUs.
 */
int calculateMaxGPUSize(std::vector<ClusterComputer> *cluster_computers, std::stringstream &message_stream);


/**
 * \brief
 *
 * \param settings
 * \param DTOFs_MPP
 * \param DTOFs_MTSF
 * \param DTOFs_VSF
 * \param voxels_MPP
 * \param voxels_MTSF
 * \param voxels_VSF
 * \param DTOFs_cut_index
 * \param ind_em_det_pair
 * \param filtfilt
 */
void calculateSensitivityFactors(Settings *settings, float **DTOFs_MPP, float **DTOFs_MTSF, float **DTOFs_VSF, float **voxels_MPP, float **voxels_MTSF, float **voxels_VSF, int *DTOFs_cut_index, int ind_em_det_pair, bool filtfilt);





/**
 * \brief Loads cluster hardware parameters.
 *
 * Please note that this function modifies input parameters. <br>
 * Please see the "cluster_parameters" file for more details.
 * \param cluster_parameters_filename name of the cluster hardware parameters file defined in the settings.set file
 * \param number_of_cluster_computers pointer to the variable holding the number of computers within the cluster (modified by values from the cluster hardware parameters file)
 * \return pointer to the array of the cluster computers hardware parameters
 */
bool loadClusterParametersFile(std::vector<ClusterComputer> *cluster_computers, string cluster_parameters_filename, int *number_of_cluster_computers, std::stringstream &message_stream);

/**
 * \brief Loads simulation settings file
 *
 * Please note that this function modifies input parameters. <br>
 * Please see setting file for file format description.
 * @param settings_filaname name of the settings file
 * @param voxels_filename pointer to variable holding the voxels file name (modified from the settings file)
 * @param optical_properties pointer to variable holding the optical properties array (modified from settings file)
 * @param total_photons pointer to variable holding the total number of photons to simulate (modified from settings file)
 * @param em_det_filename file with source-detectors definitions (optional, modified from the setting file)
 * @param opt_prop_change_filename file with time course of optical properties changes (optional, modified from the setting file)
 * @param cluster_filename file with cluster configuration (modified from the setting file)
 * @return Settings structure
 */
Settings loadSettingsFile(string settings_filaname, string *voxels_filename, float **optical_properties, float *total_photons,
		string *em_det_filename, string *opt_prop_change_filename, string *cluster_filename);

/**
 * \brief Loads ICG bolus file
 *
 * Please note that this function modifies input parameters. <br>
 * Please see Matlab *.m file for ICG bolus file generation to see the ICG bolus file format description.
 * @param bolus_filaname name of ICG bolus file
 * @param bolus_points pointer to variable holding the number of bolus point to simulate (modified by value from ICG bolus file)
 * @param muafx_changes pointer to variable holding the array of optical properties changes in structures (modified by values from ICG bolus file)
 */
bool loadEmDetFile(string em_det_filename, int **em_det_pairs, int *no_of_em_det_pairs, std::stringstream &message_stream);


void loadOpticalPropertiesCourse(string opt_prop_course_filename, float **opt_prop_course, int *no_of_opt_prop_changes);


/**
 * \brief Loads voxels structure from file
 *
 * This function allocates memory for voxels structure. Please note that this function modifies input parameters. <br>
 * Please see Matlab *.m file for voxelized structures generation to see the voxels file format description.
 * @param voxels_filaname name of voxels file
 * @param settings pointer to the simulation settings (modified by value from voxels file)
 * @param voxel_parameters pointer to voxels parameters (modified by value from voxels file)
 * @param source_x_coordinates pointer to array holding source x coordinates (allocated and modified by values from voxels file)
 * @param source_y_coordinates pointer to array holding source y coordinates (allocated and modified by values from voxels file)
 * @param source_z_coordinates pointer to array holding source z coordinates (allocated and modified by values from voxels file)
 * @return pointer to voxelized structure
 */
unsigned int* loadVoxelsFile(string voxels_filaname, Settings *settings, int **source_x_coordinates, int **source_y_coordinates, int **source_z_coordinates);

/**
 * \brief Save voxelized structure with weights in each voxel to file.
 * @param voxels pointer to array holding voxels structure
 * @param settings pointer to settings structure
 * @param voxel_parameters pointer to voxel parameters structure
 */
void saveVoxels(string voxels_filename, float *voxels, Settings *settings);

/**
 * \brief Template to save array of arbitrary type to file. <br> Array can be 1D, 2D or 3D.
 *
 * If array is 3D, each z dimension is saved in a different file (filaname_zDim=*.txt), where *=1, 2 or 3.
 * @param data pointer to an array of arbitrary type, the array is an linear array
 * @param filename file name to save the array
 * @param array_size defines dimension of the array 1D, 2D or 3D
 * @param xDim size of the array in x dimension
 * @param yDim size of the array in y dimension
 * @param zDim size of the array in z dimension
 * \exception throws all exceptions
 */
template <class T>
void saveResults(T *data, string filename, int array_size, int xDim, int yDim, int zDim)
{
	/**
	 * - x dimension in file is along rows
	 * - y dimension in file is along columns
	 * - if array is 2D, the \a zDim is ignored
	 * - if array is 1D, the \a yDim and \a zDim are ignored
	 * - if \a array_size != {1,2,3} this function does nothing
	 * .
	 */

	try
	{
		ofstream sr;
		if (array_size == 3)
		{
			stringstream fname;
			stringstream line;
			for (int z = 0; z < zDim; z++)
			{
				fname << filename << "_zDim=" << z+1 << ".txt";
				sr.open(fname.str().c_str());
				if(sr)
				{
					for (int x = 0; x < xDim; x++)
					{
						line.str("");
						for (int y = 0; y < yDim; y++)
						{
							line << data[x + xDim*y + xDim*yDim*z] << "\t";
						}
						line << "\n";
						sr << line.str();
					}

					sr.close();
				}
				fname.str("");
			}
		}
		else if (array_size == 2)
		{
			stringstream line;
			sr.open(filename.c_str());
			if(sr)
			{
				for (int x = 0; x < xDim; x++)
				{
					line.str("");
					for (int y = 0; y < yDim; y++)
					{
						line << data[x + xDim*y] << "\t";
					}
					line << "\n";
					sr << line.str();
				}

				sr.close();
			}
		}
		else if (array_size == 1)
		{
			sr.open(filename.c_str());
			if(sr)
			{
				for (int x = 0; x < xDim; x++)
				{
					sr << data[x] << "\n";
				}
				sr.close();
			}
		}
	}
	catch (exception &ex)
	{
		cerr << "ERROR: Can't write to file: " << filename << " : " << ex.what();
	}
	catch (...)
	{
		cerr << "ERROR: Can't write to file.";
	}
}

template <class T>
T max1DArray(T **array, int array_1_Dimmension)
{
	T max = (*array)[0];

	for (int i = 0; i < array_1_Dimmension; i++)
	{
		if ((*array)[i]	> max)
		{
			max = (*array)[i];
		}
	}
	return max;
}

template <class T>
int max1DArrayIndex(T **array, int array_1_Dimmension)
{
	int max_index = 0;
	T max = max1DArray<T>(array, array_1_Dimmension);
	for (int i = 0; i < array_1_Dimmension; i++)
	{
		if ((*array)[i]	== max)
		{
			max_index = i;
			break;
		}
	}
	return max_index;
}




#endif
