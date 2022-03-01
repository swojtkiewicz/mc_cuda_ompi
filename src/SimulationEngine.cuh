/**
 * \author Stanislaw Wojtkiewicz
 */

#ifndef SIMULATIONENGINE_CUH
#define SIMULATIONENGINE_CUH

#include "Photon.cuh"

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
//#include <thrust/scan.h>
//#include <thrust/unique.h>
//#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/for_each.h>
//#include <thrust/extrema.h>
//#include <thrust/inner_product.h>
//#include <thrust/sort.h>
//#include <thrust/functional.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <algorithm>

using namespace thrust;

/// size of the DTOF header in the results: time resolution in s, number of samples, source x, source y, source z, source r, detector x, detector y, detector z, detector r,
#define DTOF_HEADER_SIZE 10

/// number of Cartesian coordinates dimensions (x,y,z)
#define DIMENSIONS 3
/// desired max number of scattering events at scattering coefficient of 1 by mm
#define MAX_EVENTS_AT_MUS_1 1500

// to shut up the intellisense mouth :)
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

// to shut up the VS intellisense mouth :)
#ifdef __INTELLISENSE__
void __syncthreads(void);
void __threadfence(void);
#endif

// to shut up the Eclipse CDT parser mouth :)
#ifdef __CDT_PARSER__
#define __host__
#define __device__
void __syncthreads(void);
void __threadfence(void);
#endif

/**
 * \def CUDA_CHECK_RETURN(value)
 * \brief This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * This code is a part of the NVidia CUDA Examples.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "!CUDA! Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/**
 * \def printf(f, ...) ((void)(f, __VA_ARGS__),0)
 * Allows to use printf() inside GPU __device__ functions.
 *
 * This code is a part of the NVidia CUDA Examples.
 * \a printf() is supported only
 * for devices of compute capability 2.0 and higher.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
	#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif


/**
 * Unary function for Thrust
 */
struct castAndClear4YoungBits
{
	template <typename Tuple>
	__host__ __device__ void operator () (Tuple t)
	{

		//for each voxel
		//for (unsigned int i = 0; i < this->settings.x_dim * this->settings.y_dim * this->settings.z_dim; i++)
		//{

		unsigned char *voxel_in;
		//unsigned char *voxel_out;
		//float *voxels_float = new float[settings.x_dim * settings.y_dim * settings.z_dim];
		float voxel_tmp;
		unsigned char *voxel_out = reinterpret_cast<unsigned char *>(&voxel_tmp);

		//read pure bits from unsigned int voxel
		voxel_in = reinterpret_cast<unsigned char *>(&(thrust::get<0>(t)));

		//copy pure bits from unsigned int to float and clear(zero) 4 youngest bits
		voxel_out[0] = voxel_in[0] & 0xF0;
		voxel_out[1] = voxel_in[1];
		voxel_out[2] = voxel_in[2];
		voxel_out[3] = voxel_in[3];


		thrust::get<1>(t) = voxel_tmp;

	}
};

struct clear28OldBits
{
	template <typename T>
	__host__ __device__ void operator () (T t)
	{
		//clear(zero) 4 youngest bits
		t &= 0x0F;
	}
};


/**
 * \brief Namespace engine gives global access to variables and functions defined inside.
 */
namespace engine
{
	///copy of settings which reside in GPU shared memory
	__shared__ Settings shared_settings;

	/**
	 * \brief array to hold dynamically allocated shared GPU memory
	 *
	 * see CUDA manual for details
	 */
	extern __shared__ int array[];

	///dynamically allocated in shared memory
	__shared__ int *shared_source_x_coordinates;

	///dynamically allocated in shared memory
	__shared__ int *shared_source_y_coordinates;

	///dynamically allocated in shared memory
	__shared__ int *shared_source_z_coordinates;

	///dynamically allocated in shared memory
	//__shared__ float *shared_DTOFs;

	///dynamically allocated in shared memory
	__shared__ float *shared_optical_properties;

	/**
	 * \brief Main simulation function
	 * \param d_settings pointer to settings structure in GPU global memory
	 * \param globalState pointer to random number generators array in GPU global memory
	 * \param d_DTOFs pointer to DTOFs array in GPU global memory
	 * \param d_voxels pointer to voxels structure array in GPU global memory
	 * \param d_source_x_coordinates pointer to source x coordinates array in GPU global memory
	 * \param d_source_y_coordinates pointer to source y coordinates array in GPU global memory
	 * \param d_source_z_coordinates pointer to source z coordinates array in GPU global memory
	 * \param d_optical_properties pointer to optical properties array in GPU global memory
	 */
	__global__ void simulate(Settings *d_settings, curandStateMRG32k3a_t *globalState, float *d_DTOFs, unsigned int *d_voxels,
			int *d_source_x_coordinates, int *d_source_y_coordinates, int *d_source_z_coordinates, float *d_optical_properties,
			int *d_x_coordinates,int *d_y_coordinates, int *d_z_coordinates);

	/**
	 * \brief Set random number generators - different generator for each thread
	 * \param globalState pointer to random number generators array in GPU global memory
	 * \param seed_ptr pointer to seed in GPU global memory
	 */
	__global__ void set_randomizer(Settings *d_settings, curandStateMRG32k3a_t *globalState, unsigned long long int *seed_ptr, unsigned long long int *d_sequence, float *d_DTOFs, unsigned int *d_voxels);

	/**
	 * \brief Returns uniformly distributed random number from interval [0, 1]
	 * \param localState random number generator unique for each thread
	 * \return random number from interval [0, 1]
	 */
	__device__ float next_rand(curandStateMRG32k3a_t *localState);


	__device__ void atomicAddFloat28ToFloat32(unsigned int* address, float val);

}

/**
 * \class SimulationEngine SimulationEngine.cuh "SimulationEngine.cuh"
 * \brief Holds simulation engine variables and methods
 * \author Stanislaw Wojtkiewicz
 */
class SimulationEngine
{

private:
	///simulation settings PC
	struct Settings settings;
	///simulation settings GPU
	struct Settings *d_settings = NULL;

	///voxels structure PC
	unsigned int *voxels = NULL;
	///voxels structure GPU
	unsigned int *d_voxels = NULL;

	///source x coordinates PC
	int *source_x_coordinates = NULL;
	///source x coordinates GPU
	int *d_source_x_coordinates = NULL;

	///source y coordinates PC
	int *source_y_coordinates = NULL;
	///source y coordinates GPU
	int *d_source_y_coordinates = NULL;

	///source z coordinates PC
	int *source_z_coordinates = NULL;
	///source z coordinates GPU
	int *d_source_z_coordinates = NULL;

	///optical properties PC
	float *optical_properties = NULL;
	///optical properties GPU
	float *d_optical_properties = NULL;


	///x coordinates of scattering events PC
	int *x_coordinates = NULL;
	///x coordinates of scattering events GPU
	int *d_x_coordinates = NULL;
	///y coordinates of scattering events PC
	int *y_coordinates = NULL;
	///y coordinates of scattering events GPU
	int *d_y_coordinates = NULL;
	///z coordinates of scattering events PC
	int *z_coordinates = NULL;
	///z coordinates of scattering events GPU
	int *d_z_coordinates = NULL;



	/**
	 * \brief Resolution of distribution of time of flight of photons (DTOF)
	 *
	 * dDTOF = max_time_in_DTOF/number_of_DTOF_samples
	 */
	float dDTOF = 0.0f;

	///DTOFs PC
	float *DTOFs = NULL;
	///DTOFs GPU
	float *d_DTOFs = NULL;

	///index of DTOF sample
	int ind_DTOF = 0;

	///photon object
	Photon photon;

	///number of blocks in GPU simulation
	int blocks = 0;
	///threads per block in GPU simulation
	int threads_per_block = 0;

	///holds random number generators (one generator for one thread)
	curandStateMRG32k3a_t *globalState = NULL;
	///random number generator seed PC
	unsigned long long int seed = 0;
	///random number generator seed GPU
	unsigned long long int *seed_ptr = NULL;

	///random number generator sequence number PC
	unsigned long long int sequence = 0;
	///random number generator sequence number GPU
	unsigned long long int *d_sequence = NULL;


	///sizes of DTOF array for GPU memory allocation
	size_t size_DTOFs = 0;
	///sizes of voxels array for GPU memory allocation
	size_t size_voxels = 0;
	///sizes of number of source voxels for GPU memory allocation
	size_t size_source_coordinates = 0;
	///sizes of optical properties array for GPU memory allocation
	size_t size_optical_properties = 0;
	///sizes of number of scattering events for GPU memory allocation
	size_t size_scattering_coordinates = 0;
	///size of dynamically allocated GPU shared memory
	size_t shared_dyn_alloc_bytes = 0;

public:

	/**
	 * \brief Simulation engine constructor
	 * \param settings pointer to simulation parameters structure
	 * \param voxels pointer to voxels structure
	 * \param source_x_coordinates pointer to source voxels x parameters array
	 * \param source_y_coordinates pointer to source voxels y parameters array
	 * \param source_z_coordinates pointer to source voxels z parameters array
	 * \param optical_properties pointer to optical properties array
	 */
	//SimulationEngine(Settings *settings, unsigned int **voxels, float **optical_properties);
	void initialize(Settings *settings, unsigned int **voxels, float **optical_properties);

	SimulationEngine(){};

	///Simulation engine destructor
	~SimulationEngine(){};

	/**
	 * \brief Allocate GPU memory and copy variables to GPU
	 * \param blocks blocks
	 * \param threads_per_block threads per block
	 * \param device GPU device number - GPU number on multi-GPU PC, starts with 0
	 */
	void cudaAllocateCopy2Device(int blocks, int threads_per_block, int device, Settings *settings, float **optical_properties, unsigned int **voxels, int **source_x_coordinates, int **source_y_coordinates, int **source_z_coordinates);


	void cudaCopy2Device_EmDetPairs(Settings *settings, float **optical_properties, unsigned int **voxels, int **source_x_coordinates, int **source_y_coordinates, int **source_z_coordinates);

	void cudaCopy2Device_SensFactors(Settings *settings);

	void resetDevice() {
		cudaDeviceReset();
	};

	/**
	 * \brief Starts the simulation from PC
	 * \param GPU GPU number on multi-GPU PC, starts with 0
	 */
	void run(unsigned long long int seed, int run, int myid, int max_GPU_size);

	/**
	 * \brief Copy results to PC and and free GPU memory
	 */
	void cudaFreeCopy2Host();

	/**
	 * \brief Copy results to PC when simulating ICG bolus
	 */
	void cudaCopy2Host();

	/**
	 * \brief Get voxels structure
	 *
	 * Get voxels without 4 youngest bits (4 youngest bits holds number of structure).
	 * \return voxels structure, where each voxel is of type \a float32 and 4 youngest bits are always set to 0
	 *
	 * - returned voxels are reinterpreted from \a unsigned \a int to \a float32
	 * - 28 oldest bits holds 28 bits \a float (\a float28) number in \a unsigned \a int type
	 */
	void getVoxels(float **voxels_float)
	{

		thrust::for_each(thrust::omp::par,
				thrust::make_zip_iterator(thrust::make_tuple(this->voxels, *voxels_float)),
				thrust::make_zip_iterator(thrust::make_tuple(this->voxels + this->settings.x_dim * this->settings.y_dim * this->settings.z_dim, *voxels_float + this->settings.x_dim * this->settings.y_dim * this->settings.z_dim)),
				castAndClear4YoungBits());

	};

	/**
	 * \brief Return DTOFs
	 * \return DTOFs
	 */
	void getDTOFs(float **DTOFs)
	{
		thrust::copy(thrust::omp::par, this->DTOFs, this->DTOFs + this->settings.numDTOF, *DTOFs);
		//memcpy(*DTOFs, this->DTOFs, sizeof(float) * this->settings.numDTOF);
	};

	//A leftover from the debugging era. However, might be useful to get the scattering coordinates in a separate file.
	void getPathes(int **x_coordinates, int **y_coordinates, int **z_coordinates)
	{
		for (int i = 0; i < this->settings.scatering_coordinates_number * this->blocks * this->threads_per_block; i++)
		{
			(*x_coordinates)[i] = this->x_coordinates[i];
			(*y_coordinates)[i] = this->y_coordinates[i];
			(*z_coordinates)[i] = this->z_coordinates[i];
		}
	};

};

#endif
