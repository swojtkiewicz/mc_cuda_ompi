/**
 * \author Stanislaw Wojtkiewicz
 */

#ifndef PHOTON_CUH
#define PHOTON_CUH

//#include <string>
#include "utilities.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>


/**
 * \class Photon Photon.cuh "Photon.cuh"
 * \brief Holds photon variables and methods
 * \author Stanislaw Wojtkiewicz
 */
class Photon
{
private:
	/*
	/// Photon actual x position
	int x;
	/// Photon actual y position
	int y;
	/// Photon actual z position
	int z;

	/// Photon old x position
	int x0;
	/// Photon old y position
	int y0;
	/// Photon old z position
	int z0;

	/// Photon new x position
	int xn;
	/// Photon new y position
	int yn;
	/// Photon new z position
	int zn;

	float d_x, d_y, d_z, N, sx, sy, sz, x_tmp, y_tmp, z_tmp;

	int new_vox_structure, old_vox_structure;

	bool go_on;

	/// Distance traveled by photon during one scattering event
	float l;
	/// additional variable to calculate distance traveled by photon during one scattering event
	float l_tmp;

	/// new directional cosines calculation parameter
	float den;
	/// Calculation parameter
	float tmp;
	///Photon directional cosine in x direction
	float x_cos;
	/// Photon directional cosine in y direction
	float y_cos;
	///Photon directional cosine in z direction
	float z_cos;

	///Cosine of light scattering angle
	float cost;
	///Sine of light scattering angle
	float sint;
	///Azimuthal angle
	float Phi;
	/// Cosine of azimuthal angle
	float cosp;
	///Azimuthal angle sine
	float sinp;

	///Energy weight of photon
	float weight;

	///If foton moves for the very first time (start of photon)
	bool first;

	/// Reflectance
	float reflectance;
	//float critical_angle_cosin;
	*/

	///random number generator assigned to the very photon
	//curandStateXORWOW *localState;

	///pointer to voxels structure
	//unsigned int *voxels;

	/*
	///counts number of scattering events
	unsigned int counter;
	///holds number of scattering events up to the recent bounce on structure external surface
	///unused when simulating the fluorescence
	//unsigned int counter_old;

	///x coordinates of scattering events
	int x_coordinates[2000];
	///y coordinates of scattering events
	int y_coordinates[2000];
	///z coordinates of scattering events
	int z_coordinates[2000];
	 */

	//int rand_source_point;

	/*
	///actual structure number
	int structure_index;

	///Length of photon path from source to detector
	float path;
	///total length of photon travel in each of 12 structures
	float pathes[12];
	///length of photon travel in each of 12 structures between two scattering events
	float pathes_partial[12];

	///Length of photon path from source to detector for debugging purposes and to stop photons which
	///travel time is longer than maximum time in DTOF
	float path_debug;
	*/
public:
	/**
	 * \brief Photon constructor
	 * @param localState pointer to the random number generator
	 * @param voxels pointer to the voxels structure
	 */
	//__device__ Photon(unsigned int **voxels);

	__device__ Photon(){};

	/**
	 * \brief Photon destructor
	 */
	__device__ ~Photon(){};

	/**
	 * \brief Detect photon
	 */
	//__device__ void detect(int *thread, unsigned int **d_voxels, int *x, int *y, int *z, float *path, int *counter, int **d_x_coordinates, int **d_y_coordinates, int **d_z_coordinates);
	__device__ void detect(int *thread, unsigned int **d_voxels, float **d_DTOFs, int *x, int *y, int *z, float *x_cos, float *y_cos, float *z_cos, float *path, int *counter, int **d_x_coordinates, int **d_y_coordinates, int **d_z_coordinates, curandStateMRG32k3a_t *localState);

	/**
	 * \brief Sets photon start position according to source (x,y,z) position
	 */
	__device__ void setStartXYZ(int *thread, unsigned int **d_voxels, curandStateMRG32k3a_t *localState, int *x, int *y, int *z, int *counter, int **d_x_coordinates, int **d_y_coordinates, int **d_z_coordinates,
								float *x_cos, float *y_cos, float *z_cos);

	/**
	* \brief Updates photon energy weight according to the photon albedo.
	*/
	//__device__ void absorption();


	__device__ void getAzimuthalAngle(curandStateMRG32k3a_t *localState, float *cost, float *sint, float *cosp, float *sinp);

	/**
	* \brief Calculates cosine of the scattering angle.
	*
	* The angle is calculated according to
	* Henyey-Greenstein phase function and anisotropy coeffcient.
	* @return scattering angle
	*/
	__device__ void getScatteringAngle(unsigned int **d_voxels, int *x, int *y, int *z, curandStateMRG32k3a_t *localState, float *cost);

	/**
	* \brief Moves photon to the next scattering event
	*/
	__device__ void movePhoton(int *thread, unsigned int **d_voxels, int *x, int *y, int *z, curandStateMRG32k3a_t *localState, float *weight, float *path, bool *first, bool *detected,
							   int *counter, int **d_x_coordinates, int **d_y_coordinates, int **d_z_coordinates,
							   float *cost, float *sint, float *cosp, float *sinp, float *x_cos, float *y_cos, float *z_cos);

	/**
	 * \brief Bounce photon on the external surface of the structure
	 */
	//__device__ void onBoundary();

	/**
	 * \brief Get photon albedo
	 * @param x pointer to the actual photon x coordinate
	 * @param y pointer to the actual photon y coordinate
	 * @param z pointer to the actual photon z coordinate
	 * @return actual photon albedo
	 */
	//__device__ float getAlbedo(int *x, int *y, int *z);

	/**
	 * \brief Get critical angle cosine when bouncing on the external surface of the structure
	 * @return critical angle cosine
	 */
	//__device__ float getCriticalAngleCosine();


	/**
	* \brief Returns actual photon energy weight
	* @return actual photon weight
	*/
	/*
	__device__ float* getWeight()
	{
		return &this->weight;
	}
	*/
};

#endif
