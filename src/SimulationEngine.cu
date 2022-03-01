
#include "SimulationEngine.cuh"
#include <cuda_runtime.h>
//#include <math_functions.h>
#include <cuda_runtime_api.h>
//#include <device_functions.h>
#include <math_constants.h>

__global__ void engine::set_randomizer(Settings *d_settings, curandStateMRG32k3a_t *globalState, unsigned long long int *seed_ptr, unsigned long long int *d_sequence, float *d_DTOFs, unsigned int *d_voxels)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	//if the first thread, set the shared GPU memory
	if (threadIdx.x == 0)
	{
		//assign statically allocated shared memory
		shared_settings = *d_settings;
	}
	//Important: all threads in block should wait while the first thread sets the shared memory!
	__syncthreads();


	/**
	 *  Each thread gets the same seed, a different sequence number and no offset.
	 *  Thus, each thread has the different random number generator. The sequence number is:
	 *  \code int i = blockIdx.x * blockDim.x + threadIdx.x; \endcode
	 */
	curand_init(*seed_ptr, *d_sequence + i, 0, &globalState[i]);

}

__device__ float engine::next_rand(curandStateMRG32k3a_t *localState)
{
	//printf("%f\t%f\t%f\n",curand_uniform(localState),curand_uniform(localState),curand_uniform(localState));
	return curand_uniform(localState);
	//return ((float)curand(localState))/4294967295.0f;
}


__device__ void Photon::detect(int *thread, unsigned int **d_voxels, float **d_DTOFs, int *x, int *y, int *z, float *x_cos, float *y_cos, float *z_cos, float *path, int *counter, int **d_x_coordinates, int **d_y_coordinates, int **d_z_coordinates, curandStateMRG32k3a_t *localState)
{

	///for Bresenhalm's 3D line algorithm
	float d_x, d_y, d_z, N, x_tmp, y_tmp, z_tmp;

	int i, j;
	float Pc = 0.0f; // fluorescence conversion probability at each scattering event
	float Pc_all = 0.0f; // fluorescence conversion probability at whole path from the source to the detector
	float tau_tmp = 0.0f; // temporary variable for tau calculation, prevents from __logf equal infinity
	float Wx = 0.0f; // excitation weight
	//float Wx_all = 0.0f; // excitation weight on the detector
	float Wm = 0.0f; // emission weight
	float Wm_all = 0.0f; // emission weight

	//values used in reinterpreting unsigned int to float28
	//unsigned int voxel_tmp;
	//float weight_tmp;
	//unsigned char *voxel_weight;
	//unsigned char *voxel_weight_tmp;

	int structure_index = 0;

	int ind_DTOF = 0;

	//lengths of paths in 12 different structures between two consecutive scattering events
	float pathes_tmp[MAX_STRUCTURES];
	///total length of photon travel in each of 12 structures
	float pathes[MAX_STRUCTURES];
	///length of photon travel in each of 12 structures between two scattering events
	float pathes_partial[MAX_STRUCTURES];
	for (j = 0; j < MAX_STRUCTURES; j++)
	{
		//pathes_tmp[j] = 0;
		pathes[j] = 0.0f;
		pathes_partial[j] = 0.0f;
		//pathes_tmp[j] = 0.0f;
	}

	//calculate the whole path and paths in all 12 different structures
	for (i = 0; i < *counter; i++)
	{
		for (j = 0; j < MAX_STRUCTURES; j++)
		{
			pathes_tmp[j] = 0.0;
		}

		//Breshalm's 3D line
		//calculate coordinates differences in x, y and z directions between actual and the next scattering events
		d_x = (float)(*d_x_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_x_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
		d_y = (float)(*d_y_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_y_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
		d_z = (float)(*d_z_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_z_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
		//calculate number of steps N in the longest direction
		N = 0.0f;
		if (fabsf(d_x) > N)
			N = fabsf(d_x);
		if (fabsf(d_y) > N)
			N = fabsf(d_y);
		if (fabsf(d_z) > N)
			N = fabsf(d_z);

		//start position (integer) - the scattering event position
		(*x) = (*d_x_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
		(*y) = (*d_y_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
		(*z) = (*d_z_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];

		//which structure in the start position
		structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

		if (structure_index >= ID_STRUCT_FIRST)
		{
			//if the start position inside the voxels structure, update path of the adequate structure
			pathes_tmp[structure_index - ID_STRUCT_FIRST]++;
		}
		else //if ((structure_index == 2) || (structure_index == 3))
		{
			//if the start position on emiter or detector update the path of the most external structure
			pathes_tmp[0]++;
		}

		//temporary position (float)
		x_tmp = (float)*x;
		y_tmp = (float)*y;
		z_tmp = (float)*z;

		//if the line length longer than 1 voxel
		if (N > 1)
		{
			//for the number of steps
			//because the next scattering event starts at the end of the previous scattering event,
			//the line stops before the last step N (j < N)
			for (j = 1; j < N; j++)
			{
				//update the temporary position (float) by the steps (float)
				x_tmp += d_x/N;
				y_tmp += d_y/N;
				z_tmp += d_z/N;

				//calculate the positon (integer)
				*x = roundf(x_tmp);
				*y = roundf(y_tmp);
				*z = roundf(z_tmp);

				//chceck the structure number on the actual position
				structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

				if (structure_index >= ID_STRUCT_FIRST)
				{
					//if the actual position inside the voxels structure, update path of the adequate structure
					pathes_tmp[structure_index - ID_STRUCT_FIRST]++;
				}
				else// if (structure_index == 3)
				{
					//if the actual position on detector update the path of the most external structure
					pathes_tmp[0]++;
				}
			}
		}
		else
		{
			if (N == 1)
			{
				//update the temporary position (float) by the steps (float)
				x_tmp += d_x/N;
				y_tmp += d_y/N;
				z_tmp += d_z/N;

				//calculate the positon (integer)
				*x = roundf(x_tmp);
				*y = roundf(y_tmp);
				*z = roundf(z_tmp);
			}

			//chceck the structure number on the actual position
			structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

			if (structure_index >= ID_STRUCT_FIRST)
			{
				//if the actual position inside the voxels structure, update path of the adequate structure
				pathes_tmp[structure_index - ID_STRUCT_FIRST]++;
			}
			else// if (structure_index == 3)
			{
				//if the actual position on detector update the path of the most external structure
				pathes_tmp[0]++;
			}
		}
		//photon reached the next scattering event


		//distribute the length between two scattering events to the adequate structures
		//the Breshalm's 3D line length could be different than the length between the two
		//scattering events
		//thus, this procedure gives less rounding errors
		//l_Bresenhalm = 0.0f;
		for (j = 0; j < MAX_STRUCTURES; j++)
		{
			//the whole length of Breshalm's 3D line in voxels between two scattering events
			pathes[j] += pathes_tmp[j];
		}

	} // end tracking

	d_x = 0.0f;
	for (j = 0; j < MAX_STRUCTURES; j++)
	{
		d_x += pathes[j];
	}
	if (d_x > 0.0f)
	{
		for (j = 0; j < MAX_STRUCTURES; j++)
		{
			pathes[j] *= (*path)/d_x;
		}
	}


	//printf("%f\t%f\t%f\n",(*path),d_x,(*path)/d_x);


	///////////////////////////////////////NO SENS FACTORS - CALCULATE FLUORESCENCE////////////////////////////////////////////
	if (engine::shared_settings.sensitivity_factors == 0)
	{
		//for debugging
		//d_x = *path;
		//clear the lenght of Bresenchalm 3D line calculated by movePoton function
		//*path = 0.0f;
		//for each defined structures with different optical properties
		for (j = 0; j < MAX_STRUCTURES; j++)
		{
			//calculate the whole length of photon travel between the source and the detector. the length is expressed in voxels.
			//for debugging
			//pathes[j] *= 1.145f;
			//(*path) += pathes[j];
			//Wx += (engine::shared_optical_properties[j*7+3] + engine::shared_optical_properties[j*7+5])* pathes[j];

			//calculate fluorescence conversion probability at whole path from the source to the detector
			//if Pc_all > 0 photon crossed a structure with a fluorophore and the path should be taken into
			//account when calculating generation and visiting probability images
			//muafx for each structure * length of photon path in each structure
			Pc_all += engine::shared_optical_properties[j * NUM_OPTICAL_PROPERTIES + IDX_MUAFX] * pathes[j];


			//calculate the whole time [ps] of photon travel between the source and the detector.
			//path in structure with the same opt. prop. [voxels] * voxel_size [mm] * refractive index of the structure [-] / light_speed[mm/ps]
			tau_tmp += pathes[j] * engine::shared_settings.vox_size * engine::shared_optical_properties[j * NUM_OPTICAL_PROPERTIES] / 0.299792458f; //[ps]
		}

		//calculate the index/time window/slot where the weight should be put
		//photon travel time [ps] * number of bins [-] / max_time_in_DTOF [ps]
		ind_DTOF = (int)(roundf(tau_tmp * (float)(engine::shared_settings.numDTOF / 2) / engine::shared_settings.DTOFmax));

		//for debugging
		//printf("%f\n",d_x - *path);

		//Breshalm's 3D line between each scattering events
		//It is necessary to repeat the procedure for generation probability calculation
		//See previous comments for Breshalm's 3D algorithm description.
		for (i = 0; i < *counter; i++)
		{
			//set weights to zero
			Wx = 0.0f;
			Wm = 0.0f; //fluorescence weight
			Pc = 0.0f; //generation probability


			for (j = 0; j < MAX_STRUCTURES; j++)
			{
				pathes_tmp[j] = 0;
			}

			//Breshalm's 3D line
			d_x = (float)(*d_x_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_x_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			d_y = (float)(*d_y_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_y_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			d_z = (float)(*d_z_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_z_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];

			N = 0.0f;
			if (fabsf(d_x) > N)
				N = fabsf(d_x);
			if (fabsf(d_y) > N)
				N = fabsf(d_y);
			if (fabsf(d_z) > N)
				N = fabsf(d_z);

			*x = (*d_x_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			*y = (*d_y_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			*z = (*d_z_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];

			structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;
			if (structure_index >= ID_STRUCT_FIRST)
			{
				pathes_tmp[structure_index-4]++;
				//Generation probability between two consecutive scattering events.
				//Pc += engine::shared_optical_properties[(structure_index-4) * 7 + 5];
				Pc += 1 - __expf(-engine::shared_optical_properties[(structure_index - ID_STRUCT_FIRST) * NUM_OPTICAL_PROPERTIES + IDX_MUAFX]);
			}
			else if ((structure_index == ID_STRUCT_SOURCE) || (structure_index == ID_STRUCT_DETECTOR))
			{
				pathes_tmp[0]++;
				//Generation probability between two consecutive scattering events.
				Pc += 1- __expf(-engine::shared_optical_properties[IDX_MUAFX]);
			}

			x_tmp = (float)*x;
			y_tmp = (float)*y;
			z_tmp = (float)*z;

			if (N > 1)
			{
				for (j = 1; j < (int)N; j++)
				{
					x_tmp += d_x/N;
					y_tmp += d_y/N;
					z_tmp += d_z/N;

					*x = roundf(x_tmp);
					*y = roundf(y_tmp);
					*z = roundf(z_tmp);

					structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

					if (structure_index >= ID_STRUCT_FIRST)
					{
						pathes_tmp[structure_index - ID_STRUCT_FIRST]++;
						//Generation probability between two consecutive scattering events.
						Pc += 1- __expf(-engine::shared_optical_properties[(structure_index - ID_STRUCT_FIRST) * NUM_OPTICAL_PROPERTIES + IDX_MUAFX]);
					}
					else if (structure_index == ID_STRUCT_DETECTOR)
					{
						pathes_tmp[0]++;
						//Generation probability between two consecutive scattering events.
						Pc += 1 - __expf(-engine::shared_optical_properties[IDX_MUAFX]);
					}

				}
			}
			else
			{
				if (N == 1)
				{
					x_tmp += d_x/N;
					y_tmp += d_y/N;
					z_tmp += d_z/N;

					*x = roundf(x_tmp);
					*y = roundf(y_tmp);
					*z = roundf(z_tmp);
				}
				//chceck the structure number on the actual position
				structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

				if (structure_index >= ID_STRUCT_FIRST)
				{
					//if the actual position inside the voxels structure, update path of the adequate structure
					pathes_tmp[structure_index - ID_STRUCT_FIRST]++;
					Pc += 1- __expf(-engine::shared_optical_properties[(structure_index - ID_STRUCT_FIRST) * NUM_OPTICAL_PROPERTIES + IDX_MUAFX]);
				}
				else// if (structure_index == 3)
				{
					//if the actual position on detector update the path of the most external structure
					pathes_tmp[0]++;
					Pc += 1 - __expf(-engine::shared_optical_properties[IDX_MUAFX]);
				}
			}
			//photon reached the next scattering event

			//printf("%f\n",Pc);
			//Generation probability between two consecutive scattering events.
			//Pc = 1 - __expf(-Pc);

			//Calculate reflectance and fluorescence weights for the actual scattering event
			for (j = 0; j < MAX_STRUCTURES; j++)
			{
				pathes_partial[j] += pathes_tmp[j];

				//reflectance weight, l_real just for calculation purposes
				Wx += (engine::shared_optical_properties[j * NUM_OPTICAL_PROPERTIES + IDX_MUAX] + engine::shared_optical_properties[j * NUM_OPTICAL_PROPERTIES + IDX_MUAFX]) * pathes_partial[j];
				//fluorescence weight
				Wm += (engine::shared_optical_properties[j * NUM_OPTICAL_PROPERTIES + IDX_MUAM] + engine::shared_optical_properties[j * NUM_OPTICAL_PROPERTIES + IDX_MUAFM])*
						(pathes[j] - pathes_partial[j]);
			}

			//reflectance weight for the actual photon position
			Wx = __expf(-Wx);


			Pc = Pc * 0.03;
			//fluorescence weight for the actual scattering event
			Wm = Wx * Pc * __expf(-Wm);


			if (ind_DTOF < (engine::shared_settings.numDTOF/2))
			{

				atomicAdd(&((*d_DTOFs)[ind_DTOF + engine::shared_settings.numDTOF/2]), Wm);

				//for (j = ind_DTOF; j < engine::shared_settings.numDTOF/2; j++)
				//{
				//	atomicAdd(&((*d_DTOFs)[j + engine::shared_settings.numDTOF/2]), Wm * __expf(-(j-ind_DTOF)*(engine::shared_settings.DTOFmax / ((float)(engine::shared_settings.numDTOF / 2)))/1000));
				//}

			}


			//Wm = Wx * Pc * __expf(-Wm) * 800 * (1 - __expf(-3*800*engine::next_rand(localState)/800));
			//fluorescence weight up to the actual scattering event
			//Wm_all += Wm;


			//printf("%f\n",Wm);


			///////////////////////////////VOXELS UPDATE////////////////////////////////////////////////////



			//do not waste time for voxels update when simulating ICG bolus or sensitivity factors
			//this voxels update procedures add the same reflectance or fluorescence weight to voxels located on 3D Breshalm's line
			//between two consecutive scattering events
			if (engine::shared_settings.voxels_update > 0)
			{
				//do the Breshalm's 3D line algorithm once again
				*x = (*d_x_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
				*y = (*d_y_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
				*z = (*d_z_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];

				x_tmp = (float)*x;
				y_tmp = (float)*y;
				z_tmp = (float)*z;

				if (N > 0)
				{
					for (j = 1; j <= N; j++)
					{
						x_tmp += d_x/N;
						y_tmp += d_y/N;
						z_tmp += d_z/N;

						*x = roundf(x_tmp);
						*y = roundf(y_tmp);
						*z = roundf(z_tmp);

						//REFLECTANCE ONLY
						if (engine::shared_settings.voxels_update == 1)
						{
							engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
									Wx);
							//add new reflectance weight to the old value
							//weight_tmp += Wx;
						}
						//GENERATION PROBABILITY
						//(Pc > 0) means that there is some probability of fluorescence photon generation between two consecutive scattering events
						else if ((engine::shared_settings.voxels_update == 2) && (Pc > 0))
						{
							//get the actual structure index
							structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

							//e.g. one of the two consecutive scattering events can be located inside a structure with fluorophore and the second one
							//inside a structure without fluorophore. Thus, check each voxel between two consecutive scattering events if there is a fluorophore
							//and update only woxels with the fluorophore.
							if ((structure_index > 3) && (engine::shared_optical_properties[(structure_index-4) * 7 +5] > 0.0f))
							{
								//add new generation probability to the old value
								engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
										Pc);
								//weight_tmp += Pc;
							}

						}
						//VISITING PROBABILITY
						//(Pc_all > 0) means that photon reached a structure with fluorophore
						else if ((engine::shared_settings.voxels_update == 3) && (Pc_all > 0))
						{
							//add to the old value the new fluorescence weight if not zero, otherwise add reflectance weight
							if (Wm > 0)
							{
								engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
										Wm);
								//weight_tmp += Wm;
							}
							else
							{
								engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
										Wx);
								//weight_tmp += Wx;
							}
						}
					}
				}
			}
		}
	}
	else
	{

		////////////////////////////////////////////SENSITIVITY FACTORS///////////////////////////

		//for debugging
		//d_x = *path;
		//clear the lenght of Bresenchalm 3D line calculated by movePoton function
		//*path = 0.0f;
		//for each defined structures with different optical properties
		for (j = 0; j < 12; j++)
		{
			//calculate the whole length of photon travel between the source and the detector. the length is expressed in voxels.
			//for debugging
			//pathes[j] *= 1.145f;
			//whole path
			//(*path) += pathes[j];
			//reflectance weight
			Wx += (engine::shared_optical_properties[j*7+3] + engine::shared_optical_properties[j*7+5])* pathes[j];

		}
		//*path *= engine::shared_settings.vox_size;
		Wx = __expf(-Wx);

		//printf("%f\t%f\t%f\t%f\t%f\n", d_x*0.3,*path,pathes_partial[0],*path/pathes_partial[0],expf(-d_x*0.3 * (0.01+0.0002)));

		//printf("%f\t%f\t%f\t%f\n",(*path * engine::shared_settings.vox_size), pathes_partial[0], ((*path * engine::shared_settings.vox_size) / pathes_partial[0]), Wx_all);

		for (i = 0; i < *counter; i++)
		{
			//Breshalm's 3D line
			//calculate coordinates differences in x, y and z directions between actual and the next scattering events
			d_x = (float)(*d_x_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_x_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			d_y = (float)(*d_y_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_y_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			d_z = (float)(*d_z_coordinates)[i + 1 + engine::shared_settings.scatering_coordinates_number * *thread] - (float)(*d_z_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];

			//calculate number of steps N in the longest direction
			N = 0.0f;
			if (fabsf(d_x) > N)
				N = fabsf(d_x);
			if (fabsf(d_y) > N)
				N = fabsf(d_y);
			if (fabsf(d_z) > N)
				N = fabsf(d_z);

			//start position (integer) - the scattering event position
			*x = (*d_x_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			*y = (*d_y_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];
			*z = (*d_z_coordinates)[i + engine::shared_settings.scatering_coordinates_number * *thread];

			//temporary position (float)
			x_tmp = (float)*x;
			y_tmp = (float)*y;
			z_tmp = (float)*z;

			if (N > 0)
			{
				for (j = 1; j <= N; j++)
				{
					x_tmp += d_x/N;
					y_tmp += d_y/N;
					z_tmp += d_z/N;

					*x = roundf(x_tmp);
					*y = roundf(y_tmp);
					*z = roundf(z_tmp);

					///////MPP//////////
					if (engine::shared_settings.sensitivity_factors == 1)
					{
						//1.1432 - FACTOR!!!!!!!!!
						//actual mean path in cube voxel is 1.1432 * vox_size
						engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
								engine::shared_settings.vox_size * 1.1432f * Wx);
						//weight_tmp += engine::shared_settings.vox_size * Wx_all;
						//weight_tmp += (1 / *path) * Wx_all;
					}

					///////MTSF//////////
					if (engine::shared_settings.sensitivity_factors == 2)
					{
						structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;
						if (structure_index > 3)
						{
							engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
									(*path) * Wx * engine::shared_settings.vox_size *
									engine::shared_settings.vox_size * 1.1432f * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f);
							//weight_tmp += (*path) * Wx_all * engine::shared_settings.vox_size *
							//			  engine::shared_settings.vox_size * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f;
						}
						else
						{
							engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
									(*path) * Wx * engine::shared_settings.vox_size *
									engine::shared_settings.vox_size * 1.1432f * engine::shared_optical_properties[0] / 0.299792458f);
							//weight_tmp += (*path) * Wx_all * engine::shared_settings.vox_size *
							//			  engine::shared_settings.vox_size * engine::shared_optical_properties[0] / 0.299792458f;
						}
					}

					///////VSF//////////
					if (engine::shared_settings.sensitivity_factors == 3)
					{
						structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;
						if (structure_index > 3)
						{
							engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
									(*path) * (*path) * Wx * engine::shared_settings.vox_size *
									engine::shared_settings.vox_size * engine::shared_settings.vox_size * 1.1432f *
									engine::shared_optical_properties[(structure_index-4)*7] * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f / 0.299792458f);
							//weight_tmp += (*path) * (*path) * Wx_all * engine::shared_settings.vox_size *
							//			  engine::shared_settings.vox_size * engine::shared_settings.vox_size *
							//		 	  engine::shared_optical_properties[(structure_index-4)*7] * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f / 0.299792458f;
						}
						else
						{
							engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
									(*path) * (*path) * Wx * engine::shared_settings.vox_size *
									engine::shared_settings.vox_size * engine::shared_settings.vox_size * 1.1432f *
									engine::shared_optical_properties[0] * engine::shared_optical_properties[0] / 0.299792458f / 0.299792458f);
							//weight_tmp += (*path) * (*path) * Wx_all * engine::shared_settings.vox_size *
							//			  engine::shared_settings.vox_size * engine::shared_settings.vox_size *
							//			  engine::shared_optical_properties[0] * engine::shared_optical_properties[0] / 0.299792458f / 0.299792458f;
						}
					}
				}
			}
			else
			{

				///////MPP//////////
				if (engine::shared_settings.sensitivity_factors == 1)
				{
					engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
							engine::shared_settings.vox_size * 1.1432f * Wx);
					//weight_tmp += engine::shared_settings.vox_size * Wx_all;
					//weight_tmp += (1 / *path) * Wx_all;
				}

				///////MTSF//////////
				if (engine::shared_settings.sensitivity_factors == 2)
				{
					structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;
					if (structure_index > 3)
					{
						engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
								(*path) * Wx * engine::shared_settings.vox_size *
								engine::shared_settings.vox_size * 1.1432f * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f);
						//weight_tmp += (*path) * Wx_all * engine::shared_settings.vox_size *
						//			  engine::shared_settings.vox_size * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f;
					}
					else
					{
						engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
								(*path) * Wx * engine::shared_settings.vox_size *
								engine::shared_settings.vox_size * engine::shared_optical_properties[0] / 0.299792458f);
						//weight_tmp += (*path) * Wx_all * engine::shared_settings.vox_size *
						//			  engine::shared_settings.vox_size * engine::shared_optical_properties[0] / 0.299792458f;
					}
				}

				///////VSF//////////
				if (engine::shared_settings.sensitivity_factors == 3)
				{
					structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;
					if (structure_index > 3)
					{
						engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
								(*path) * (*path) * Wx * engine::shared_settings.vox_size *
								engine::shared_settings.vox_size * engine::shared_settings.vox_size * 1.1432f *
								engine::shared_optical_properties[(structure_index-4)*7] * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f / 0.299792458f);
						//weight_tmp += (*path) * (*path) * Wx_all * engine::shared_settings.vox_size *
						//			  engine::shared_settings.vox_size * engine::shared_settings.vox_size *
						//		 	  engine::shared_optical_properties[(structure_index-4)*7] * engine::shared_optical_properties[(structure_index-4)*7] / 0.299792458f / 0.299792458f;
					}
					else
					{
						engine::atomicAddFloat28ToFloat32(&((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z]),
								(*path) * (*path) * Wx * engine::shared_settings.vox_size *
								engine::shared_settings.vox_size * engine::shared_settings.vox_size * 1.1432f *
								engine::shared_optical_properties[0] * engine::shared_optical_properties[0] / 0.299792458f / 0.299792458f);
						//weight_tmp += (*path) * (*path) * Wx_all * engine::shared_settings.vox_size *
						//			  engine::shared_settings.vox_size * engine::shared_settings.vox_size *
						//			  engine::shared_optical_properties[0] * engine::shared_optical_properties[0] / 0.299792458f / 0.299792458f;
					}
				}

			}
		}
	}

	//printf("%f\t%f\t%f\t%f\t%f\n",*path,pathes_partial[0],pathes_partial[2],pathes_partial[1]+pathes_partial[2],*path /(pathes_partial[1]+pathes_partial[2]));

	///DTOFs only when not in the CW MULTIPLE DETECTORS mode


	//consider different n in different voxels
	d_x = *path;
	*path = 0.0f;
	for (j = 0; j < 12; j++)
	{
		//calculate the whole time [ps] of photon travel between the source and the detector.
		//path in structure with the same opt. prop. [voxels] * voxel_size [mm] * refractive index of the structure [-] / light_speed[mm/ps]
		(*path) += pathes[j] * engine::shared_settings.vox_size * engine::shared_optical_properties[j*7] / 0.299792458f; //[ps]
	}

	//calculate the index/time window/slot where the weight should be put
	//photon travel time [ps] * number of bins [-] / max_time_in_DTOF [ps]
	ind_DTOF = (int)(roundf((*path) * (float)(engine::shared_settings.numDTOF / 2) / engine::shared_settings.DTOFmax));

	//printf("%f\t%f\t%d\t%f\t%f\n",(d_x * 0.3 * 1.4f / 0.299792458f) * (float)(engine::shared_settings.numDTOF / 2) / engine::shared_settings.DTOFmax,
	//					  (*path) * (float)(engine::shared_settings.numDTOF / 2) / engine::shared_settings.DTOFmax,
	//					  ind_DTOF, Wx, Wm_all);

	//if the index lower than the number of bins in the DTOF
	//half of the DTOF array holds reflectance and the other half fluorescence, thus there should be the division by 2
	//if ((ind_DTOF > 0) && (ind_DTOF < (engine::shared_settings.numDTOF/2)))
	//if (ind_DTOF == 0)
	//	printf("\n%d\t%f\t%f\t%f\n",ind_DTOF, d_x, *path, Wx);






	/************************************************/	
	//boundary fresnel reflection
	bool isReflect = 0;

	if (isReflect){

		//first calculate normal vector

		float& x_mass = d_x;
		float& y_mass = d_y;
		float& z_mass = d_z;
		float& mass = N;
		float& d = tau_tmp;
		float& det_norm_x = x_tmp;
		float& det_norm_y = y_tmp;
		float& det_norm_z = z_tmp;
		//int structure_index;
		for (int ind_x = *x - 1; ind_x <= *x + 1; ind_x++)
		{
			for (int ind_y = *y - 1; ind_y <= *y + 1; ind_y++)
			{
				for (int ind_z = *z - 1; ind_z <= *z + 1; ind_z++)
				{
					//if ((ind_x < 0) || (ind_x >= engine::shared_settings.x_dim) || (ind_y < 0) || (ind_y >= engine::shared_settings.y_dim) || (ind_z < 0) || (ind_z >= engine::shared_settings.z_dim))
					//	printf("%d\t%d\t%d\t\t%d\t%d\t%d\n",ind_x,ind_y,ind_z,engine::shared_settings.x_dim,engine::shared_settings.y_dim,engine::shared_settings.z_dim);

					structure_index = (*d_voxels)[ind_x + engine::shared_settings.x_dim*ind_y + engine::shared_settings.x_dim*engine::shared_settings.y_dim*ind_z] & 0x0F;
					x_mass += float(structure_index * ind_x);
					y_mass += float(structure_index * ind_y);
					z_mass += float(structure_index * ind_z);
					mass += float(structure_index);
				}
			}
		}

		if (mass != 0.0f)
		{
			x_mass /= mass;
			z_mass /= mass;
			y_mass /= mass;
			d = __frsqrt_rn((x_mass - float(*x))*(x_mass - float(*x)) + (y_mass - float(*y))*(y_mass - float(*y)) + (z_mass - float(*z))*(z_mass - float(*z)));

			det_norm_x = (x_mass - float(*x)) * d;
			det_norm_y = (y_mass - float(*y)) * d;
			det_norm_z = (z_mass - float(*z)) * d;
			//printf("%f\t%f\t%f\t%f\n",*x_cos,*y_cos,*z_cos,__fsqrt_rn(*x_cos * *x_cos  + *y_cos * *y_cos + *z_cos * *z_cos));
		}
		else
		{
			//det_norm_x = *x_cos;
			//det_norm_y = *y_cos;
			//det_norm_z = *z_cos;
			det_norm_x = 0;
			det_norm_y = 0;
			det_norm_z = -1;
		}

		////fix for slab
		//det_norm_x = 0;
		//det_norm_y = 0;
		//det_norm_z = -1;


		//incidence cos
		float& cos_incidence = pathes_tmp[0];
		cos_incidence = det_norm_x * *x_cos + det_norm_y * *y_cos + det_norm_z * *z_cos;
		// n incidence 
		float& n_incidence = pathes_tmp[1];
		n_incidence = engine::shared_optical_properties[0*7 + 0];
		// n air = 1

		if(n_incidence == 1) { /** matched boundary. **/
			
		}
		else if(cos_incidence > (1.0 - 1.0e-12)) { /** normal incidence. **/
			Wx *= 1 - ((1 - n_incidence)/(1 + n_incidence)) * ((1 - n_incidence)/(1 + n_incidence));
		}
		else if(cos_incidence < 1.0e-6)  {	/** very slanted. **/
			Wx = 0;
		}
		else  {			  		/** general. **/
			//double sa1, sa2; /* sine of incident and transmission angles. */
			//double ca2;      /* cosine of transmission angle. */
			float& sin_incidence = pathes_tmp[2];
			sin_incidence = sqrt(1 - cos_incidence * cos_incidence);
			float& sin_transmit = pathes_tmp[3];
			sin_transmit = n_incidence * sin_incidence/1;
			if(sin_transmit >= 1.0) {	
				/* double check for total internal reflection. */
				Wx = 0;
			}
			else {
				//double cap, cam;	/* cosines of sum ap or diff am of the two */
				/* angles: ap = a1 + a2, am = a1 - a2. */
				//double sap, sam;	/* sines. */
				float& cos_transmit = pathes_tmp[4]; cos_transmit = sqrt(1 - sin_transmit * sin_transmit);
				float& cos_ap = pathes_tmp[5]; cos_ap = cos_incidence*cos_transmit - sin_incidence*sin_transmit; /* c+ = cc - ss. */
				float& cos_am = pathes_tmp[6]; cos_am = cos_incidence*cos_transmit + sin_incidence*sin_transmit; /* c- = cc + ss. */
				float& sin_ap = pathes_tmp[7]; sin_ap = sin_incidence*cos_transmit + cos_incidence*sin_transmit; /* s+ = sc + cs. */
				float& sin_am = pathes_tmp[8]; sin_am = sin_incidence*cos_transmit - cos_incidence*sin_transmit; /* s- = sc - cs. */
				
				Wx *= 1 - 0.5*sin_am*sin_am*(cos_am*cos_am+cos_ap*cos_ap)/(sin_ap*sin_ap*cos_am*cos_am);
			}
		}



	}



	/************************************************/	



	if (ind_DTOF < (engine::shared_settings.numDTOF/2))
	{
		//__threadfence_block();

		atomicAdd(&((*d_DTOFs)[ind_DTOF]), Wx);
		//atomicAdd(&(engine::shared_DTOFs[ind_DTOF]), Wx);

		//__threadfence_block();
		//engine::shared_DTOFs[ind_DTOF] += Wx_all; // add reflectance
		//engine::shared_DTOFs[ind_DTOF + engine::shared_settings.numDTOF/2] += Wm_all; //add fluorescence
		//atomicAdd(&(engine::shared_DTOFs[ind_DTOF + engine::shared_settings.numDTOF/2]), Wm_all);
		//atomicAdd(&((*d_DTOFs)[ind_DTOF + engine::shared_settings.numDTOF/2]), Wm_all);

	}
	//engine::shared_DTOFs[0]++;


}


__device__ void engine::atomicAddFloat28ToFloat32(unsigned int* address, float val)
{
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;

	//values used in reinterpreting unsigned int to float28
	float weight_tmp;
	unsigned char *voxel_weight;
	unsigned char *voxel_weight_tmp;


	do {

		assumed = old;

		//clock_t start_time = clock(); 

		//get voxel weight as unsigned chars table
		voxel_weight = reinterpret_cast<unsigned char *>(&old);
		//get voxel temporary weight as unsigned chars table
		voxel_weight_tmp = reinterpret_cast<unsigned char *>(&weight_tmp);
		//copy voxel weight to temporary voxel weight (in fact, it is a cast of types unsigned int -> float but without rounding.
		//instead it copies bytes from uint to float). 4 oldest bits with structure number are set to 0.
		voxel_weight_tmp[0] = voxel_weight[0] & 0xF0; voxel_weight_tmp[1] = voxel_weight[1];
		voxel_weight_tmp[2] = voxel_weight[2]; voxel_weight_tmp[3] = voxel_weight[3];

		weight_tmp += val;

		//copy back new voxel reflectance weight to temporary voxel value (this value has 4 oldest bits set according to the structure number)
		//clear 4 oldest bits in the float value (prepare space to copy back the structure number)
		voxel_weight_tmp[0] &= 0xF0;
		//copy the structure number
		voxel_weight_tmp[0] |= (voxel_weight[0] & 0x0F);
		//copy the amended weight to the temporary voxel
		voxel_weight[0] = voxel_weight_tmp[0]; voxel_weight[1] = voxel_weight_tmp[1];
		voxel_weight[2] = voxel_weight_tmp[2]; voxel_weight[3] = voxel_weight_tmp[3];

		//clock_t stop_time = clock();

		//printf("%ld\n",(long)(stop_time - start_time));

		//put back the amended voxel to the voxel 3D structure

		old = atomicCAS(address_as_int, assumed, __float_as_int(weight_tmp));

	} while (assumed != old);
}

__global__ void engine::simulate(Settings *d_settings, curandStateMRG32k3a *globalState, float *d_DTOFs, unsigned int *d_voxels,
								 int *d_source_x_coordinates, int *d_source_y_coordinates, int *d_source_z_coordinates,
								 float *d_optical_properties, int *d_x_coordinates,int *d_y_coordinates, int *d_z_coordinates)
{
	//actual thread number
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//calculation parameters
	int ind = 0;
	int size = 0;

	//if the first thread, set the shared GPU memory
	if (threadIdx.x == 0)
	{
		//assign statically allocated shared memory
		shared_settings = *d_settings;

		//assign dynamically allocated shared memory
		//assign the first shared array to the 'array' variable
		shared_source_x_coordinates = (int*)array;
		size = shared_settings.source_coordinates_number;
		//copy values from GPU global memory
		for (ind = 0; ind < size; ind++)
		{
			shared_source_x_coordinates[ind] = d_source_x_coordinates[ind];
		}

		//assign the second shared variable to the end of the first shared variable
		shared_source_y_coordinates = (int*)&shared_source_x_coordinates[size];
		for (ind = 0; ind < size; ind++)
		{
			//copy values from GPU global memory
			shared_source_y_coordinates[ind] = d_source_y_coordinates[ind];
		}

		//assign the third shared variable to the end of the second shared variable
		shared_source_z_coordinates = (int*)&shared_source_y_coordinates[size];
		for (ind = 0; ind < size; ind++)
		{
			//copy values from GPU global memory
			shared_source_z_coordinates[ind] = d_source_z_coordinates[ind];
		}

		/*
		//assign the fourth shared variable to the end of the third shared variable
		shared_DTOFs = (float*)&shared_source_z_coordinates[size];
		size = shared_settings.numDTOF;
		for (ind = 0; ind < size; ind++)
		{
			//copy values from GPU global memory
			shared_DTOFs[ind] = 0.0f;
		}
		*/

		//assign the fitht shared variable to the end of the fourth shared variable
		shared_optical_properties = (float*)&shared_source_z_coordinates[size];//(float*)&shared_DTOFs[size];
		size = 12*7;
		for (ind = 0; ind < size; ind++)
		{
			//copy values from GPU global memory
			shared_optical_properties[ind] = d_optical_properties[ind];
		}
	}
	//Important: all threads in block should wait while the first thread sets the shared memory!
	__syncthreads();

	//copy the actual thread random number generator from the GPU global memory
	curandStateMRG32k3a_t localState = globalState[i];

	//Create photon object
	Photon photon = Photon();

	//set the scattering coordinates
	for(ind = 0; ind < engine::shared_settings.scatering_coordinates_number; ind++)
	{
		d_x_coordinates[ind + engine::shared_settings.scatering_coordinates_number * i] = 0;
		d_y_coordinates[ind + engine::shared_settings.scatering_coordinates_number * i] = 0;
		d_z_coordinates[ind + engine::shared_settings.scatering_coordinates_number * i] = 0;
	}


	int x;
	int y;
	int z;
	int counter = 0;
	float weight = 1.0f;
	float path = 0.0f;
	bool first = 1;
	bool detected = 0;
	float cost, sint, cosp, sinp, x_cos, y_cos, z_cos;

	photon.setStartXYZ(&i, &d_voxels, &localState, &x, &y, &z, &counter, &d_x_coordinates, &d_y_coordinates, &d_z_coordinates, &x_cos, &y_cos, &z_cos);



	//do the simulation, while the photon is 'alive' - the weight is nonzero
	while (weight > 0.0f)
	{
		//break;
		photon.getScatteringAngle(&d_voxels, &x, &y, &z, &localState, &cost);
		photon.getAzimuthalAngle(&localState, &cost,&sint, &cosp, &sinp);
		//move photon to the next scattering event
		photon.movePhoton(&i, &d_voxels, &x, &y, &z, &localState, &weight, &path, &first, &detected, &counter,
						  &d_x_coordinates, &d_y_coordinates, &d_z_coordinates, &cost, &sint, &cosp, &sinp, &x_cos, &y_cos, &z_cos);
	}


	if (detected)
	{
		photon.detect(&i, &d_voxels, &d_DTOFs, &x, &y, &z, &x_cos, &y_cos, &z_cos, &path, &counter, &d_x_coordinates, &d_y_coordinates, &d_z_coordinates, &localState);
	}


	globalState[i] = localState;


}

void SimulationEngine::cudaAllocateCopy2Device(int blocks, int threads_per_block, int device, Settings *settings, float **optical_properties, unsigned int **voxels, int **source_x_coordinates, int **source_y_coordinates, int **source_z_coordinates)
{
	this->settings = *settings;

	this->optical_properties = *optical_properties;

	this->source_x_coordinates = new int[this->settings.source_coordinates_number];
	this->source_y_coordinates = new int[this->settings.source_coordinates_number];
	this->source_z_coordinates = new int[this->settings.source_coordinates_number];

	this->source_x_coordinates = *source_x_coordinates;
	this->source_y_coordinates = *source_y_coordinates;
	this->source_z_coordinates = *source_z_coordinates;

	//set the active GPU
	CUDA_CHECK_RETURN(cudaSetDevice(device));

	this->blocks = blocks;
	this->threads_per_block = threads_per_block;
	//allocate and copy simulation settings
	size_t size_settings = sizeof(Settings);
	this->d_settings = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_settings, size_settings));
	CUDA_CHECK_RETURN(cudaMemcpy(d_settings, &this->settings, size_settings, cudaMemcpyHostToDevice));

	//allocate randomizer parameters, seed is copied elsewhere, globalState is calculated inside a kernel function engine::set_randomizer(...)
	this->globalState=NULL;
	this->seed_ptr = NULL;
	this->sequence = this->blocks * this->threads_per_block;
	this->d_sequence = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&globalState , this->blocks * this->threads_per_block * sizeof(curandStateMRG32k3a_t)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&seed_ptr , sizeof(unsigned long long int)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_sequence , sizeof(unsigned long long int)));

	//allocate and copy results and optical parameters
	//DTOFs
	this->size_DTOFs = sizeof(float) * this->settings.numDTOF;
	this->d_DTOFs = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_DTOFs, size_DTOFs));
	CUDA_CHECK_RETURN(cudaMemcpy(d_DTOFs, DTOFs, size_DTOFs, cudaMemcpyHostToDevice));

	//voxels
	this->voxels = *voxels;
	this->size_voxels = sizeof(unsigned int) * this->settings.x_dim * this->settings.y_dim * this->settings.z_dim;
	this->d_voxels = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_voxels, size_voxels));
	CUDA_CHECK_RETURN(cudaMemcpy(d_voxels, this->voxels, size_voxels, cudaMemcpyHostToDevice));

	//source x coordinates
	this->size_source_coordinates = sizeof(int) * this->settings.source_coordinates_number;
	this->d_source_x_coordinates = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_source_x_coordinates, size_source_coordinates));
	CUDA_CHECK_RETURN(cudaMemcpy(d_source_x_coordinates, this->source_x_coordinates, size_source_coordinates, cudaMemcpyHostToDevice));
	//source y coordinates
	this->d_source_y_coordinates = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_source_y_coordinates, size_source_coordinates));
	CUDA_CHECK_RETURN(cudaMemcpy(d_source_y_coordinates, this->source_y_coordinates, size_source_coordinates, cudaMemcpyHostToDevice));
	//source z coordinates
	this->d_source_z_coordinates = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_source_z_coordinates, size_source_coordinates));
	CUDA_CHECK_RETURN(cudaMemcpy(d_source_z_coordinates, this->source_z_coordinates, size_source_coordinates, cudaMemcpyHostToDevice));

	//optical properties
	this->size_optical_properties = sizeof(float) * 12 * 7;
	this->d_optical_properties = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_optical_properties, size_optical_properties));
	CUDA_CHECK_RETURN(cudaMemcpy(d_optical_properties, this->optical_properties, size_optical_properties, cudaMemcpyHostToDevice));


	//scattering events coordinates, calculated inside the kernel
	this->d_x_coordinates=NULL;
	this->d_y_coordinates=NULL;
	this->d_z_coordinates=NULL;
	this->size_scattering_coordinates =sizeof(int) * this->settings.scatering_coordinates_number * blocks * threads_per_block;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_x_coordinates , size_scattering_coordinates));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_y_coordinates , size_scattering_coordinates));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_z_coordinates , size_scattering_coordinates));


	//set the size of dynamically allocated shared GPU memory
	this->shared_dyn_alloc_bytes = this->size_DTOFs + 3 * this->size_source_coordinates + this->size_optical_properties;
}


void SimulationEngine::cudaCopy2Device_EmDetPairs(Settings *settings, float **optical_properties, unsigned int **voxels,  int **source_x_coordinates, int **source_y_coordinates, int **source_z_coordinates)
{
	this->settings = *settings;
	CUDA_CHECK_RETURN(cudaMemcpy(d_settings, &this->settings, sizeof(Settings), cudaMemcpyHostToDevice));

	//optical properties
	this->optical_properties = *optical_properties;
	CUDA_CHECK_RETURN(cudaMemcpy(d_optical_properties, this->optical_properties, size_optical_properties, cudaMemcpyHostToDevice));

	/*
	for (int i = 0; i < this->settings.numDTOF; i++)
	{
		DTOFs[i] = 0.0f;
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_DTOFs, DTOFs, size_DTOFs, cudaMemcpyHostToDevice));
	*/

	for (int i = 0; i < (int)(this->settings.x_dim*this->settings.y_dim*this->settings.z_dim); i++)
	{
		this->voxels[i] = (*voxels)[i];
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_voxels, this->voxels, size_voxels, cudaMemcpyHostToDevice));


	//source x coordinates
	CUDA_CHECK_RETURN(cudaFree(d_source_x_coordinates));
	this->source_x_coordinates = *source_x_coordinates;
	this->size_source_coordinates = sizeof(int) * this->settings.source_coordinates_number;
	this->d_source_x_coordinates = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_source_x_coordinates, size_source_coordinates));
	CUDA_CHECK_RETURN(cudaMemcpy(d_source_x_coordinates, this->source_x_coordinates, size_source_coordinates, cudaMemcpyHostToDevice));
	//source y coordinates
	CUDA_CHECK_RETURN(cudaFree(d_source_y_coordinates));
	this->source_y_coordinates = *source_y_coordinates;
	this->d_source_y_coordinates = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_source_y_coordinates, size_source_coordinates));
	CUDA_CHECK_RETURN(cudaMemcpy(d_source_y_coordinates, this->source_y_coordinates, size_source_coordinates, cudaMemcpyHostToDevice));
	//source z coordinates
	CUDA_CHECK_RETURN(cudaFree(d_source_z_coordinates));
	this->source_z_coordinates = *source_z_coordinates;
	this->d_source_z_coordinates = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_source_z_coordinates, size_source_coordinates));
	CUDA_CHECK_RETURN(cudaMemcpy(d_source_z_coordinates, this->source_z_coordinates, size_source_coordinates, cudaMemcpyHostToDevice));

	//set the size of dynamically allocated shared GPU memory
	this->shared_dyn_alloc_bytes = this->size_DTOFs + 3 * this->size_source_coordinates + this->size_optical_properties;
}

void SimulationEngine::cudaCopy2Device_SensFactors(Settings *settings)
{
	this->settings = *settings;
	CUDA_CHECK_RETURN(cudaMemcpy(d_settings, &this->settings, sizeof(Settings), cudaMemcpyHostToDevice));

	/*
	for (int i = 0; i < this->settings.numDTOF; i++)
	{
		DTOFs[i] = 0.0f;
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_DTOFs, DTOFs, size_DTOFs, cudaMemcpyHostToDevice));

	for (int i = 0; i < this->settings.x_dim*this->settings.y_dim*this->settings.z_dim; i++)
	{
		this->voxels[i] = (*voxels)[i];
	}
	CUDA_CHECK_RETURN(cudaMemcpy(d_voxels, this->voxels, size_voxels, cudaMemcpyHostToDevice));
	*/
}


void SimulationEngine::run(unsigned long long int seed, int run, int myid, int max_GPU_size)
{
	cudaError_t error;


	//if we run for the very first time, set random number generators and clear/set results memory
	if (run == 0)
	{
		//set the seed for each GPU, if the first run
		//using the worker index 'myid' prevents from situation when multiple GPU cards within the cluster have the same sequence number

		this->seed = seed;
		//this->sequence = (myid - 1) * this->blocks * this->threads_per_block;
		//the max_GPU_size equals the 'this->blocks * this->threads_per_block' but for the biggest value within the cluster
		this->sequence = (myid - 1) * max_GPU_size;
		//cout << seed << " : " << run << " : " << myid << " : " << max_runs << " : " << this->seed << endl << flush;
		//copu seed to the global GPU memory
		CUDA_CHECK_RETURN(cudaMemcpy(seed_ptr, &this->seed, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_sequence, &this->sequence, sizeof(unsigned long long int), cudaMemcpyHostToDevice));



		thrust::device_ptr<float> d_DTOFs_ptr(d_DTOFs);
		thrust::device_ptr<unsigned int> d_voxels_ptr(d_voxels);

		//clear results on the GPU
		thrust::fill(thrust::device, d_DTOFs_ptr, d_DTOFs_ptr + this->settings.numDTOF, 0.0f);
		if (this->settings.sensitivity_factors > 0){
			//clear between consecutive sensitivity factor calculation
			thrust::for_each(thrust::device, d_voxels_ptr, d_voxels_ptr + this->settings.x_dim*this->settings.y_dim*this->settings.z_dim, clear28OldBits());
		}

		//set the random number generators
		//splitting this kernel function from the simulate kernel function speeds up the whole simulation
		engine::set_randomizer<<<this->blocks, this->threads_per_block>>>(d_settings, globalState,
																				   seed_ptr, d_sequence,
																				   d_DTOFs, d_voxels);

	}

	//printf("shared_mem: %d\n",this->shared_dyn_alloc_bytes);


	//do the simulation
	engine::simulate<<<this->blocks,
					   this->threads_per_block,
					   this->shared_dyn_alloc_bytes>>>(d_settings,
													   globalState,
													   d_DTOFs,
													   d_voxels,
													   d_source_x_coordinates,
													   d_source_y_coordinates,
													   d_source_z_coordinates,
													   d_optical_properties,
													   d_x_coordinates,
													   d_y_coordinates,
													   d_z_coordinates);

	//wait while the simulation ends
	//TO_DO - this slows down by ~9% (to check on large number of photons) but is needed to show the progress bar. Lets move this option to the setting file.
	cudaDeviceSynchronize();

	// check for error
	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error simulation kernel (MPI processor %d): %s\n", myid, cudaGetErrorString(error));
		exit(-1);
	}

}

void SimulationEngine::cudaFreeCopy2Host()
{
	//copy results from GPU to PC
	CUDA_CHECK_RETURN(cudaMemcpy(DTOFs, d_DTOFs, size_DTOFs, cudaMemcpyDeviceToHost));
	//do not waste time to copy the voxels structure when there is no need of voxels update
	if ((settings.voxels_update > 0) || (settings.sensitivity_factors > 0))
	{
		CUDA_CHECK_RETURN(cudaMemcpy(voxels, d_voxels, size_voxels, cudaMemcpyDeviceToHost));
	}

	//free the GPU memory
	CUDA_CHECK_RETURN(cudaFree(d_DTOFs));
	CUDA_CHECK_RETURN(cudaFree(d_settings));
	CUDA_CHECK_RETURN(cudaFree(d_voxels));
	CUDA_CHECK_RETURN(cudaFree(d_source_x_coordinates));
	CUDA_CHECK_RETURN(cudaFree(d_source_y_coordinates));
	CUDA_CHECK_RETURN(cudaFree(d_source_z_coordinates));
	CUDA_CHECK_RETURN(cudaFree(d_optical_properties));
	CUDA_CHECK_RETURN(cudaFree(d_x_coordinates));
	CUDA_CHECK_RETURN(cudaFree(d_y_coordinates));
	CUDA_CHECK_RETURN(cudaFree(d_z_coordinates));
	CUDA_CHECK_RETURN(cudaFree(globalState));
	CUDA_CHECK_RETURN(cudaFree(seed_ptr));
	CUDA_CHECK_RETURN(cudaFree(d_sequence));
	//set the GPU to default
	CUDA_CHECK_RETURN(cudaDeviceReset());
}

void SimulationEngine::cudaCopy2Host()
{
	//if simulating ICG bolus, copy only the DTOFs
	CUDA_CHECK_RETURN(cudaMemcpy(DTOFs, d_DTOFs, size_DTOFs, cudaMemcpyDeviceToHost));
	if ((settings.voxels_update > 0) || (settings.sensitivity_factors > 0))
	{
		CUDA_CHECK_RETURN(cudaMemcpy(voxels, d_voxels, size_voxels, cudaMemcpyDeviceToHost));
	}
	//CUDA_CHECK_RETURN(cudaMemcpy(x_coordinates, d_x_coordinates, size_scattering_coordinates, cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpy(y_coordinates, d_y_coordinates, size_scattering_coordinates, cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpy(z_coordinates, d_z_coordinates, size_scattering_coordinates, cudaMemcpyDeviceToHost));
}

//SimulationEngine::SimulationEngine(Settings *settings, unsigned int **voxels, float **optical_properties)
void SimulationEngine::initialize(Settings *settings, unsigned int **voxels, float **optical_properties)
		//: settings(*settings), voxels(*voxels), optical_properties(*optical_properties)
{
	this->settings = *settings;
	this->voxels = *voxels;
	this->optical_properties = *optical_properties;

	//allocate DTOFs and set to zero
	int size = this->settings.numDTOF;
	this->DTOFs = new float[size];
	for (int i = 0; i < size; i++)
	{
		this->DTOFs[i] = 0.0f;
	}

	size = this->settings.scatering_coordinates_number * settings->blocks * settings->threads_per_block;
	this->x_coordinates = new int[size];
	this->y_coordinates = new int[size];
	this->z_coordinates = new int[size];
	for (int i = 0; i < size; i++)
	{
		this->x_coordinates[i] = 0;
		this->y_coordinates[i] = 0;
		this->z_coordinates[i] = 0;
	}

}

/*
SimulationEngine::~SimulationEngine()
{
	//free/erase the PC memory
	delete[] DTOFs;

	delete[] voxels;

	delete[] source_x_coordinates;

	delete[] source_y_coordinates;

	delete[] source_z_coordinates;

	delete[] optical_properties;

	delete[] x_coordinates;

	delete[] y_coordinates;

	delete[] z_coordinates;
}
*/


__device__ void Photon::setStartXYZ(int *thread, unsigned int **d_voxels, curandStateMRG32k3a_t *localState, int *x, int *y, int *z, int *counter, int **d_x_coordinates, int **d_y_coordinates, int **d_z_coordinates,
									float *x_cos, float *y_cos, float *z_cos)
{
	//uniformly distributed random source voxel
	int rand_source_point = floorf(engine::next_rand(localState)*(engine::shared_settings.source_coordinates_number - 1));

	//set the photon start position to uniformly distributed random source voxel
	*x = engine::shared_source_x_coordinates[rand_source_point];
	*y = engine::shared_source_y_coordinates[rand_source_point];
	*z = engine::shared_source_z_coordinates[rand_source_point];

	//calculate first scattering direction (perpendicular to the surface at the start point position)

	//printf("%d\t%d\t%d\n",this->x+1,this->y+1,this->z+1);


	float x_mass = 0.0f;
	float y_mass = 0.0f;
	float z_mass = 0.0f;
	float mass = 0.0f;
	int structure_index;
	for (int ind_x = *x - 1; ind_x <= *x + 1; ind_x++)
	{
		for (int ind_y = *y - 1; ind_y <= *y + 1; ind_y++)
		{
			for (int ind_z = *z - 1; ind_z <= *z + 1; ind_z++)
			{
				//if ((ind_x < 0) || (ind_x >= engine::shared_settings.x_dim) || (ind_y < 0) || (ind_y >= engine::shared_settings.y_dim) || (ind_z < 0) || (ind_z >= engine::shared_settings.z_dim))
				//	printf("%d\t%d\t%d\t\t%d\t%d\t%d\n",ind_x,ind_y,ind_z,engine::shared_settings.x_dim,engine::shared_settings.y_dim,engine::shared_settings.z_dim);

				structure_index = (*d_voxels)[ind_x + engine::shared_settings.x_dim*ind_y + engine::shared_settings.x_dim*engine::shared_settings.y_dim*ind_z] & 0x0F;
				x_mass += float(structure_index * ind_x);
				y_mass += float(structure_index * ind_y);
				z_mass += float(structure_index * ind_z);
				mass += float(structure_index);
			}
		}
	}

	if (mass != 0.0f)
	{
		x_mass /= mass;
		z_mass /= mass;
		y_mass /= mass;
		float d = __frsqrt_rn((x_mass - float(*x))*(x_mass - float(*x)) + (y_mass - float(*y))*(y_mass - float(*y)) + (z_mass - float(*z))*(z_mass - float(*z)));

		*x_cos = (x_mass - float(*x)) * d;
		*y_cos = (y_mass - float(*y)) * d;
		*z_cos = (z_mass - float(*z)) * d;
		//printf("%f\t%f\t%f\t%f\n",*x_cos,*y_cos,*z_cos,__fsqrt_rn(*x_cos * *x_cos  + *y_cos * *y_cos + *z_cos * *z_cos));
	}
	else
	{
		*x_cos = 0.0f;
		*y_cos = 0.0f;
		*z_cos = -1.0f;
	}

	//*x_cos = 0.0f;
	//*y_cos = 0.0f;
	//*z_cos = -1.0f;

	//put the start photon position into the scattering events arrays
	(*d_x_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *x;
	(*d_y_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *y;
	(*d_z_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *z;
	(*counter)++;
}


__device__ void Photon::getScatteringAngle(unsigned int **d_voxels, int *x, int *y, int *z, curandStateMRG32k3a_t *localState, float *cost)
{
	float g = 0.0f;

	//get the structure number at the current photon position
	int structure_index = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

	//and set the proper anisotropy factor value
	if ((structure_index) >= ID_STRUCT_FIRST)
	{
		g = engine::shared_optical_properties[(structure_index-ID_STRUCT_FIRST) * NUM_OPTICAL_PROPERTIES + IDX_G];
	}
	else
	{
		g = engine::shared_optical_properties[IDX_G];
	}

	//calculate the scattering angle cosine according to the Henyey-Greenstein phase function
    if (g == 0.0f)
    {
        *cost = 2.0f * engine::next_rand(localState) - 1.0f;
    }
    else
    {
    	*cost = (1.0f + g * g -
        		__powf((1.0f - g * g) / (1.0f - g + 2.0f * g * engine::next_rand(localState)), 2)) /
        				(2.0f * g);
    }
}

__device__ void Photon::getAzimuthalAngle(curandStateMRG32k3a_t *localState, float *cost, float *sint, float *cosp, float *sinp)
{
	//get new scattering and azimuthal angles sines and cosines
    //scattering angle
	//float cost = cosTheta(d_voxels, x, y, z, &structure_index, localState);
    *sint = __fsqrt_rn(1.0f - *cost * *cost);
    //azimuthal angle
    float Phi = 2.0f * CUDART_PI_F * engine::next_rand(localState);
    *cosp = __cosf(Phi);

    *sinp = Phi<=CUDART_PI_F ? __fsqrt_rn(1.0f - *cosp * *cosp) : -__fsqrt_rn(1.0f - *cosp * *cosp);

}

__device__ void Photon::movePhoton(int *thread, unsigned int **d_voxels, int *x, int *y, int *z, curandStateMRG32k3a_t *localState, float *weight, float *path, bool *first, bool *detected, int *counter,
								   int **d_x_coordinates, int **d_y_coordinates, int **d_z_coordinates,
								   float *cost, float *sint, float *cosp, float *sinp, float *x_cos, float *y_cos, float *z_cos)
{
	/// Photon old position
	int x0 = 0; int y0 = 0; int z0 = 0;

	/// Photon new position
	int xn = 0; int yn = 0; int zn = 0;

	///for Bresenhalm's 3D line algorithm
	float d_x, d_y, d_z, N, x_tmp, y_tmp, z_tmp;

	/// Distance traveled by photon during one scattering event
	float l;
	/// additional variable to calculate distance traveled by photon during one scattering event
	//float l_tmp;

	/// new directional cosines calculation parameter
	//float den;
	/// Calculation parameter
	float tmp;
	///Photon directional cosine in x direction
	//float x_cos;
	/// Photon directional cosine in y direction
	//float y_cos;
	///Photon directional cosine in z direction
	//float z_cos;

	int structure_index;
	int new_vox_structure;
	int old_vox_structure;

	//a flag which stops the Bresjalm's 3D algorithm
	bool go_on = 1;

	//if the photon travel time is longer than the maximum time in the DTOF, kill the photon,
	//there is no need to track the photon which would not be added to the resulting DTOF!
	if ((*path * engine::shared_settings.vox_size / (0.299792458f / engine::shared_optical_properties[0]) > engine::shared_settings.DTOFmax) ||
			(*counter >= engine::shared_settings.scatering_coordinates_number - 1))
	//if (*counter >= engine::shared_settings.scatering_coordinates_number - 1)
	{
		//printf("too long %f\t%d\n",*path,*counter);
		//kill the photon and return
		*weight = 0.0f;
		return;
	}

    //copy current coordinates to old one
    x0 = *x;
    y0 = *y;
    z0 = *z;


    //if first photon move
    if ((*first) == 1)
    {
		//calculate the distance between two consecutive scattering event considering only the scattering coefficient mus
		l = -__logf(engine::next_rand(localState)) / (engine::shared_optical_properties[2]);

    	//first scattering direction corresponds to the normal vector of the source
		//*x_cos = engine::shared_settings.source_x_cos;
		//*y_cos = engine::shared_settings.source_y_cos;
		//*z_cos = engine::shared_settings.source_z_cos;

		//printf("%f\t%f\t%f\n",*x_cos,*y_cos,*z_cos);



		//calculate the new scattering event coordinates
		xn = x0 + (int)roundf(l * *x_cos);
		yn = y0 + (int)roundf(l * *y_cos);
		zn = z0 + (int)roundf(l * *z_cos);

		if ((xn >= 0) && (xn < engine::shared_settings.x_dim) &&
			(yn >= 0) && (yn < engine::shared_settings.y_dim) &&
			(zn >= 0) && (zn < engine::shared_settings.z_dim))
		{
			//get the structure number on the new scattering event coordinates
			structure_index = (*d_voxels)[xn + engine::shared_settings.x_dim*yn + engine::shared_settings.x_dim*engine::shared_settings.y_dim*zn] & 0x0F;
		}
		else
		{
			structure_index = 0;
		}

		//printf("%f : %f : %f : %d \n",*x_cos,*y_cos,*z_cos, structure_index);

		//to avoid problems with too short l in the first attempt to calculate new coordinates (new coordinates still in source voxel)
		//do the calculations until the new scattering events is outside the source
    	while (structure_index < 3)
    	{
    		//printf("DUPA");

			/*
			//rondomize angle in case photon is outside
    		*x_cos = 2.0f * engine::next_rand(localState) - 1.0f;
    		*y_cos = 2.0f * engine::next_rand(localState) - 1.0f;
    		*z_cos = 2.0f * engine::next_rand(localState) - 1.0f;

			tmp = __frsqrt_rn(*x_cos * *x_cos + *y_cos * *y_cos + *z_cos * *z_cos);

			(*x_cos) *= tmp;
			(*y_cos) *= tmp;
			(*z_cos) *= tmp;
			*/

			l = -__logf(engine::next_rand(localState)) / (engine::shared_optical_properties[2]);

			xn = x0 + (int)roundf(l * *x_cos);
			yn = y0 + (int)roundf(l * *y_cos);
			zn = z0 + (int)roundf(l * *z_cos);

			if ((xn >= 0) && (xn < engine::shared_settings.x_dim) &&
				(yn >= 0) && (yn < engine::shared_settings.y_dim) &&
				(zn >= 0) && (zn < engine::shared_settings.z_dim))
			{
				//get the structure number on the new scattering event coordinates
				structure_index = (*d_voxels)[xn + engine::shared_settings.x_dim*yn + engine::shared_settings.x_dim*engine::shared_settings.y_dim*zn] & 0x0F;
			}

			//give it just one chance
			if (structure_index < 3){
				*weight = 0.0f;
				return;				
			}

    	}



		//printf("%d\t%d\t%d\t%f\t%f\t%f\n",this->x,this->y,this->z,engine::shared_settings.source_x_cos,engine::shared_settings.source_y_cos,engine::shared_settings.source_z_cos);

		//photon ended its first move
		*first = 0;
		//(*path)++;

		//if ((this->xn < 0) || (this->xn >= engine::shared_settings.x_dim) ||
		//	(this->yn < 0) || (this->yn >= engine::shared_settings.y_dim) ||
		//	(this->zn < 0) || (this->zn >= engine::shared_settings.z_dim))
		//	printf("%d\t%d\t%d\t%f\t%f\t%f\n",this->xn,this->yn,this->zn,engine::shared_settings.source_x_cos,engine::shared_settings.source_y_cos,engine::shared_settings.source_z_cos);
		//printf("%d\n",this->structure_index);
    }
    else
    {//if normal photon move (not the first move)

		structure_index = ((*d_voxels)[x0 + engine::shared_settings.x_dim*y0 + engine::shared_settings.x_dim*engine::shared_settings.y_dim*z0] & 0x0F);

		//if (structure_index > 3)
		//{
		l = -__logf(engine::next_rand(localState)) / (engine::shared_optical_properties[(structure_index-4) * 7 + 2]);
		//}
		//else
		//{
		//	//printf("x0 error : %d\n",structure_index);
		//	//l = -logf(engine::next_rand(localState)) / (engine::shared_optical_properties[2]);
		//	*weight = 0.0f;
		//	return;
		//}




		//calculate the new scattering event coordinates
		xn = x0 + (int)roundf(l * *x_cos);
		yn = y0 + (int)roundf(l * *y_cos);
		zn = z0 + (int)roundf(l * *z_cos);
    }

	//Breshalm's 3D line
    //see comments inside the detect() function for the Breshalm's 3D algorithm description
	d_x = (float)(xn) - (float)(x0);
	d_y = (float)(yn) - (float)(y0);
	d_z = (float)(zn) - (float)(z0);

	N = 0.0f;
	if (fabsf(d_x) > N)
		N = fabsf(d_x);
	if (fabsf(d_y) > N)
		N = fabsf(d_y);
	if (fabsf(d_z) > N)
		N = fabsf(d_z);

	//float sx;
	//float sy;
	//float sz;

	x_tmp = (float)x0;
	y_tmp = (float)y0;
	z_tmp = (float)z0;

	//if (((*d_voxels)[(int)x_tmp + engine::shared_settings.x_dim* (int)y_tmp + engine::shared_settings.x_dim*engine::shared_settings.y_dim* (int)z_tmp] & 0x0F) == 0)
	//	printf("%d\t%f\t%d\n",((*d_voxels)[(int)x_tmp + engine::shared_settings.x_dim* (int)y_tmp + engine::shared_settings.x_dim*engine::shared_settings.y_dim* (int)z_tmp] & 0x0F), x_tmp, *x);

	//structure number at the old and new voxel lying on the Breshalm's 3D line
	//the assigned 4 is arbitrary, it could be a different number
	new_vox_structure = 4;
	old_vox_structure = 4;


	(*path) += l;

	//go from the old scattering event and try to reach the new scattering event
	while (go_on)
	{
		//(*path)++;

		//if number of scattering events is higher than the size of arrays holding the coordinates of the scattering event
		//kill the photon and break the procedure
		//the number could be high, it is a compromise to keep the GPU memory usage low
		if (*counter == engine::shared_settings.scatering_coordinates_number - 1)
		{
			//printf("too long %f\t%d\n",*path,*counter);
			//printf("%d\t%d\t%d\n",this->x,this->y,this->z);
			go_on = 0;
			*weight = 0.0f;
			return;
			//break;
		}

		//if line from x0 to xn is longer than the voxel size
		if (N > 0)
		{
			//(*path)++;
			//old_vox_structure = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;
			old_vox_structure = (*d_voxels)[(int)x_tmp + engine::shared_settings.x_dim* (int)y_tmp + engine::shared_settings.x_dim*engine::shared_settings.y_dim* (int)z_tmp] & 0x0F;

			x_tmp += d_x/N;
			y_tmp += d_y/N;
			z_tmp += d_z/N;

			//calculate the new voxel lying on the 3D line
			*x = (int)roundf(x_tmp);
			*y = (int)roundf(y_tmp);
			*z = (int)roundf(z_tmp);

			//printf("%d\t%f\t%d\t%d\n",old_vox_structure, x_tmp, *x, *first);

			//Important! if the new point extends the defined voxels structure
			//kill the photon and break the procedure
			if ((*x < 0) || (*y < 0) || (*z < 0) || (*x >= engine::shared_settings.x_dim) || (*y >= engine::shared_settings.y_dim) || (*z >= engine::shared_settings.z_dim))
			{
				//printf("\n%d\t%d\t%d\n",*x,*y,*z);
				//printf("%d\t%f\t%d\t%d\n",old_vox_structure, x_tmp, *x, *first);
				*weight = 0.0f;
				go_on = 0;
				return;
			//	//break;
			}

			new_vox_structure = (*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F;

			//inside (>3), no external boundary (0), no detector (3), no emitter (2)
			//if change of optical properties
			//(old_vox_structure != 2) do not do this when first photon move!
			if ((new_vox_structure > 3) && (new_vox_structure != old_vox_structure) && (old_vox_structure > 3))
			{

				(*path) -= l;
				//Breshalm's 3D line correction
				//do the same byt consider only the scattering coefficient mus
				l = __fsqrt_rn((*x - x0)*(*x - x0) + (*y - y0)*(*y - y0) + (*z - z0)*(*z - z0)) +
						__fsqrt_rn((xn - *x)*(xn - *x) + (yn - *y)*(yn - *y) + (zn - *z)*(zn - *z))*
						(engine::shared_optical_properties[(old_vox_structure-4)*7 + 2] / engine::shared_optical_properties[(new_vox_structure-4)*7 + 2]);

				(*path) += l;

				//if (!isfinite(l))
				//{
				//	printf("\n%d\t%d\t%d\t%d:%d:%d\t%d:%d:%d\t%d:%d:%d\n",(yn - (*y))*(yn - (*y)),old_vox_structure,new_vox_structure,x0,*x,xn,y0,*y,yn,z0,*z,zn);
				//}

				//calculate/update the new coordinates of the next scattering event corrected by the optical properties at the current voxel
				//this->xn = (float)this->x0 + roundf(this->l * this->x_cos);
				//this->yn = (float)this->y0 + roundf(this->l * this->y_cos);
				//this->zn = (float)this->z0 + roundf(this->l * this->z_cos);
				xn = x0 + (int)roundf(l * *x_cos);
				yn = y0 + (int)roundf(l * *y_cos);
				zn = z0 + (int)roundf(l * *z_cos);

				xn = xn<0 ? 0 : xn; xn = xn>=engine::shared_settings.x_dim ? engine::shared_settings.x_dim-1 : xn;
				yn = yn<0 ? 0 : yn; yn = yn>=engine::shared_settings.y_dim ? engine::shared_settings.y_dim-1 : yn;
				zn = zn<0 ? 0 : zn; zn = zn>=engine::shared_settings.z_dim ? engine::shared_settings.z_dim-1 : zn;


				//calculate/update the new parameters of the Breshalm's 3D algorithm
				d_x = (float)xn - (float)(*x);
				d_y = (float)yn - (float)(*y);
				d_z = (float)zn - (float)(*z);

				N = 0.0f;
				if (fabsf(d_x) > N)
					N = fabsf(d_x);
				if (fabsf(d_y) > N)
					N = fabsf(d_y);
				if (fabsf(d_z) > N)
					N = fabsf(d_z);

			}
			//if the nev voxel lying on the line is the detector voxel
			else if ((new_vox_structure == 3))
			{
				//check if between min and max source-detector diastance
				if ((engine::shared_settings.em_det_in_separate_file == 1) ||
					((engine::shared_settings.em_det_in_separate_file == 0) &&
					 (((*d_x_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *x)*((*d_x_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *x) +
					  ((*d_y_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *y)*((*d_y_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *y) +
					  ((*d_z_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *z)*((*d_z_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *z) >= (engine::shared_settings.em_det_distance_min/engine::shared_settings.vox_size)*(engine::shared_settings.em_det_distance_min/engine::shared_settings.vox_size)) &&
					 (((*d_x_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *x)*((*d_x_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *x) +
					  ((*d_y_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *y)*((*d_y_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *y) +
					  ((*d_z_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *z)*((*d_z_coordinates)[engine::shared_settings.scatering_coordinates_number * *thread] - *z) <= (engine::shared_settings.em_det_distance_max/engine::shared_settings.vox_size)*(engine::shared_settings.em_det_distance_max/engine::shared_settings.vox_size))))
				{

					//put the detector coordinates to the scattering events arrays
					(*d_x_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *x;
					(*d_y_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *y;
					(*d_z_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *z;
					//(*counter)++;

					//add to the path_debug (parameter to check if the photon travel time is less than the maximum time in DTOF)
					//only the distance between the old scattering event and the detector
					//this->path_debug += sqrtf((this->x - this->x0)*(this->x - this->x0) + (this->y - this->y0)*(this->y - this->y0) + (this->z - this->z0)*(this->z - this->z0));
					//this->path_debug++;

					(*path) -= l;
					//Breshalm's 3D line correction
					//do the same byt consider only the scattering coefficient mus
					l = __fsqrt_rn((*x - x0)*(*x - x0) + (*y - y0)*(*y - y0) + (*z - z0)*(*z - z0));

					(*path) += l;


					//(*path)++;

					*detected = 1;
				}

				*weight = 0.0f;
				go_on = 0;
				return;
				//break;
			}
			//if the nev voxel lying on the line is the source voxel or a voxel outside the voxels structure
			else if ((new_vox_structure < 3))
			{
				//emitter
				//if MC code according to Liebert2008
				//just kill the photon and break the procedure
				//photon just reached the external voxels structure surface
				*weight = 0.0f;
				go_on = 0;
				return;
				//break;
			}

			//if photon is still alive, calculate the distances (in fact squares of the distances):
			//r1 - from the old scattering event to the actual photon position
			//r2 - from the old scattering event to the new scattering event
			//int r1 = (this->x - this->x0)*(this->x - this->x0) + (this->y - this->y0)*(this->y - this->y0) + (this->z - this->z0)*(this->z - this->z0);
			//int r2 = (this->xn - this->x0)*(this->xn - this->x0) + (this->yn - this->y0)*(this->yn - this->y0) + (this->zn - this->z0)*(this->zn - this->z0);

			//photon reached the end point of the step - break the procedure
			//the actual photon position reached the new scattering event coordinates
			//if (r1 >= r2)
			if (((*x - x0)*(*x - x0) + (*y - y0)*(*y - y0) + (*z - z0)*(*z - z0)) >= ((xn - x0)*(xn - x0) + (yn - y0)*(yn - y0) + (zn - z0)*(zn - z0)))
			{
				//if (this->x - this->xn + this->y - this->yn + this->z - this->zn != 0)
				//{
				//if (((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F) < 4)
				//	printf("%d\n",(*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F);
					//printf("%f\t%d : %d : %d -- %d : %d : %d -- %d : %d : %d\n",__fsqrt_rn((xn - x0)*(xn - x0) + (yn - y0)*(yn - y0) + (zn - z0)*(zn - z0)),x0,*x,xn,y0,*y,yn,z0,*z,zn);
				//}

				//put the actual photon position to the scattering coordinates arrays
				(*d_x_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *x;
				(*d_y_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *y;
				(*d_z_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *z;
				(*counter)++;

				//add to the path_debug (parameter to check if the photon travel time is less than the masimum time in DTOF)
				//the distance from the old to the new scattering event
				//this->path_debug += this->l;
				go_on = 0;
				break;
			}

		}
		//line from x0 to xn shorter than voxel size, photon still inside the same voxel - break the procedure
		else
		{
			//add to the path_debug (parameter to check if the photon travel time is less than the masimum time in DTOF)
			//the distance from the old to the new scattering event
			//*path += 1f;

			//if (((*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F) < 4)
			//	printf("%d\n",(*d_voxels)[*x + engine::shared_settings.x_dim* *y + engine::shared_settings.x_dim*engine::shared_settings.y_dim* *z] & 0x0F);

			//printf("%d\t%d\t%d\n",*x,*y,*z);

			//put the actual photon position to the scattering coordinates arrays
			(*d_x_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *x;
			(*d_y_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *y;
			(*d_z_coordinates)[*counter + engine::shared_settings.scatering_coordinates_number * *thread] = *z;
			(*counter)++;
			go_on = 0;
			break;
		}
	}

	//if (!isfinite(l))
	//{
	//	printf("%f\t%d\t%d\t%d:%d:%d\t%d:%d:%d\t%d:%d:%d\n",(*weight),old_vox_structure,new_vox_structure,x0,*x,xn,y0,*y,yn,z0,*z,zn);
	//}

	//this->path_debug += sqrtf((this->xn - this->x0)*(this->xn - this->x0) + (this->yn - this->y0)*(this->yn - this->y0) + (this->zn - this->z0)*(this->zn - this->z0));

	/*
	//if photon still alive, copy the new scattering event coordinates to the actual photon coordinates
	if (*weight > 0.0f)
	{
		if ((xn < 0) || (xn >= engine::shared_settings.x_dim) ||
			(yn < 0) || (yn >= engine::shared_settings.y_dim) ||
			(zn < 0) || (zn >= engine::shared_settings.z_dim))
		{
			//printf("%d\t%d\t%d\t%f\t%f\t%f\t%f\n",this->xn,this->yn,this->zn,engine::shared_settings.source_x_cos,engine::shared_settings.source_y_cos,engine::shared_settings.source_z_cos,this->l);
			//printf("%d\n",this->structure_index);
			printf("%d\t%f\t%d\t%d\n",old_vox_structure, x_tmp, xn, *first);
			*weight = 0.0f;
			return;
		}
		*x = xn;
		*y = yn;
		*z = zn;
	}
	*/

	//if photon still alive, copy the new scattering event coordinates to the actual photon coordinates
	//*x = xn;
	//*y = yn;
	//*z = zn;

	//if ((*x < 0) || (*x >= engine::shared_settings.x_dim) ||
	//	(*y < 0) || (*y >= engine::shared_settings.y_dim) ||
	//	(*z < 0) || (*z >= engine::shared_settings.z_dim))
	//{
	//	printf("DUPA\n");
	//}


	//if photon still alive, calculate the new directional cosines of photon moving direction
	if ((*z_cos == 1.0f) || (*z_cos == -1.0f))
	{//this prevents from the numerical calculations problems for abs(this->z_cos) == 0
		*x_cos = *sint * *cosp;
		*y_cos = *sint * *sinp;
		*z_cos = *cost*(*z_cos>=0 ? 1.0f :-1.0f);
	}
	else
	{
		//den = sqrtf(1.0f - z_cos * z_cos);
		tmp = (*sint * (*x_cos * *z_cos * *cosp - *y_cos * *sinp)) / __fsqrt_rn(1.0f - *z_cos * *z_cos) + *x_cos * *cost;
		*y_cos = (*sint * (*y_cos * *z_cos * *cosp + *x_cos * *sinp)) / __fsqrt_rn(1.0f - *z_cos * *z_cos) + *y_cos * *cost;
		*x_cos = tmp;
		*z_cos = -__fsqrt_rn(1.0f - *z_cos * *z_cos) * *sint * *cosp + *z_cos * *cost;
	}
}






