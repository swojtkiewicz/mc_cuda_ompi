/**
 * @author Stanislaw Wojtkiewicz
 */

#include "utilities.h"

#include <cmath>
#include <cfloat>

using namespace std;



void calculateSensitivityFactors(Settings *settings, float **DTOFs_MPP, float **DTOFs_MTSF, float **DTOFs_VSF, float **voxels_MPP, float **voxels_MTSF, float **voxels_VSF, int *DTOFs_cut_index, int ind_em_det_pair, bool filtfilt)
{

	//TO_DO add Thrust support

	int i, ind, windowSize;

	saveResults<float>((*DTOFs_MPP), "DTOFs_MPP.txt", 1, (*settings).numDTOF / 2, 0, 0);
	saveResults<float>((*DTOFs_MTSF), "DTOFs_MTSF.txt", 1, (*settings).numDTOF / 2, 0, 0);
	saveResults<float>((*DTOFs_VSF), "DTOFs_VSF.txt", 1, (*settings).numDTOF / 2, 0, 0);

	if (filtfilt)
	{

		windowSize = (int)ceilf(4.0f / 256.0f * (float)((*settings).numDTOF / 2));

		float mean_MPP;
		float mean_MTSF;
		float mean_VSF;
		for (i = 0; i <= (*settings).numDTOF / 2 - windowSize; i++)
		{
			mean_MPP = 0.0f;
			mean_MTSF = 0.0f;
			mean_VSF = 0.0f;
			for (ind = i; ind < i + windowSize; ind++)
			{
				mean_MPP +=  (*DTOFs_MPP)[ind];
				mean_MTSF +=  (*DTOFs_MTSF)[ind];
				mean_VSF +=  (*DTOFs_VSF)[ind];
			}
			mean_MPP /= windowSize;
			mean_MTSF /= windowSize;
			mean_VSF /= windowSize;

			(*DTOFs_MPP)[i] = mean_MPP;
			(*DTOFs_MTSF)[i] = mean_MTSF;
			(*DTOFs_VSF)[i] = mean_VSF;
		}

		for (i = (*settings).numDTOF / 2 - 1; i >= windowSize - 1; i--)
		{
			mean_MPP = 0.0f;
			mean_MTSF = 0.0f;
			mean_VSF = 0.0f;
			for (ind = i; ind > i - windowSize; ind--)
			{
				mean_MPP +=  (*DTOFs_MPP)[ind];
				mean_MTSF +=  (*DTOFs_MTSF)[ind];
				mean_VSF +=  (*DTOFs_VSF)[ind];
			}
			mean_MPP /= windowSize;
			mean_MTSF /= windowSize;
			mean_VSF /= windowSize;

			(*DTOFs_MPP)[i] = mean_MPP;
			(*DTOFs_MTSF)[i] = mean_MTSF;
			(*DTOFs_VSF)[i] = mean_VSF;
		}
		saveResults<float>((*DTOFs_MPP), "DTOFs_MPP_filtfilt.txt", 1, (*settings).numDTOF / 2, 0, 0);
		saveResults<float>((*DTOFs_MTSF), "DTOFs_MTSF_filtfilt.txt", 1, (*settings).numDTOF / 2, 0, 0);
		saveResults<float>((*DTOFs_VSF), "DTOFs_VSF_filtfilt.txt", 1, (*settings).numDTOF / 2, 0, 0);
		//cout << windowSize << endl << flush;
	}

	///where to cut the DTOFs!
	if (ind_em_det_pair == 0)
	{

		*DTOFs_cut_index = (*settings).numDTOF / 2;

		float max_value = max1DArray<float>(DTOFs_MPP,*DTOFs_cut_index);

		int max_index = max1DArrayIndex<float>(DTOFs_MPP,*DTOFs_cut_index);

		windowSize = (int)ceilf(4.0f / 256.0f * (float)(*DTOFs_cut_index));

		float dtof_mean = 0.0f;
		for (i = 0; i <= (*settings).numDTOF / 2 - windowSize; i++)
		{
			dtof_mean = 0.0f;
			for (ind = i; ind < i + windowSize; ind++)
			{
				dtof_mean += (*DTOFs_MPP)[ind];
			}
			dtof_mean /= windowSize;

			if ((i > max_index) && (dtof_mean <= 0.01f*max_value))
			{
				*DTOFs_cut_index = i;
				break;
			}
		}

		if (*DTOFs_cut_index <= max_index)
		{
			*DTOFs_cut_index = (*settings).numDTOF / 2;
		}

		//cout << "max_val = " << max_value << ", max_ind = " << max_index << ", winSize = " << windowSize << ", i = " << i << endl << flush;
	}

	//cout << ind_em_det_pair << " : " << *DTOFs_cut_index << endl << flush;

	float *Ntot = new float[3];
	Ntot[0] = 0.0f; Ntot[1] = 0.0f; Ntot[2] = 0.0f;
	float *mean_time = new float[2];
	mean_time[0] = 0.0f; mean_time[1] = 0.0f;
	float mean_time_square = 0.0f;


	for (i = 0; i < *DTOFs_cut_index; i++)
	{
		Ntot[0] += (*DTOFs_MPP)[i];
		Ntot[1] += (*DTOFs_MTSF)[i];
		Ntot[2] += (*DTOFs_VSF)[i];
		mean_time[0] += (*DTOFs_MTSF)[i] * i * ((*settings).DTOFmax / ((*settings).numDTOF/2));
		mean_time[1] += (*DTOFs_VSF)[i] * i * ((*settings).DTOFmax / ((*settings).numDTOF/2));
		mean_time_square += (*DTOFs_VSF)[i] * (i * ((*settings).DTOFmax / ((*settings).numDTOF/2))) * (i * ((*settings).DTOFmax / ((*settings).numDTOF/2)));
	}

	if ((Ntot[0] <= 0.0f) || (Ntot[1] <= 0.0f) || (Ntot[2] <= 0.0f) || !(std::isfinite(Ntot[0])) || !(std::isfinite(Ntot[1])) || !(std::isfinite(Ntot[2])))
	{
		cerr << "Error: Reflectance DTOFs integrals <= 0 (MPP: " << Ntot[0] << ", MTSF: " << Ntot[1] << ", VSF: " << Ntot[2] << ")." << endl << flush;
		exit(-1);
	}

	mean_time[0] /= Ntot[1];
	mean_time[1] /= Ntot[2];
	mean_time_square /= Ntot[2];


	for (i = 0; i < (int)((*settings).x_dim * (*settings).y_dim * (*settings).z_dim); i++)
	{
		if (!isfinite((*voxels_MPP)[i]))
			cout << "1";
		(*voxels_MPP)[i] = (*voxels_MPP)[i]/Ntot[0];

		//(*voxels_VSF)[i] = -(*voxels_VSF)[i]/Ntot[2] + 2 * (*voxels_MTSF)[i] / Ntot[1] * mean_time[1] +
		//					(*voxels_MPP)[i] * (mean_time_square - 2 * mean_time[1] * mean_time[1]);

		//(*voxels_MTSF)[i] = -(*voxels_MTSF)[i]/Ntot[1] + (*voxels_MPP)[i] * mean_time[0];



		(*voxels_VSF)[i] = -(*voxels_VSF)[i]/Ntot[0] + 2 * (*voxels_MTSF)[i]/Ntot[0] * mean_time[1] +
							(*voxels_MPP)[i] * (mean_time_square - 2 * mean_time[1] * mean_time[1]);

		(*voxels_MTSF)[i] = -(*voxels_MTSF)[i]/Ntot[0] + (*voxels_MPP)[i] * mean_time[0];

	}
}

int calculateMaxGPUSize(std::vector<ClusterComputer> *cluster_computers, std::stringstream &message_stream)
{
	try
	{
		int max_GPU_size = std::numeric_limits<int>::min();

		for (std::vector<ClusterComputer>::iterator it = (*cluster_computers).begin() ; it != (*cluster_computers).end(); ++it){

			auto it_block_max = std::max_element((*it).no_of_blocks.begin(), (*it).no_of_blocks.end());
			auto it_threads_max = std::max_element((*it).threads_per_block.begin(), (*it).threads_per_block.end());

			if (max_GPU_size < ((*it_block_max)*(*it_threads_max))){
				max_GPU_size = ((*it_block_max)*(*it_threads_max));
			}
		}

		if (max_GPU_size < 1){
			message_stream << "Bad value of the GPU run size (blocks*thread) as read from the cluster file." << std::endl;
			message_stream << WHERE_AM_I << std::endl;
		}

		return max_GPU_size;

	}
	catch (exception &ex)
	{
		message_stream << ex.what() << std::endl;
		message_stream << WHERE_AM_I << std::endl;
		return -1;
	}
}

int calculateWorkers(std::vector<ClusterComputer> *cluster_computers, std::stringstream &message_stream)
{
	try
	{
		int number_of_workers = 0;

		for (std::vector<ClusterComputer>::iterator it = (*cluster_computers).begin() ; it != (*cluster_computers).end(); ++it){
		    number_of_workers += (*it).no_of_GPUs;
		}



		if (number_of_workers == 0){
			message_stream << "Zero workers read from file." << std::endl;
			message_stream << WHERE_AM_I << std::endl;
		}

		return number_of_workers;

	}
	catch (exception &ex)
	{
		message_stream << ex.what() << std::endl;
		message_stream << WHERE_AM_I << std::endl;
		return -1;
	}
}


bool loadClusterParametersFile(std::vector<ClusterComputer> *cluster_computers, string cluster_parameters_filename, int *number_of_cluster_computers, std::stringstream &message_stream)
{
	try
	{
		ifstream sr;
		sr.open(cluster_parameters_filename.c_str());
		stringstream strstream;
		string property_name;
		string line;

		(*number_of_cluster_computers) = 0;

		if (sr)
		{
			while (getline(sr,line))
			{
				strstream.str(line);

				if (line[0] != '#')
				{
					getline(strstream,property_name,':');

					if (strcmp(property_name.c_str(),"Hostname") == 0)
					{
						(*number_of_cluster_computers)++;
						//cout << "Property := " << property_name << "=" << *number_of_cluster_computers << endl;
					}
				}
				strstream.str("");
				strstream.clear();
			}
			sr.close();
		}

		*number_of_cluster_computers = -1;

		sr.open(cluster_parameters_filename.c_str());

		if (sr)
		{
			while (getline(sr,line))
			{
				strstream.str(line);

				if (line[0] != '#')
				{
					getline(strstream,property_name,':');

					if (strcmp(property_name.c_str(),"Hostname") == 0)
					{
						(*number_of_cluster_computers)++;
						(*cluster_computers).push_back(ClusterComputer());
						strstream >> (*cluster_computers).back().name;
						//cout << "Property := " << property_name << "=" << (*cluster_computers).back().name << endl << flush;

					}
					else if (strcmp(property_name.c_str(),"Machine_GPUs") == 0)
					{
						strstream >> (*cluster_computers).back().no_of_GPUs;
						//cout << "Property := " << property_name << "=" << (*cluster_computers).back().no_of_GPUs << endl << flush;


					}
					else if (strcmp(property_name.c_str(),"Threads per block") == 0)
					{
						while (strstream)
						{

							(*cluster_computers).back().threads_per_block.push_back(0);
							strstream >> (*cluster_computers).back().threads_per_block.back();
							//cout << "Property := " << property_name << "=" << (*cluster_computers).back().threads_per_block.back() << endl << flush;
						}
						(*cluster_computers).back().threads_per_block.pop_back();
					}
					else if (strcmp(property_name.c_str(),"Number of blocks") == 0)
					{
						while (strstream)
						{
							(*cluster_computers).back().no_of_blocks.push_back(0);
							strstream >> (*cluster_computers).back().no_of_blocks.back();
						}
						(*cluster_computers).back().no_of_blocks.pop_back();
					}
				}
				strstream.str("");
				strstream.clear();
			}
			sr.close();
		}

		(*number_of_cluster_computers)++;

		return true;

	}
	catch (exception &ex)
	{
		message_stream << "ERROR: Can't read file: '" << cluster_parameters_filename << "'. " << ex.what();
		message_stream << WHERE_AM_I << std::endl;
		return false;
	}

}



unsigned int* loadVoxelsFile(string voxels_filaname, Settings *settings, int **source_x_coordinates, int **source_y_coordinates, int **source_z_coordinates)
{
	unsigned int *voxels = NULL;

	ifstream sr;
	sr.open(voxels_filaname.c_str(), ios::binary);

	if (sr.is_open())
	{
		sr.read((char*)&settings->vox_size, sizeof(float));

//		sr.read((char*)&settings->source_x_cos, 4);
//		sr.read((char*)&settings->source_y_cos, 4);
//		sr.read((char*)&settings->source_z_cos, 4);
		sr.read((char*)&settings->x_dim, sizeof(unsigned int));
		sr.read((char*)&settings->y_dim, sizeof(unsigned int));
		sr.read((char*)&settings->z_dim, sizeof(unsigned int));

		int em_det_pairs_in_file;
		sr.read((char*)&em_det_pairs_in_file, sizeof(int));

		if (em_det_pairs_in_file != !settings->em_det_in_separate_file)
		{
			cerr << "\nLocations of sources and detectors are differently defined in *.vox and *.set files!\n"
					"*.vox file - sources and detectors in separate file = " << em_det_pairs_in_file <<
					"\n*.set file - sources and detectors in separate file = " << !settings->em_det_in_separate_file << endl;
			exit(1);
		}


		if (em_det_pairs_in_file == 1)
		{
			sr.read((char*)&settings->source_coordinates_number, sizeof(unsigned int));

			*source_x_coordinates = new int[settings->source_coordinates_number];
			*source_y_coordinates = new int[settings->source_coordinates_number];
			*source_z_coordinates = new int[settings->source_coordinates_number];

			sr.read((char*)*source_x_coordinates, sizeof(int) * settings->source_coordinates_number);
			sr.read((char*)*source_y_coordinates, sizeof(int) * settings->source_coordinates_number);
			sr.read((char*)*source_z_coordinates, sizeof(int) * settings->source_coordinates_number);
		}

		voxels = new unsigned int[settings->x_dim * settings->y_dim * settings->z_dim];
		sr.read((char*)voxels, sizeof(unsigned int) * settings->x_dim * settings->y_dim * settings->z_dim);

	}
	sr.close();

	return voxels;
}

/**
 * \exception throws all exceptions
 */
void saveVoxels(string voxels_filename, float *voxels, Settings *settings)
{
	/**
	 * Binary output file format:
	 * -# \code 1 x unsigned int\endcode - voxel structure x dimension (x_dim)
	 * -# \code 1 x unsigned int\endcode - voxel structure y dimension (y_dim)
	 * -# \code 1 x unsigned int\endcode - voxel structure z dimension (z_dim)
	 * -# \code x_dimmension * y_dimmension * z_dimmension x float32 \endcode - weights of voxels
	 * .
	 *
	 * To get weight of voxel in position \code (ind_x, ind_y, ind_z) \endcode use formula:
	 * \code ind_x + x_dim*ind_y + x_dim*y_dim*ind_z \endcode
	 *
	 * where voxels positions starts with 0
	 */

	try
	{
		//printf("saving voxels file...\n");

		FILE* pFile;
		pFile = fopen(voxels_filename.c_str(), "wb");
		fwrite(&settings->x_dim, 1, sizeof(unsigned int), pFile);
		fwrite(&settings->y_dim, 1, sizeof(unsigned int), pFile);
		fwrite(&settings->z_dim, 1, sizeof(unsigned int), pFile);

		fwrite(voxels, 1, settings->x_dim * settings->y_dim * settings->z_dim * sizeof(float), pFile);

		fclose(pFile);

	}
	catch (exception &ex)
	{
		cerr << "ERROR: Can't read settings.set file. " << ex.what();
	}
	catch (...)
	{
		cerr << "ERROR: Can't read settings.set file.";
	}
}

/**
 * \exception throws all exceptions
 */
bool loadEmDetFile(string em_det_filename, int **em_det_pairs, int *no_of_em_det_pairs, std::stringstream &message_stream)
{
	try
	{
		ifstream sr;
		sr.open(em_det_filename.c_str());
		stringstream strstream;
		string property_name;
		string line;

		*no_of_em_det_pairs = 0;
		int coordinate_counter = 0;

		if (sr)
		{
			while (getline(sr,line))
			{
				strstream.str(line);

				if (line[0] != '#')
				{
					(*no_of_em_det_pairs)++;
				}

				strstream.str("");
				strstream.clear();
			}
			sr.close();
		}


		*em_det_pairs = new int[EM_DET_PAIR_PARAMS*(*no_of_em_det_pairs)];
		memset(*em_det_pairs, 0, sizeof(int) * EM_DET_PAIR_PARAMS * (*no_of_em_det_pairs));

		sr.open(em_det_filename.c_str());


		if (sr)
		{
			while (getline(sr,line))
			{
				strstream.str(line);

				if (line[0] != '#')
				{
					while (strstream)
					{
						strstream >> (*em_det_pairs)[coordinate_counter];
						coordinate_counter++;
					}
					coordinate_counter--;
				}

				strstream.str("");
				strstream.clear();
			}
			sr.close();
		}

		return true;
	}
	catch (exception &ex)
	{
		message_stream << "ERROR: Can't read file: '" << em_det_filename << "'. " << ex.what();
		message_stream << WHERE_AM_I << std::endl;
		return false;
	}
}


void loadOpticalPropertiesCourse(string opt_prop_course_filename, float **opt_prop_course, int *no_of_opt_prop_changes)
{
	///- there could be up to optical properties changes

	//TO_DO change as in reading the sor-det pairs

	//declare optical properties changes 12 structures x 7 optical properties [n g mus muax muam muafx muafm]
	*opt_prop_course = new float[12*7*1000];
	for (int i = 0; i < 12*7*1000; i++)
	{
		(*opt_prop_course)[i] = 0.0f;
	}
	*no_of_opt_prop_changes = 0;
	int opt_prop_change_counter = 0;

	try
	{
		ifstream sr;
		sr.open(opt_prop_course_filename.c_str());
		stringstream strstream;
		string line;

		if (sr)
		{
			while (getline(sr,line))
			{
				strstream.str(line);

				if (line[0] != '#')
				{
					if ((*no_of_opt_prop_changes) > 999)
					{
						cerr << "\nNumber of optical properties changes exceeded 1000! Please reduce it.";
						exit(1);
					}
					while (strstream)
					{
						strstream >> (*opt_prop_course)[opt_prop_change_counter];
						opt_prop_change_counter++;
					}
					opt_prop_change_counter--;
					*no_of_opt_prop_changes += 1;
				}

				strstream.str("");
				strstream.clear();
			}
			sr.close();
		}
	}
	catch (exception &ex)
	{
		cerr << "ERROR: Can't read optical properties changes file: " << opt_prop_course_filename << ex.what();
	}
	catch (...)
	{
		cerr << "ERROR: Can't read optical properties changes file: " << opt_prop_course_filename;
	}
}



/**
 * \exception throws all exceptions
 */
Settings loadSettingsFile(string settings_filaname, string *voxels_filename, float **optical_properties, float *total_photons,
		  	  	  	  	  string *em_det_filename, string *opt_prop_change_filename, string *cluster_filename)
{

	Settings settings;

	//initialize with defaults
	settings.GPU_ID = 0;

	//declare optical properties (7 optical properties in 12 structures)
	*optical_properties = new float[MAX_STRUCTURES * NUM_OPTICAL_PROPERTIES];
	for (int i = 0; i < MAX_STRUCTURES * NUM_OPTICAL_PROPERTIES; i++)
	{
		(*optical_properties)[i] = 0.0f;
	}
	int optical_property_counter = 0;

	try
	{
		ifstream sr;
		sr.open(settings_filaname.c_str());
		stringstream strstream;
		string property_name;
		string line;

		if (sr)
		{
			while (getline(sr,line))
			{
				strstream.str(line);

				if (line[0] != '#')
				{
					getline(strstream,property_name,':');

					if (strcmp(property_name.c_str(),"MAX time in DTOFs [ps]") == 0)
					{
						strstream >> settings.DTOFmax;
						//cout << "Property := " << property_name << "=" << settings.DTOFmax << endl;
					}
					else if (strcmp(property_name.c_str(),"Number of DTOFs samples") == 0)
					{
						strstream >> settings.numDTOF;
						//cout << "Property := " << property_name << "=" << settings.numDTOF << endl;
					}
					else if (strcmp(property_name.c_str(),"Total number of photons") == 0)
					{
						strstream >> *total_photons;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Voxels update") == 0)
					{
						strstream >> settings.voxels_update;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Voxels filename") == 0)
					{
						strstream >> *voxels_filename;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"optical_properties") == 0)
					{
						while (strstream)
						{
							strstream >> (*optical_properties)[optical_property_counter];
							optical_property_counter++;
							//cout << "Property := " << property_name << " = " << (*optical_properties)[optical_property_counter-1] << endl;
						}
						optical_property_counter--;
					}
					else if (strcmp(property_name.c_str(),"Em_det_in_separate_file") == 0)
					{
						strstream >> settings.em_det_in_separate_file;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Em_det_distance_min") == 0)
					{
						strstream >> settings.em_det_distance_min;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Em_det_distance_max") == 0)
					{
						strstream >> settings.em_det_distance_max;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Em_det_filename") == 0)
					{
						strstream >> *em_det_filename;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Optical_properties_change") == 0)
					{
						strstream >> settings.opt_prop_change;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Optical_properies_changes_filename") == 0)
					{
						strstream >> *opt_prop_change_filename;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Cluster_filename") == 0)
					{
						strstream >> *cluster_filename;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
					else if (strcmp(property_name.c_str(),"Sensitivity factors") == 0)
					{
						strstream >> settings.sensitivity_factors;
						//cout << "Property := " << property_name << "=" << settings.total_photons << endl;
					}
				}
				strstream.str("");
				strstream.clear();
			}
			sr.close();
		}
		return settings;
	}
	catch (exception &ex)
	{
		cerr << "ERROR: Can't read settings.set file. " << ex.what();
	}
	catch (...)
	{
		cerr << "ERROR: Can't read settings.set file.";
	}
	return settings;
}



