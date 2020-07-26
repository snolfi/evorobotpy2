/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * utilities.h, includes an implementation of utility functions
 */

#ifndef UTILS_H
#define UTILS_H

#define MAX_STR_LEN 1024

class RandomGenerator;
class RandomGeneratorPrivate;

// ang constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef PI2
#define PI2 (M_PI * 2.0)
#endif

/*!  \brief Random Generator Utility Class
 *
 *  \par Description
 *    Create, Manage Random Numbers Generators
 *  \par Warnings
 *
 * \ingroup utilities_rng
 */
class RandomGenerator
{
public:
	/*! Default Constructor */
	RandomGenerator( int seed = 0 );
	/*! Destructor */
	~RandomGenerator();
	/*! set the seed of this random number generator */
	void setSeed( int seed );
	/*! Return the seed setted */
	int seed();
	/*! return a random number within range specified (extreme inclusive) with a Flat distribution */
	int getInt( int min, int max );
	/*! return a random number within range specified (extreme inclusive) with a Flat distribution */
	double getDouble( double min, double max );
	/*! return a random number accordlying to a Gaussian distribution
	 *  \param var is the variance of the Gaussian distribution
	 *  \param mean is the centre of the Gaussian distribution
	 */
	double getGaussian( double var, double mean = 0.0 );

private:
	/*! encapsulate all third-party dependent code */
	RandomGeneratorPrivate* prive;
	/*! Seed */
	int seedv;
};

// sorting functions
void sortDoubles(int n, double *vector, int *rank);
void sortDoubles2(int n, double *vector, int *rank);
void sortDoublesAsc(int n, double *vector, int *rank, double *rankv);
// trigonometric functions
double xvect(double ang, double module);
double yvect(double ang, double module);
double angv(double x1, double y1, double x2, double y2);
double angdelta(double a1, double a2);
bool anginrange(double a, double r1, double r2);
double mangrelr(double absolute, double orient);
// string functions
void copyandclear(char *s, char *sc);
bool filenendwith(char *str, char *suffix);
bool parsebool(char *str);
//statistics functions
double computeSdv(double *vector, int nelements);


#endif
