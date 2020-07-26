/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * utilities.cpp, include utility functions
 */


#include "utilities.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/******************       RANDOM NUMBER GENERATOR FUNCTIONS          ******************************/

class RandomGeneratorPrivate {
public:
	gsl_rng* rng;
	RandomGeneratorPrivate() {
		rng = gsl_rng_alloc( gsl_rng_taus2 );
	};
	~RandomGeneratorPrivate() {
		gsl_rng_free( rng );
	};
};

RandomGenerator::RandomGenerator( int seed ) {
	prive = new RandomGeneratorPrivate();
	gsl_rng_set( prive->rng, seed );
}

RandomGenerator::~RandomGenerator() {
	delete prive;
}

void RandomGenerator::setSeed( int seed ) {
	this->seedv = seed;
	gsl_rng_set( prive->rng, seed );
}

int RandomGenerator::seed() {
	return seedv;
}

int RandomGenerator::getInt( int min, int max ) {
    return ( gsl_rng_uniform_int( prive->rng, max-min+1) + min);
}

double RandomGenerator::getDouble( double min, double max ) {
	//--- FIXME this implementation never return max
	return gsl_ran_flat( prive->rng, min, max );
}

double RandomGenerator::getGaussian( double var, double mean )
{
	return gsl_ran_gaussian( prive->rng, var ) + mean;
}

/******************       SORTING FUNCTIONS                ******************************/

/*
 * sort a vector of doubles in descending order and create a ranking list with id numbers
 * in case of parity the latter come first
 */
void sortDoubles(int n, double *vector, int *rank)
{
    int i,ii;
    int *r;
    double *v;
    int *rr;
    double max;
    int maxi;
    bool *exist;
    bool *e;

    exist = (bool *) malloc(n * sizeof(bool));
    for(i=0, e=exist; i < n; i++, e++)
        *e = true;

    for(i=0, r=rank; i < n; i++, r++)
    {
      for(ii=0,rr=rank,v=vector, e=exist, maxi=0, max = -999999999999; ii < n; ii++, rr++, v++, e++)
      {
        if (*e)
        {
           if (*v >= max)
           {
              maxi = ii;
              max = *v;
           }
        }
      }
      *r = maxi;
      *(exist + maxi) = false;
    }
}

/*
 * sort a vector of doubles in descending order and create a ranking list with id numbers
 * in case of parity the first come first
 */
void sortDoubles2(int n, double *vector, int *rank)
{
    int i,ii;
    int *r;
    double *v;
    int *rr;
    double max;
    int maxi;
    bool *exist;
    bool *e;
	
    exist = (bool *) malloc(n * sizeof(bool));
    for(i=0, e=exist; i < n; i++, e++)
        *e = true;
	
    for(i=0, r=rank; i < n; i++, r++)
    {
        for(ii=0,rr=rank,v=vector, e=exist, maxi=0, max = -999999999999; ii < n; ii++, rr++, v++, e++)
        {
            if (*e)
            {
                if (*v > max)
                {
                    maxi = ii;
                    max = *v;
                }
            }
        }
        *r = maxi;
        *(exist + maxi) = false;
    }
}

/*
 * sort a vector of doubles with an ascending order (from the lowest to the highest)
 * fill the rank vector of integers with the id and the rankfit vector with the corresponding value
 * in case of parity the first come first
 */
void sortDoublesAsc(int n, double *vector, int *rank, double *rankv)
{
    int i,ii;
    int *r;
    double *v;
    double *rv;
    int *rr;
    double min;
    int mini;
    bool *exist;
    bool *e;
	
    exist = (bool *) malloc(n * sizeof(bool));
    for(i=0, e=exist; i < n; i++, e++)
        *e = true;
	
    for(i=0, r=rank, rv=rankv; i < n; i++, r++, rv++)
    {
        for(ii=0,rr=rank,v=vector, e=exist, mini=0, min = 999999999999; ii < n; ii++, rr++, v++, e++)
        {
            if (*e)
            {
                if (*v < min)
                {
                    mini = ii;
                    min = *v;
                }
            }
        }
        *r = mini;
        *rv = min;
        *(exist + mini) = false;
    }
}

/******************       TRIGONOMENTRIC FUNCTIONS          ******************************/

/*
 * return the x endpoint of a vector starting from x
 */
double xvect(double ang, double module)

{
    return(cos(ang) * module);
}

/*
 * return the x endpoint of a vector starting from x
 */
double yvect(double ang, double module)

{
    return(sin(ang) * module);
}

/*
 * return the (clockwise) angle between two vectors
 */
double angv(double x1, double y1, double x2, double y2)

{

    return(atan2(y2 - y1, x2 - x1) + M_PI);
}

/*
 * return the smallest angle span from a1 to a2
 */
double angdelta(double a1, double a2)

{

    double da;

    da = a2 - a1;
    if (da > M_PI)
      da = (M_PI * 2) - da;
    if (da < -M_PI)
      da = da + (M_PI * 2);

    return(da);
}

/*
 * verify whether the angle a is in the range [r1,r2] or [r1-PI2,r2-PI2] or [r1+PI2,r2+PI2]
 *   assume that the angle is in the range [0, PI2]
 */
bool anginrange(double a, double r1, double r2)

{

    if (a > r1 && a < r2)
      {
      return(true);
      }
     else
      {
        if (a > (r1 + PI2) && a < (r2 + PI2))
         {
           return(true);
         }
         else
          {
            if (a > (r1 - PI2) && a < (r2 - PI2))
              {
               return(true);
              }
              else
              {
                return(false);
              }
           }
        }

}

/*
 * return the relative angle
 */
double mangrelr(double absolute, double orient)


{

    double  relative;

    relative = absolute - orient;

    if (relative < 0.0)
        relative += (M_PI * 2.0);

    return(relative);

}


/******************       STRING UTILITY FUNCTIONS          ******************************/

/*
 * copy the first string in the second string by removing initial spaces
 * and by terminating when a newline is encountered
 * strings should be shorter than 1024 characters
 */
void copyandclear(char *s, char *sc)

{
    int n = 0;

    while (*s == ' ' && n < 1024)
    {
        s++;
        n++;
    }

    while (*s != ' ' && *s != '\n' && *s != '\0' && n < 1024)
    {
        *sc = *s;
        sc++;
        s++;
    }
    *sc = '\0';

}

/*
 * check whether a filename string end with a suffix
 * the suffix should start with a dot
 */
bool filenendwith(char *str, char *suffix)
{
    char *dot = strrchr(str, '.');
    if (dot && !strcmp(dot, suffix))
       return(true);
     else
       return(false);
}

/*
 * return the binary value of a true/false string
 */
bool parsebool(char *str)

{
  if (strcmp(str, "true") || strcmp(str, "True"))
      return(true);
  if (strcmp(str, "false") || strcmp(str, "False"))
      return(false);
  printf("Warning: %s is not a true/false string\n", str);
  return(false);
	
}

/******************       STATISTICS FUNCTIONS          ******************************/

/*
 * return the standard deviation of a vector of double numbers containing nelements element
 */
double computeSdv(double *vector, int nelements)

{
    double sdv = 0.0;
    double ave = 0.0;
    double *v;
    int i;
	
    for(i=0, v=vector; i < nelements; i++, v++)
        ave += *v;
    ave /= (double) nelements;
	
    for(i=0, v=vector; i < nelements; i++, v++)
        sdv += (*v - ave) * (*v - ave);
    sdv = sqrt(sdv / (double) nelements);
	
    return(sdv);
	
}
