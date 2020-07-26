/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * robot-env.cpp, includes functions for initializing the environmental objects and the robots,
 *                for simulating the sensors, and for moving the robot
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>
#include "robot-env.h"
#include "utilities.h"

// global variables
struct robot *rob;					// list of robots
int nrobots;						// number of robots
struct envobjects *envobjs;			// list of environemntal objects
int nenvobjs;						// number of objects
double worldx;						// world x dimension
double worldy;						// world y dimension
double *initialstates;				// robots initial positions
int cells[100][100];                // virtual cells
double cellsize=0.0;                // virtual cell size

// object samples
float **wall = NULL;				// wall samples
int *wallcf = NULL;					// description of wall samples
float **cylinder = NULL;			// cylindrical sample
int *cylindercf = NULL;				// description of cylindrical sample
float **scylinder = NULL;			// small cylindrical sample
int *scylindercf = NULL;			// description of small cylindrical sample
float **light = NULL;				// light samples
int *lightcf = NULL;				// description of light samples
int **iwall = NULL;                 // wall samples  (old style)
int **iobst = NULL;					// cylindrical sample (old style)
int **isobst = NULL;				// small cylindrical sample (old style)
int **ilight = NULL;				// light samples (old style)

int nfreep = 0;						// number of free parameters
int renderrobotworld=0;             // whether we update the graphic renderer

/*
 * Allocate and initialize environmental objects
 */
void initEnvObjects(int nobjects)
{
  int n;

  // allocate objects
  envobjs = (struct envobjects *) malloc(nenvobjs * sizeof(struct envobjects));
	
  for (n=0; n < nobjects; n++)
  {
  	envobjs[n].type = 1;
	envobjs[n].x = 100;
	envobjs[n].y = 100;
	envobjs[n].x2 = 150;
	envobjs[n].y2 = 150;
	envobjs[n].r = 25;
	// default colour is black
	envobjs[n].color[0] = 0;
	envobjs[n].color[1] = 0;
	envobjs[n].color[2] = 0;
  }
}

/*
 * Initialize a robot
 * Receive as input the pointer to the robot structure, the id number of the robot, and the robot's type
 * Set the morphological characteristics of the robot, radius, max speed of the wheels, distance between wheels
 * Initialize the variables of the robot's structure
 */
void initRobot(struct robot *cro, int n, int robottype)

{
    
    switch(robottype)
    {
        case Khepera:
            cro->radius = 27.5;
            cro->axleLength = 53.0;
            cro->maxSpeed = 144.0;
            break;
        case ePuck:
            cro->radius = 35.0;
            cro->axleLength = 55.0;
            cro->maxSpeed = 200.0;
            break;
        case MarXBot:
            cro->radius = 85.0;
            cro->axleLength = 104.0;
            cro->maxSpeed = 360.0; // (270) it was 144, then 240
            break;
            
    }
    
    // general
    cro->idn = n;                        // id
    cro->type = robottype;               // type
    cro->color = 0;                      // robot's color 0=black, 1=red, 2=green, 3=blue, 4=red/green
    cro->rgbcolor[0] = 0.0;              // LED ring color intensity (RED)
    cro->rgbcolor[1] = 0.0;              // LED ring color intensity (GREEN)
    cro->rgbcolor[2] = 0.0;              // LED ring color intensity (BLUE)
    cro->alive = true;                   // whether the robot is alive
    
    cro->x = 100;                        // x-pos
    cro->y = 100;                        // y-pos
    cro->dir = 0.0;                      // direction
    cro->leftSpeed = 0.0;                // speed of the left wheel
    cro->rightSpeed = 0.0;               // speed of the right wheel
    cro->dx = 0.0;                       // last deltax
    cro->dy = 0.0;                       // last deltay
    cro->energy = 0.0;                   // the energy level of the robot
    
    // sensors
    cro->sensorinfraredid = -1;          // id of the first infrared sensor neuron
    cro->sensorcameraid = 0;             // id of the first camera sensory neuron
    cro->sensorcameran = 0;              // n of infrared sensory neurons
    cro->camnsectors = 0;                // the number of camera setctors

    // motors
    cro->motorwheels = 0;                // number of motors neurons controlling the wheels
    cro->motorwheelstype = 0;            // type of motor wheel control (0=direct, 1=traslation-rotation)
    cro->motorleds = 0;                  // number of motors neurons controlling the wheels
    cro->motorwheelsid = 0;              // id of the first motorwheels neuron
    cro->motorledsid = 0;                // id of the first motorleds neuron
	
    if (n == 0)
     switch(robottype)
      {
        case Khepera:
            printf("Robot: Khepera\n");
            break;
        case ePuck:
            printf("Robot: ePuck\n");
            break;
        case MarXBot:
			printf("Robot: MarXBot\n");
            break;
			
    }
}

/*
 * Initialize the sensor vector that is used to store the current state of the simulated sensors
 * Initialize the current sensor pointer (csensors) that point to the first element of the sensor vector that should be uodated
 * Initially the pointer point to the first element of the vector
 * Sensor functions use csensor to know the section of the vector that should be updated and update the pointer
 * so that the next sensor updates the next units
 */
void initRobotSensors(struct robot *cro, int nsensors)

{
    
    int s;
    
    cro->sensors = (double *) malloc(nsensors * sizeof(double));
    for(s=0, cro->csensors = cro->sensors; s < nsensors; s++, cro->csensors++)
        *cro->csensors = 0.0;
    cro->csensors = cro->sensors;
}

/*
 * update the state of the robot on the basis of the state of the outputs
 * assume that outputs are in the range [-1.0,1.0]
 * move the robot on the basis of the desired speed of left and right wheels
 * eventually set the intensity of the red, greeen, and blue LEDs
 * handle collision with obstacles originating as a result of the robot motion
 * take as parameter a pointer to the robot structure and network
 * return 0 when the update do not generate collision
 * return the id of the collided robot (+1) in the case of collision with robots
 * return the id of the colliding object (+1 * -1) in the case of collision with objects
 */
int updateRobot(struct robot *cro, float *actions)

{
    double updateFrequency = 0.1;   // update frequency in s., e.g. 0.1 = 100ms
    double anglVari;
    double roboPivoDist;
    double roboCentPivoDist;
    double olddir;
    double halfPI = M_PI/2.0;
    double twoPI  = M_PI*2.0;
    double dx, dy, da;
    double x,y, odx, ody;
    double dist, smallerd;
    struct envobjects *ob;
    int nob;
    int idcoll;
    struct robot *ro;
    int r;
    double dspeed,dturn;
    
    olddir = cro->dir;
    
    // calculate the speed of the two wheels on the basis of the first two motor neurons
    switch (cro->motorwheelstype)
    {
            // standard direct control mode
            // the first 2 motor neurons control the desired speed of the left and right wheels
            // normalized in the range [-maxspeed, maxspeed]
        case 0: // standard direct control
            cro->leftSpeed  = actions[0] * cro->maxSpeed * updateFrequency;
            cro->rightSpeed = actions[1] * cro->maxSpeed * updateFrequency;
            break;
            // speed and direction mode
            // the first motor neuron encodes the desired speed
            // the second motor neuron encodes the desired angular turn
        case 1:
            dspeed = ((actions[0] + 1.0) / 2.0) * cro->maxSpeed * updateFrequency;
            dturn = ((actions[1] + 1.0) / 2.0);
            if (dturn < 0.5)
            {
                cro->leftSpeed = dspeed * (1.0 - ((0.5 - dturn) * 4.0));
                cro->rightSpeed = dspeed;
            }
            else
            {
                cro->leftSpeed = dspeed;
                cro->rightSpeed = dspeed * (1.0 - ((dturn - 0.5) * 4.0));
            }
            break;
            // speed and direction mode
            // the first motor neuron encodes the desired speed
            // the second motor neuron encodes the desired angular turn (the speed of slower wheel is at least 50% of the speed of the faster wheel)
        case 2:
            dspeed = ((actions[0] + 1.0) / 2.0) * cro->maxSpeed * updateFrequency;
            dturn = ((actions[1] + 1.0) / 2.0);
            if (dturn < 0.5)
            {
                cro->leftSpeed = dspeed * (1.0 - ((0.5 - dturn) * 2.0));
                cro->rightSpeed = dspeed;
            }
            else
            {
                cro->leftSpeed = dspeed;
                cro->rightSpeed = dspeed * (1.0 - ((dturn - 0.5) * 2.0));
            }
            break;
    }
    // update robot position on the basis of the speed of the two wheels
    // the robot is moving straight
    if ((fabs(cro->leftSpeed-cro->rightSpeed)) < 0.00001)
    {
        cro->x += dx = cos(cro->dir) * cro->rightSpeed;
        cro->y += dy = sin(cro->dir) * cro->rightSpeed;
        da = 0.0;
    }
    else
    {
        // the robot moves by turning left
        if (cro->leftSpeed < cro->rightSpeed)
        {
            anglVari = (cro->rightSpeed-cro->leftSpeed) / cro->axleLength;
            cro->dir += da = anglVari;
            roboPivoDist = cro->leftSpeed / anglVari;
            roboCentPivoDist = roboPivoDist + (cro->axleLength/2.0);
            cro->x += dx = (cos(cro->dir - halfPI) - cos(olddir - halfPI)) * roboCentPivoDist;
            cro->y += dy = (sin(cro->dir - halfPI) - sin(olddir - halfPI)) * roboCentPivoDist;
        }
        else
            // the robot moves by turning right
        {
            anglVari = (cro->leftSpeed - cro->rightSpeed) / cro->axleLength;
            cro->dir += da = -anglVari;
            roboPivoDist = cro->rightSpeed / anglVari;
            roboCentPivoDist = roboPivoDist + (cro->axleLength/2.0);
            cro->x += dx = (cos(cro->dir + halfPI) - cos(olddir + halfPI)) * roboCentPivoDist;
            cro->y += dy = (sin(cro->dir + halfPI) - sin(olddir + halfPI)) * roboCentPivoDist;
        }
    }
    
    // update the color of the LEDs
    // the third, fourth, and fifth motor neurons encode the state of the RED, GREEN, and BLUE ledsls)
    int c;
    if (cro->motorleds > 0)
    {
        for (c=0; c < cro->motorleds; c++)
            cro->rgbcolor[c] = actions[cro->motorwheels + c];
    }
    else
    {
        cro->rgbcolor[0] = 0.0;
        cro->rgbcolor[1] = 0.0;
        cro->rgbcolor[2] = 0.0;
        if (cro->color == 1) cro->rgbcolor[0] = 1.0;
        if (cro->color == 2) cro->rgbcolor[1] = 1.0;
        if (cro->color == 3) cro->rgbcolor[2] = 1.0;
    }
    
    // we now check whether the robot is colliding with an obstacle after the movement
    idcoll = 0;
    smallerd = 99999.0;
    // processing environmental object list
    for(nob=0, ob=envobjs; nob < nenvobjs; ob++, nob++)
    {
        switch(ob->type)
        {
                // we check for possible collision with wall objects
            case WALL:
                if (cro->x >= ob->x && cro->x <= ob->x2)
                    x = cro->x;
                else
                    x = ob->x;
                if (cro->y >= ob->y && cro->y <= ob->y2)
                    y = cro->y;
                else
                    y = ob->y;
                odx = (cro->x - x);
                ody = (cro->y - y);
                dist = sqrt(odx*odx+ody*ody) - cro->radius;
                if (dist <= 0.0)
                {
                    idcoll = -nob - 1;
                }
                if (dist < smallerd)
                    smallerd = dist;
                break;
                // we check for possible collisions of small cylindrical objects
            case SAMPLEDSCYLINDER:
                x = ob->x;
                y = ob->y;
                odx = (cro->x - x);
                ody = (cro->y - y);
                dist = sqrt(odx*odx+ody*ody) - (cro->radius + ob->r);
                if (dist <= 0.0)
                {
                    idcoll = -nob - 1;
                }
                if (dist < smallerd)
                    smallerd = dist;
                break;
                // we check for possible collisions with large cylindrical objects
            case SAMPLEDCYLINDER:
                x = ob->x;
                y = ob->y;
                odx = (cro->x - x);
                ody = (cro->y - y);
                dist = sqrt(odx*odx+ody*ody) - (cro->radius + ob->r);
                if (dist <= 0.0)
                {
                    idcoll = -nob - 1;
                }
                if (dist < smallerd)
                    smallerd = dist;
                break;
        }
    }
    // we check for possible collisions with other robots
    if (nrobots > 1)
    {
        for (r=0, ro=rob; r < nrobots; r++, ro++)
        {
            if (cro->idn != ro->idn)
            {
                odx = (cro->x - ro->x);
                ody = (cro->y - ro->y);
                dist = sqrt(odx*odx+ody*ody) - (cro->radius * 2.0);
                if (dist < 0.0)
                    idcoll = ro->idn + 1;
                if (dist < smallerd)
                    smallerd = dist;
            }
        }
    }
    
    // in case the robot compenetrated with an obstacle we simply move it back
    // we should refine this part by moving the robot back only the amount necessary
    // to eliminate the compenetration
    if (idcoll != 0)
    {
        cro->dir = olddir;
        cro->x = cro->x - dx;
        cro->y = cro->y - dy;
    }
    
    
    // we normalize the direction of the robot in the range [0, 2PI]
    if (cro->dir >= twoPI)
        cro->dir -= twoPI;
    if (cro->dir < 0.0)
        cro->dir += twoPI;
    
    return(idcoll);
    
}


/*
 * this is an utility function used by the initInfraredSensor and initLightSensor functions
 * it load the sample data from a .sample file
 * this data is then used to calculate the state of the simulated sensor in a fast and reliable way
 * sample data is obtained by reconding the activity of the robot's sensor in hardware
 * at different angle and distances from the robot
 */
float **
load_obstacle(char *filename, int  *objectcf)

{

   float **object;
   float  **ob;
   float  *o;
   int    *ocf;
   FILE *fp;
   int  i,ii,t,v,s;
   char sbuffer[128];

   if ((fp = fopen(filename,"r")) == NULL)
	 {
	    printf("I cannot open file %s", filename);
		fflush(stdout);
	  }
   // read configuration data: 0-nsensors, 1-nangles, 2-ndistances,
   //                          3-initdist,4-distinterval
   for (i=0, ocf=objectcf; i < 5; i++, ocf++)
     fscanf(fp,"%d ",ocf);
   fscanf(fp,"\n");
	
   // allocate space and initialize
   object = (float **) malloc(objectcf[2] * sizeof(float *));
   ob = object;
   for (i=0; i<objectcf[2]; i++,ob++)
	  {
	   *ob = (float *)malloc((objectcf[0] * objectcf[1]) * sizeof(float));
	   for (ii=0,o=*ob; ii < (objectcf[0] * objectcf[1]); ii++)
		*o = 0;
	  }
   // load data
   for (t=0, ob=object; t < objectcf[2]; t++, ob++)
    {
     fscanf(fp,"TURN %d\n",&v);
     for (i = 0, o = *ob; i < objectcf[1]; i++)
      {
       for (s=0; s < objectcf[0]; s++,o++)
        {
	   fscanf(fp,"%f ",o);
        }
       fscanf(fp,"\n");
      }
    }
   // sanity check
   fscanf(fp, "%s\n",sbuffer);
   if (strcmp(sbuffer,"END") != 0)
     {
      printf("loading file %s: the number of sample is inconsistent", filename);
	  fflush(stdout);
	 }
	
   fclose(fp);
   return object;
}


/*
 * INFRARED SENSORS INITIALIZATION (8 sensors)
 * and load associated sample data from .sample files
 */
int initInfraredSensor(struct robot *cro)
{
   if (cro->idn == 0) printf("Sensor[%d]: sampled infrared sensors (robot %d)\n", 8, cro->type);

   // init the environment and load environmental samples
   switch (cro->type)
    {
	  case Khepera:
		wallcf = (int *) malloc(5 * sizeof(int));
        wall = load_obstacle("khepera-wall.sample", wallcf);
        scylindercf = (int *) malloc(5 * sizeof(int));
        scylinder = load_obstacle("khepera-scylinder.sample", scylindercf);
        cylindercf = (int *) malloc(5 * sizeof(int));
        cylinder = load_obstacle("khepera-cylinder.sample", cylindercf);
	  break;
	  case ePuck: // TO DO: cylider should be resampled (we are using scylinder)
		wallcf = (int *) malloc(5 * sizeof(int));
        wall = load_obstacle("epuck-wall.sample", wallcf);
        scylindercf = (int *) malloc(5 * sizeof(int));
        scylinder = load_obstacle("epuck-scylinder.sample", scylindercf);
        cylindercf = (int *) malloc(5 * sizeof(int));
        cylinder = load_obstacle("epuck-cylinder.sample", cylindercf);
	  break;
	  case MarXBot:
		wallcf = (int *) malloc(5 * sizeof(int));
        wall = load_obstacle("marxbot-wall.sample", wallcf);
        scylindercf = (int *) malloc(5 * sizeof(int));
        scylinder = load_obstacle("marxbot-scylinder.sample", scylindercf);
        cylindercf = (int *) malloc(5 * sizeof(int));
        cylinder = load_obstacle("marxbot-cylinder.sample", cylindercf);
	  break;
   }

   return(8);
}

/*
 * INFRARED SENSOR UPDATE
 * The activation of this sensor is influenced by the presence of walls, cylinders, scylinders, and other robots
 * and is computed calculating the relative angle and distance of the object
 * and by selecting the corresponding activation pattern from the sample data
 */
void updateInfraredSensor(struct robot *cro)

{

	int ob;
	double x, y, dx, dy, dist, angle;
	int relang, idist;
	float *o;
	int s,r;
	struct robot *ro;
	float act[8];
	
	// initialize
	for(s=0; s < 8; s++)
	    act[s] = 0.0;
	// calculate the stimolation generated environmental objects
	for(ob=0; ob < nenvobjs; ob++)
	{
		switch(envobjs[ob].type)
		  {
			// sampled walls (we assume they are orthogonal with respect to the carthesian axes)
			case WALL:
				if (cro->x >= envobjs[ob].x && cro->x <= envobjs[ob].x2)
					x = cro->x;
				 else
					x = envobjs[ob].x;
				if (cro->y >= envobjs[ob].y && cro->y <= envobjs[ob].y2)
					y = cro->y;
				 else
					y = envobjs[ob].y;
				dx = (cro->x - x);
				dy = (cro->y - y);
				dist = sqrt(dx*dx+dy*dy);
				if (dist < (wallcf[3] + cro->radius + (wallcf[2] * wallcf[4])))
				 {
					angle = angv(x, y, cro->x, cro->y);
					angle = mangrelr(angle, cro->dir);
				    if (angle > (M_PI * 2.0)) angle -= (M_PI * 2.0);
					relang  = (int) round(angle / PI2 * 180.0);
					if (relang == 180) relang = 0;
				   //if (x == cro->x)
					//  if (y < cro->y) angle = M_PI + (M_PI * 2.0); else angle = M_PI / 2.0;
				   //else
					// if (x < cro->x) angle = M_PI; else angle = 0.0;
				   //relang  = (int) round((mangrelr(angle,cro->dir) / (M_PI * 2.0)) * 180.0);
				   idist = (int) round((dist - cro->radius - wallcf[3]) / wallcf[4]);
				   if (idist < 0)
					  idist = 0;
				   if (idist >= wallcf[2])
				      idist = wallcf[2]-1;
				   o = *(wall + idist);
				   o = (o + (relang * wallcf[0]));
				   for(s=0;s<wallcf[0];s++,o++)
                      {
						act[s] += *o;
                      }
                 }
			break;
			// sampled cylinder
			case SAMPLEDSCYLINDER:
				x = envobjs[ob].x;
				y = envobjs[ob].y;
				dx = (cro->x - x);
				dy = (cro->y - y);
				dist = sqrt(dx*dx+dy*dy) - (cro->radius + envobjs[ob].r - scylindercf[3]);
				//dot = cro->x*x + cro->y*y;
				//det = cro->x*y - cro->y*x;
				//ang = atan2(det, dot) - cro->dir;
				angle = angv(x,y, cro->x, cro->y);
				angle = mangrelr(angle, cro->dir);
				relang  = (int) round(angle / (M_PI * 2.0) * 180.0);
				if (relang == 180) relang = 0;
				if (dist < (scylindercf[2] * scylindercf[4]))
				 {
				   idist = (int) round(dist / scylindercf[4]);
				   if (idist < 0)
					  idist = 0;
				   if (idist >= scylindercf[2])
				      idist = scylindercf[2]-1;
				   o = *(scylinder + idist);
				   o = (o + (relang * wallcf[0]));
				   for(s=0;s<wallcf[0];s++,o++)
                      {
						act[s] += *o;
                      }
                 }
			break;
            // sampled cylinder
            case SAMPLEDCYLINDER:
                  x = envobjs[ob].x;
                  y = envobjs[ob].y;
                  dx = (cro->x - x);
                  dy = (cro->y - y);
                  dist = sqrt(dx*dx+dy*dy) - (cro->radius + envobjs[ob].r - cylindercf[3]);
                  angle = angv(x,y, cro->x, cro->y);
                  angle = mangrelr(angle, cro->dir);
                  relang  = (int) round(angle / (M_PI * 2.0) * 180.0);
                  if (relang == 180) relang = 0;
                  if (dist < (cylindercf[2] * cylindercf[4]))
                  {
                      idist = (int) round(dist / cylindercf[4]);
                      if (idist < 0)
                          idist = 0;
                      if (idist >= cylindercf[2])
                          idist = cylindercf[2]-1;
                      o = *(cylinder + idist);
                      o = (o + (relang * cylindercf[0]));
                      for(s=0;s<wallcf[0];s++,o++)
                      {
                          act[s] += *o;
                      }
                  }
                  break;
			// default
			//default:
			//	printf("ERROR: undefined object type %d \n",envobjs[ob].type);
			break;
		 }
	}
	// calculate the stimulation generated by other croots (treated as cylindrical objects)
	if (nrobots > 1)
	  {
	    for (r=0, ro=rob; r < nrobots; r++, ro++)
	      {
		    if (cro->idn != ro->idn)
			 {
				dx = (cro->x - ro->x);
				dy = (cro->y - ro->y);
				dist = sqrt(dx*dx+dy*dy) - (cro->radius * 2.0);
				angle = angv(ro->x, ro->y, cro->x, cro->y);
				angle = mangrelr(angle, cro->dir);
				if (angle > PI2) angle -= PI2;
				relang  = (int) round(angle / PI2 * 180.0);
				if (relang == 180) relang = 0;
				if (dist < (cylindercf[2] * cylindercf[4]))
				 {
				   idist = (int) round(dist / wallcf[4]);
				   if (idist < 0)
					  idist = 0;
				   if (idist >= cylindercf[2])
				      idist = cylindercf[2]-1;
				   o = *(cylinder + idist);
				   o = (o + (relang * wallcf[0]));
				   for(s=0;s<cylindercf[0];s++,o++)
					 act[s] += *o;
                 }
			  }
		   }
		}
	
	// normalize and copy on the imput vector
	for(s=0; s < 8; s++)
		{
		 if (act[s] > 1.0)
             act[s] = 1.0;
         *cro->csensors = act[s];
         cro->csensors++;
		}
}

/*
 * INIT TIME SENSOR (1 sensor)
 * the sensor encode the time passed since the beginning of the trials in the range [0.0, 1.0]
 */
int initTimeSensor(struct robot *cro)
{
    
    if (cro->idn == 0) printf("Sensor[%d]: trial time \n", 1);
    return(1);
}

/*
 * UPDATE TIME SENSOR (1 sensor)
 */
void updateTimeSensor(struct robot *cro, int step, int nsteps)
{
    
    *cro->csensors = (double) step / (double)nsteps;
    cro->csensors++;
    
}


/*
 * INIT ENERGY SENSOR (1 sensor)
 * the initial state of the energy and variations of the energy state
 * are supposed to be performed in the experimental component
 */
int initEnergySensor(struct robot *cro)
{
    
    if (cro->idn == 0) printf("Sensor[%d]: energy sensor \n", 1);
    return(1);
}

/*
 * UPDATE ENERGY SENSOR (1 sensor)
 */
void updateEnergySensor(struct robot *cro)
{
    
    *cro->csensors = cro->energy;
    cro->csensors++;
    
}

/*
 * INIT BIAS SENSOR (1 sensor)
 * a simple bias sensor that always return 1.0
 */
int initBiasSensor(struct robot *cro)
{
    
    if (cro->idn == 0) printf("Sensor[%d]: bias\n", 1);
    return(1);
}

/*
 * UPDATE BIAS SENSOR (1 sensor)
 * a simple bias sensor that always return 1.0
 */
void updateBiasSensor(struct robot *cro)
{
    
    *cro->csensors = 1.0;
    cro->csensors++;
    
}


/*
 * verify whether the angle a is in the range [r1,r2]
 * assume that a,r1 and r2 are in the range [0, PI2]
 */
bool anginside(double a, double r1, double r2)

{

    if ((r2 - r1) > 0)
    {
      if ((r2 - r1) < M_PI)
        {
         if (a > r1 && a < r2) // clockwise from r1 to r2
           return(true);
          else
           return(false);
        }
         else
        {
         if (a > r2 || a < r1) // clockwise from r2 to PI2 and from 0 to r1
           return(true);
          else
           return(false);
         }
    }
    else
    {
        if ((r1 - r2) < M_PI)
          {
           if (a > r2 && a < r1) // counter-clockwise from r2 to r1
             return(true);
            else
             return(false);
          }
           else
          {
           if (a > r1 || a < r2) // couner-clockwise from r1 to PI2 and from 0 to r2
             return(true);
            else
             return(false);
           }
    }


}

/*
 *	update the laser sensor
 *  measure the closest distance detected in each perceptual sector
 */
int initLaserDistanceSensor(struct robot *cro)
{
   if (cro->nlasersensors <= 0 || cro->nlasersensors > 8)
     cro->nlasersensors = 8;
   printf("Sensor[%d]: laser sensors\n", cro->nlasersensors);
   return(cro->nlasersensors);
}


void updateLaserDistanceSensor(struct robot *cro)

{

    double maxdist = 100;          // maximum dist at which  objects are detected
    double starta = -M_PI;          // initial angle of the first sectonr
    double inta = M_PI / (double) (cro->nlasersensors / 2);
    int ob;
    double x, y, dx, dy, dist;
    double a1, a2;
    double angle;                   // angle of cylindrical object or of the nearest point of a wall
    double angle1, angle2;          // angle of the initial and final point of a wall
    int s,r;
    struct robot *ro;
    double mdist[8];                // minimum detected distance
    double cdist;
    double ldist;                   // lateral distance
    double angoffset;               // angular offset between the angle of the perceived object and the angular center of the sector
    double maxperceivableaoffset;   // the maximum angular offset of a perceived object with respect to a sector

    // initialize
    for(s=0; s < cro->nlasersensors; s++)
        mdist[s] = maxdist;

    // calculate the stimulation generated environmental objects
    for(ob=0; ob < nenvobjs; ob++)
    {
        switch(envobjs[ob].type)
          {
            // walls (we assume they are orthogonal with respect to the carthesian axes)
            case WALL:
                if (cro->x >= envobjs[ob].x && cro->x <= envobjs[ob].x2)
                    x = cro->x;
                 else
                    x = envobjs[ob].x;
                if (cro->y >= envobjs[ob].y && cro->y <= envobjs[ob].y2)
                    y = cro->y;
                 else
                    y = envobjs[ob].y;
                dx = (cro->x - x);
                dy = (cro->y - y);
                dist = sqrt(dx*dx+dy*dy) - cro->radius;
                angle = angv(x, y, cro->x, cro->y);
                angle = mangrelr(angle, cro->dir);
                if (angle > M_PI) angle -= PI2; // direction of the perceived object normalized between -M_PI and M_PI
                if (angle < -M_PI) angle += PI2;
                if (dist < maxdist)
                 {
                  // we compute the angle of the initial and final point of the wall
                  angle1 = angv(envobjs[ob].x, envobjs[ob].y, cro->x, cro->y);
                  angle1 = mangrelr(angle1, cro->dir);
                  if (angle1 > M_PI) angle1 -= PI2; // direction of the perceived object normalized between -M_PI and M_PI
                  if (angle1 < -M_PI) angle1 += PI2;
                  angle2 = angv(envobjs[ob].x2, envobjs[ob].y2, cro->x, cro->y);
                  angle2 = mangrelr(angle2, cro->dir);
                  if (angle2 > M_PI) angle2 -= PI2; // direction of the perceived object normalized between -M_PI and M_PI
                  if (angle2 < -M_PI) angle2 += PI2;

                  for(s=0, a1 = starta, a2 = starta + inta; s < cro->nlasersensors; s++, a1 += inta, a2 += inta)
                   {
                    // the pointer to the nearest point of wall is inside the sector
                    if (angle > a1 && angle < a2)
                     {
                      if (dist < mdist[s]) mdist[s] = dist;
                     }
                    else
                    {
                     // the first side point is inside the wall
                     if (anginside(a1, angle1, angle2))
                      {
                        // calculates the length of the vector between the sensor and the object providing that the angular offset is smaller than 60 degrees
						// for angular offset between 45 and 60 degrees the distance is increase linearly of a fraction of maxdist
						angoffset = fabs(angdelta(a1, angle));
						if (angoffset < (M_PI / 3.0))
                          {
						   ldist = dist * sin(angoffset);
						   cdist = sqrt((dist*dist)+(ldist*ldist));
						   if (angoffset > (M_PI / 4.0))
							  cdist += maxdist * ((angoffset - (M_PI / 4.0)) / (M_PI / 12.0));
						   if (cdist < mdist[s])
                             mdist[s] = cdist;
						   }
                       }
                     // the second side point is inside the wall
                     if (anginside(a2, angle1, angle2))
                      {
                        // calculates the length of the vector between the sensor and the object providing that the angular offset is smaller than 60 degrees
						// for angular offset between 45 and 60 degrees the distance is increase linearly of a fraction of maxdist
                        angoffset = fabs(angdelta(a2, angle));
                        if (angoffset < (M_PI / 3.0))
                          {
                           ldist = dist * sin(angoffset);
                           cdist = sqrt((dist*dist)+(ldist*ldist));
						   if (angoffset > (M_PI / 4.0))
						     cdist += maxdist * ((angoffset - (M_PI / 4.0)) / (M_PI / 12.0));
                           if (cdist < mdist[s])
                             mdist[s] = cdist;
                          }
                       }
                      }
					  
                     }
					 
                }
            break;
            // sampled cylinder
            case SAMPLEDSCYLINDER:
            break;
            // default
            //default:
            //	printf("ERROR: undefined object type %d \n",envobjs[ob].type);
            break;
         }
    }
    // calculate the stimulation generated by the cylindrical body of other robots
//double absa;
    if (nrobots > 1)
      {
        for (r=0, ro=rob; r < nrobots; r++, ro++)
          {
            if (cro->idn != ro->idn)
             {
                dx = (cro->x - ro->x);
                dy = (cro->y - ro->y);
                dist = sqrt(dx*dx+dy*dy) - (cro->radius * 2.0);
                angle = angv(ro->x, ro->y, cro->x, cro->y);
                angle = mangrelr(angle, cro->dir);
                if (angle > M_PI) angle -= PI2; // direction of the perceived object normalized between -M_PI and M_PI
                if (angle < -M_PI) angle += PI2;
//printf("robot %d absa %.2f robotdir %.2f angle %.2f dist %.2f\n", cro->idn, absa, cro->dir, angle, dist);
                if (dist < maxdist)
                 {
                   for(s=0, a1 = starta, a2 = starta + inta; s < cro->nlasersensors; s++, a1 += inta, a2 += inta)
                      {
                        // the center of the perceived robot is inside the sector
                        if (anginside(angle, a1, a2))
                         {
                          if (dist < mdist[s]) mdist[s] = dist;
//printf("center inside - robot %d angle %.2f sector %.2f %.2f dist %.2f\n", cro->idn, angle, a1, a2, dist);
                         }
                        else
                        {
                         if (dist < (cro->radius * 4.0)) // adjacent sectors can detect only in near proximity
                         {
                          // one of the side of the perceived robot is inside the maximum perceived angular offset of the sector
                          // when the distance is equal to the radius of the robot the angular offset is 45 degrees, longer the distance less the angularoffset
                          angoffset = angdelta(a1+(inta/2), angle);
                          maxperceivableaoffset = cro->radius / dist * (M_PI / 4);
                          if (maxperceivableaoffset > (M_PI / 4.0)) maxperceivableaoffset = M_PI / 4.0;  // cannot exceed 45 degrees
                          if (fabs(angoffset) < maxperceivableaoffset)
                            {
                             ldist = cro->radius * (angoffset / maxperceivableaoffset);
                             cdist = sqrt((dist*dist)+(ldist*ldist));
                             if (cdist < mdist[s])
                                mdist[s] = cdist;
//printf("side inside - robot %d angle %.2f sector %.2f %.2f dist %.2f angoffset %.2f maxperceivableoff %.2f ldist %.2f\n", cro->idn, angle, a1, a2, dist, angoffset, maxperceivableaoffset, ldist);
                            }
                           }
                        }


                      }
                 }
              }
           }
        }

    // normalize and copy on the imput vector
    for(s=0; s < cro->nlasersensors; s++)
        {
         *cro->csensors = (maxdist - mdist[s]) / maxdist;
         *cro->csensors = *cro->csensors * *cro->csensors * *cro->csensors;
         cro->csensors++;
        }
}



/*
 * save a genotype in a file
 */
void savegenotype(char* filename, double* genotype, const int glen, int mode)

{

	double *gene;
	int g;
	FILE *fp;
	
	fp = fopen(filename,"w!");
	for (g=0, gene=genotype; g < glen; g++, gene++)
	  {
	   fprintf(fp, "%lf ", *gene);
	  }
	fprintf(fp, "\n");
	fclose(fp);

}

/*
 * LIGHT SENSOR INITIALIZATION (8 sensors)
 * also load sample data from .sample file
 */
int initLightSensor(struct robot *cro)
{
    
    if (cro->idn == 0) printf("Sensor[%d]: khepera light sensors\n", 8);
    
    // init the environment and load environmental samples
    switch (cro->type)
    {
        case Khepera:
            lightcf = (int *) malloc(5 * sizeof(int));
            light = load_obstacle("khepera-light.sample", lightcf);
            break;
    }
    
    return(8);
}

/*
 * LIGHT SENSOR UPDATE
 * calculate the relative angle and distance of the light-bulb
 * and activete the sensors on the basis of the corresponding appropriate sample
 */
void updateLightSensor(struct robot *cro)

{
    
    int ob;
    double x, y, dx, dy, dist, angle;
    int relang, idist;
    float *o;
    int s;
    float act[8];
    
    // initialize
    for(s=0; s < 8; s++)
        act[s] = 0.0;
    // calculate the stimolation generated environmental objects
    for(ob=0; ob < nenvobjs; ob++)
    {
        switch(envobjs[ob].type)
        {
                // sampled walls (we assume they are orthogonal with respect to the carthesian axes)
            case LIGHTBULB:
                x = envobjs[ob].x;
                y = envobjs[ob].y;
                dx = (cro->x - x);
                dy = (cro->y - y);
                dist = sqrt(dx*dx+dy*dy);
                if (dist < (lightcf[3] + cro->radius + (lightcf[2] * lightcf[4])))
                {
                    angle = angv(x, y, cro->x, cro->y);
                    angle = mangrelr(angle, cro->dir);
                    if (angle > (M_PI * 2.0)) angle -= (M_PI * 2.0);
                    relang  = (int) round(angle / PI2 * 180.0);
                    if (relang == 180) relang = 0;
                    idist = (int) round((dist - cro->radius - lightcf[3]) / lightcf[4]);
                    if (idist < 0)
                        idist = 0;
                    if (idist >= lightcf[2])
                        idist = lightcf[2]-1;
                    o = *(wall + idist);
                    o = (o + (relang * lightcf[0]));
                    for(s=0;s<lightcf[0];s++,o++)
                    {
                        act[s] += *o;
                    }
                }
                break;
        }
    }
    // normalize and copy on the input vector
    for(s=0; s < 8; s++)
    {
        if (act[s] > 1.0)
            act[s] = 1.0;
        *cro->csensors = act[s];
        cro->csensors++;
    }
}
