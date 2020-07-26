/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * discrim.cpp, include an implementation of the discrim environment

 * This file is part of the python module ErDiscrim.so that include the following files:
 * discrim.cpp, discrim.h, robot-env.cpp, robot-env.h, utilities.cpp, utilities.h, ErDiscrim.pxd, ErDiscrim.pyx and setupErDiscrim.py
 * And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErDiscrim.py build_ext –inplace; cp ErDiscrim*.so ../bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "discrim.h"
#include "utilities.h"
#include "robot-env.h"

// Pointer to the observations
float* cobservation;
// Pointer to the actions
float* caction;
// Pointer to termination flag
int* cdone;
// Pointer to world objects to be rendered
double* dobjects;


#define MAX_STR_LEN 1024

/*
 * env constructor
 */
Problem::Problem()
{

	// create the environment
	initEnvironment();
	// define the number of robots situated in the environment
	nrobots = 1;
    // creates the list of robot structures that contain the robot data
    // the list include a single element when we have a single robot
    // the robot structure and associated variables are define in the file robot-env.h
    rob = (struct robot *) malloc(nrobots * sizeof(struct robot));
    // initilize robot variable (e.g. the size of the robots' radius, the maximum speed of the robots' wheels ext).
    // the initRobot function is defined in the robot-env.cpp file
	initRobot(rob, 0, Khepera);
    // initialize sensors set the number of input neurons of each network
    // the number of neurons required depends on the number of input units required by each sensor
    // which is returned by the sensors initialization function
	ninputs = 0;
    ninputs += initInfraredSensor(rob);   		// infrared sensor (8 units required to store the value of the corresponding 8 IF sensors)
	rob->sensorinfraredid = 0; 			        // the id of the first infrared sensors (used for graphic purpose only to visually siaplay infrared activity)
	// allocate and initialize the robot->sensor vector that contain net->ninputs values and is passed to the function that update the state of the robots' network
	initRobotSensors(rob, ninputs);
    // define the motors used and set the number of motor neurons
    // currently evorobot support only 2 wheel motors and active LEDs
    // motorleds can be set to 1,2, or 3 to control 1 to 3 colors (1=red, 2-red/green, 3=red/green/blue)
    rob->motorwheels = 2;
    rob->motorleds = 0;
    noutputs = rob->motorwheels + rob->motorleds;
    // set the id number of the first wheel motor neurons and of the first led motor neuron (if present)
    // the networks include ninputs+nhiddens+noutputs neurons with IDs ranging from 0 to nneurons-1 in the corresponding order
    rob->motorwheelsid = ninputs + 10; // +net->nhiddens;
    rob->motorledsid = 0;
	
	rng = new RandomGenerator(time(NULL));
	
	
}


Problem::~Problem()
{
}


/*
 * set the seed
 */
void Problem::seed(int s)
{
    	rng->setSeed(s);
}

/*
 * reset the initial condition randomly
 * when seed is different from 0, reset the seed
 */
void Problem::reset()
{

    // position of the cylindrical object
	envobjs[0].x = rng->getDouble(100.0, worldx - 100.0);
	envobjs[0].y = rng->getDouble(100.0, worldy/2.0);
	// orientation and position of the robot
	rob->dir = rng->getDouble(0.0, PI2);
	rob->x = worldx * 0.5 + rng->getDouble(-50, 50);
	rob->y = worldy * 0.8 + rng->getDouble(-50, 50);

	// Get observations
	getObs();
	
}


void Problem::copyObs(float* observation)
{
	cobservation = observation;
}

void Problem::copyAct(float* action)
{
	caction = action;
}

void Problem::copyDone(int* done)
{
	cdone = done;
}

void Problem::copyDobj(double* objs)
{
	dobjects = objs;
}


/*
 * perform the action, update the state of the environment, update observations, return the reward
 */
double Problem::step()
{

    double dx, dy;
    double dist;
	
    *cdone = 0;
	if (updateRobot(rob, caction) != 0)
		*cdone = 1;
	
	getObs();
	
	//printf("xyd %.1f %.1f %.1f  o %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f  a %.1f %.1f done %.1f\n",
	//   rob->x, rob->y, rob->dir, cobservation[0],cobservation[1],cobservation[2],cobservation[3],
	//   cobservation[4],cobservation[5],cobservation[6],cobservation[7],caction[0], caction[1], *cdone);
    
    /*
     * return 1.0 when the robot is near the cylinder
     * and is not colliding
     */
    if (*cdone == 1)
      {
        return(0);
      }
      else
      {
        dx = envobjs[0].x - rob->x;
        dy = envobjs[0].y - rob->y;
        dist = sqrt((dx*dx)+(dy*dy));
        if (dist < (rob->radius + envobjs[0].r + 60))
           return(1.0);
         else
           return(0.0);
      }

}

double Problem::isDone()
{
	return *cdone;
}

void Problem::close()
{
    	//printf("close() not implemented\n");
}

/*
 * create the list of robots and environmental objects to be rendered graphically
 */
void Problem::render()
{
    
    int i;
    int c;
    struct robot *ro;
    
    c=0;
    // robots
    for (i=0, ro = rob; i < nrobots; i++, ro++)
    {
        dobjects[c] = 1.0;
        dobjects[c+1] = ro->x;
        dobjects[c+2] = ro->y;
        dobjects[c+3] = ro->radius;
        dobjects[c+4] = 0.0;
        dobjects[c+5] = ro->rgbcolor[0];
        dobjects[c+6] = ro->rgbcolor[1];
        dobjects[c+7] = ro->rgbcolor[2];
        dobjects[c+8] = ro->x + xvect(ro->dir, ro->radius);
        dobjects[c+9] = ro->y + yvect(ro->dir, ro->radius);
        c += 10;
    }
    for(i=0; i < nenvobjs; i++)
    {
        switch(envobjs[i].type)
        {
            case SAMPLEDSCYLINDER:
                dobjects[c] = 3.0;
                dobjects[c+3] = envobjs[i].r;
                dobjects[c+4] = 0.0;
                dobjects[c+8] = 0.0;
                dobjects[c+9] = 0.0;
                break;
            case WALL:
                dobjects[c] = 2.0;
                dobjects[c+3] = envobjs[i].x2;
                dobjects[c+4] = envobjs[i].y2;
                dobjects[c+8] = 0.0;
                dobjects[c+9] = 0.0;
                break;
        }
        dobjects[c+1] = envobjs[i].x;
        dobjects[c+2] = envobjs[i].y;
        dobjects[c+5] = envobjs[i].color[0];
        dobjects[c+6] = envobjs[i].color[1];
        dobjects[c+7] = envobjs[i].color[2];
        c += 10;
    }
    dobjects[c] = 0.0;
    
}

/*
 * update observation vector
 * ROB->SENSORS SHOULD POINT DIRECTLY TO COBSERVATION, WE DO NOT NEED THE FOR IN THAT CASE
 */
void Problem::getObs()
{

	// during each step reset the pointer of the input pattern that is later updated by the sensor function
	rob->csensors = rob->sensors;
	// update the input pattern by calling the sensor function (in this case we are using sigle sensor fuction)
	// the sensor function used here should be initialized in the initialize() function
	updateInfraredSensor(rob);

	// add noise to observations
	int i;
	for(i=0, rob->csensors = rob->sensors; i < ninputs; i++, rob->csensors++)
	  cobservation[i] = *rob->csensors += rng->getGaussian(0.03, 0.0);
	
}



/*
 * Initialize the environment
 * Environment are costituted by a rectangular area of size worldx*worldy and by a list of objects (walls, cylinders, coloured portions of the ground ext).
 * which are stored in a list of envojects structures (see the file robot-env.h for the definition of the structure)
 * Consequently thus function:
 * set the size of the arena, allocate the objects, and set the relevant characteristics of the objects
 */
void
Problem::initEnvironment()

{

	int cobj=0;
	
	// set the size of the arena
	worldx = 400.0;		// world x dimension in mm
	worldy = 400.0;		// world y dimension in mm
	
	// allocate the objects
	nenvobjs = 5;		// number of objects (1 cylindrical object and 4 wall objects)
	initEnvObjects(nenvobjs); // allocate and intilialize the environmental objects
	
	// set the properties of the objects
	// object 0 is a cylinder, x and y indicate the position of the barycenter of the cylinder, r indicate the radius
	envobjs[cobj].type = SAMPLEDSCYLINDER;
	envobjs[cobj].x = worldx/2.0;
	envobjs[cobj].y = worldy/2.0;
	envobjs[cobj].r = 12.5;
	cobj++;
	// object 1 is a wall, x and y indicate the position of the begin of the wall, x2 and y2 the position of the end of the wall
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = 0.0;
	envobjs[cobj].y = 0.0;
	envobjs[cobj].x2 = worldx;
	envobjs[cobj].y2 = 0.0;
	cobj++;
	// object 2 is a wall
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = 0.0;
	envobjs[cobj].y = 0.0;
	envobjs[cobj].x2 = 0.0;
	envobjs[cobj].y2 = worldy;
	cobj++;
	// object 3 is a wall
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = worldx;
	envobjs[cobj].y = 0.0;
	envobjs[cobj].x2 = worldx;
	envobjs[cobj].y2 = worldy;
	cobj++;
	// object 4 is a wall
	envobjs[cobj].type = WALL;
	envobjs[cobj].x = 0.0;
	envobjs[cobj].y = worldy;
	envobjs[cobj].x2 = worldx;
	envobjs[cobj].y2 = worldy;
	cobj++;
	// the four wall surround the worldx*worldy area
	
	if (cobj > nenvobjs)
		{
			printf("ERROR: you should allocate more space for environmental objects");
			fflush(stdout);
			exit;
		}
	
}




