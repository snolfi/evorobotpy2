/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * robot-env.h, includes functions for initializing the environmental objects and the robots,
 *              for simulating the sensors, and for moving the robot
 *              this .h file include the definition of the robot and envobjects structures
 *              Moreover it declare functions that can be called by other software components,
 */


#ifndef ROBOTENV_H
#define ROBOTENV_H

// Robot types
#define Khepera 0
#define ePuck   1
#define MarXBot 2

// the robot's structure
struct robot
{
	
	int idn;							// robot's identification number
	int type;                           // robot's type (Khepera, ePuck, MarXBot)
    double x;                           // robot's x position
    double y;                           // robot's y position
    double dir;                         // robot's direction in radiants
    double *sensors;                    // sensors state
    double radius;                      // Robot Radius (mm) - Khepera 27.5m, ePuch 37.5m
    double axleLength;                  // Distance between wheels (mm)
    double maxSpeed;                    // Max linear speed (mm/s)
    double *csensors;                   // pointer to the first sensor to be updated
    double dx;                          // robot's current x offset
    double dy;                          // robot's current y offset
	int color;							// 0=black, 1=red, 2=green, 3=blue, 4=red/green
	double rgbcolor[3];					// the intensity of the RGB color of the LED ring
    bool alive;                         // whether the robot is alive
    double energy;                      // the energy level of the robot
	int sensorinfraredid;				// id of the first infrared sensor neuron
	int sensorcameraid;				    // id of the first camera sensory neuron
	int sensorcameran;                  // n of infrared sensory neurons
	double leftSpeed;					// speed of the left wheel
	double rightSpeed;					// speed of the right wheel
	
	
    // motors
    int motorwheels;                    // number of motors neurons controlling the wheels
    int motorwheelstype;                // type of motor wheel control (0=direct, 1=traslation-rotation)
    int motorleds;                      // number of motors neurons controlling the wheels
	int motorwheelsid;                  // id of the first motorwheels neuron
    int motorledsid;                    // id of the first motorleds neuron

    // variables used by sensors
	double *proprioceptors;				// robots' proprioceptors state
	int nifsensors;                     // number of infrared sensors
    int nlasersensors;                  // number of laser sensors
	int camnsectors;					// the number of camera setctors
	double **camblobs;					// matrix of color blobs delected by the camera, firstlevel (sector) secondlevel (blobs)
	int *camblobsn;                     // the number of color blobs for each sector
};

// Environmental object type
#define WALL 0                          // wall type (we assume orthogonal with respect to x and y axes)
#define SAMPLEDCYLINDER 1               // sampled cylindrical objects type
#define SAMPLEDSCYLINDER 2              // sampled small cylindrical objects type
#define LIGHTBULB 3                     // bulbs emitting lights type
#define RTARGETAREA 4                   // coloured rectangular portion of the ground coloured
#define STARGETAREA 5                   // coloured spherical portion of the ground


// the environmental structure (objects contained in the environment)
struct envobjects
{
	int type;
	double x;
	double y;
	double x2;
	double y2;
	double r;
	float color[3];
};


// public variables
extern int trials;					// number of trials
extern int ttrials;					// number of testing trials
extern int steps;					// number of steps x trial
extern int stepgduration;			// time length of a step during graphic visualization
extern int nfreep;					// number of free parameters
extern int renderrobotworld;       // whether we update the graphic renderer

extern struct robot *rob;			// the list of robots is publicly available
extern int nrobots;					// number of robots
extern struct envobjects *envobjs;	// list of environemntal objects is publicly available
extern int nenvobjs;				// number of objects is publicly available
extern double worldx;				// world x dimension
extern double worldy;				// world y dimension
extern double *initialstates;		// initial states of the robots and of the environment
extern int cells[100][100];         // virtual cells
extern double cellsize;             // virtual cells size
extern float **wall;				// wall samples
extern int *wallcf;					// description of wall samples
extern float **cylinder;			// cylindrical sample
extern int *cylindercf;				// description of cylindrical sample
extern float **scylinder;			// small cylindrical sample
extern int *scylindercf;			// description of small cylindrical sample
extern float **light;				// light samples
extern int *lightcf;				// description of light samples

// public functions
void initEnvObjects(int nobjects);								 // allocate and initialize the environmental objects
void initRobot(struct robot *cro, int n, int robottype);         // intialize a robot
void initRobotSensors(struct robot *cro, int nsensors);          // intialize the sensor input vector
int updateRobot(struct robot *cro, float *actions);             // update the robot on the basis of the state of the robot motor
// each sensor has an initialization and update function
int initInfraredSensor(struct robot *rob); 						  // infrared sensor
void updateInfraredSensor(struct robot *rob);
int initEnergySensor(struct robot *ro);                           // energy sensor
void updateEnergySensor(struct robot *ro);
int initBiasSensor(struct robot *rob);                            // bias sensor
void updateBiasSensor(struct robot *rob);
int initTimeSensor(struct robot *cro);                            // time sensors
void updateTimeSensor(struct robot *cro, int step, int nsteps);
int initLaserDistanceSensor(struct robot *cro);                   // laser sensor
void updateLaserDistanceSensor(struct robot *cro);
int initLightSensor(struct robot *cro);							  // light sensor
void updateLightSensor(struct robot *cro);

#endif
