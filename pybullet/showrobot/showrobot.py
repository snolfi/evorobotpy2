import pybullet as p
import pybullet_data
import time
import sys

robotfile = ""
gravity = -9.8;

if len(sys.argv) == 1:
    print("ROBOTSHOW.PY: shows a robot within the GUI")
    print("You can indicate the following robot names as parameters:")
    print("  swimmer.xml, hopper.xml, ant.xml, half_cheetah.xml, walker2d.xml, humanoid_symmetric.xml, humanoid.xml")
    print("Optional parameters:")
    print("  G0 = Gravity = 0.0")
    
if "G0" in sys.argv:
    gravity = 0.0

if len(sys.argv) >= 2:
    robotfile = sys.argv[1]

    p.connect(p.GUI)

    if not(robotfile in ["humanoid.xml"]):
        # load the plane and then the robot 
        planeId = p.loadURDF("plane.urdf")
        obUids = p.loadMJCF(robotfile)
        robot = obUids[0]
    else:
        # load a single file containing the plane and the robot
        obUids = p.loadMJCF(robotfile)
        robot = obUids[1]        

    gravId = p.addUserDebugParameter("gravity", -10, 10, gravity)
    jointIds = []
    paramIds = []

    p.setPhysicsEngineParameter(numSolverIterations=10)
    p.changeDynamics(robot, -1, linearDamping=0, angularDamping=0)

    # create the interface that enables the user to modify gravity and the positions of the jonts
    for j in range(p.getNumJoints(robot)):
      p.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
      info = p.getJointInfo(robot, j)
      print(info)
      jointName = info[1]
      jointType = info[2]
      if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        jointIds.append(j)
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

    # run the simulation and set the desired position of the joints set by the user
    p.setRealTimeSimulation(1)
    while (1):
      p.setGravity(0, 0, p.readUserDebugParameter(gravId))
      for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        p.setJointMotorControl2(robot, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
      time.sleep(0.01)

