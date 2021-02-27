import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground. There's no coordinates
# in the state vector.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 90#80 # To be determined!!!

VIEWPORT_W = 1500#600
VIEWPORT_H = 600#400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 40#20    # in steps
TERRAIN_STEPNESS = 2.0#1.0 # Default 1.0
FRICTION = 2.5

BIPED_LIMIT = 1600
BIPED_HARDCORE_LIMIT = 2000

# Rate of parameter variation
MOD_RATE = 0.9 # Percentage: 0.5 --> 50%, 0.2 --> 20%, etc.
# Penalty coeff
PENALTY = 0.01 # To be tuned!
# Threshold for minimum height
THRES_H = 1.5

# Convert angle in range [-180°,180°]
def setAngleInRange(angle):
    a = angle
    # Angles belong to range [-180°,180°]
    if a > math.pi:
        a -= (2.0 * math.pi)
    if a < -math.pi:
        a += (2.0 * math.pi)
    return a

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.torso==contact.fixtureA.body or self.env.torso==contact.fixtureB.body:
            self.env.torso.ground_contact = True
        for seg in self.env.segs:
            if seg in [contact.fixtureA.body, contact.fixtureB.body]:
                seg.ground_contact = True
    def EndContact(self, contact):
        for seg in self.env.segs:
            if seg in [contact.fixtureA.body, contact.fixtureB.body]:
                seg.ground_contact = False

class customEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False
    # Task parameters
    numSegs = 10 # Number of segment does not include torso
    segSpeed = 5.0
    segWidth = 24.0
    segHeight = 6.0
    segDensity = 5.0
    jointRange = math.pi / 2.0
    initHeightFactor = 3.0

    def __init__(self):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.torso = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )

        # Morphology parameters
        self.nsizes = 2 + (2 * self.numSegs)
        self.nangleranges = (2 * self.numSegs)
        self.nangles = (1 + self.numSegs)
        self.ndensities = (1 + self.numSegs)
        self.njoints = self.numSegs # number of joints matches number of segments (there is one joint between each pair of segments including torso)
        self.nparams = (self.nsizes + self.nangleranges + self.nangles + self.ndensities) # Torso width and height + 2 for each segment (width and height) + 2 for each joint (asymmetric angular ranges) except for the head + torso angle + one for each segment angle + torso density + one for each segment density. N.B.: number of joints matches number of segments (there is one joint between each pair of segments including torso)
        self.params = np.zeros(self.nparams, dtype=np.float64)#np.ones(self.nparams, dtype=np.float64)
        self.factors = np.ones(self.nparams, dtype=np.float64) # 2 for each segment (width and height) + 2 for each joint (asymmetric angular ranges) + torso angle + segment angle
        self.rate = 0.0

        # Test flag
        self.test = False

        self.reset()

        # Number of observations
        ob_len = 5 + (self.njoints * 2) + self.numSegs # 5 for torso, njoints * 2 for each joint, contact flag for each segment
        # Number of actions
        ac_len = self.njoints # 1 action for each joint
        act = np.ones(ac_len, dtype=np.float32)
        high = np.array([np.inf]*ob_len, dtype=np.float32)# 24)
        self.action_space = spaces.Box(-act,act)#np.array([-1,-1,-1,-1]), np.array([+1,+1,+1,+1]))
        self.observation_space = spaces.Box(-high, high)

        self.timer = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.torso)
        self.torso = None
        for seg in self.segs:
            self.world.DestroyBody(seg)
        self.segs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE*TERRAIN_STEPNESS   #1
                y += velocity

            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def getNumParams(self):
        return self.nparams

    def setParams(self, params):
        self.params = params
        # And the rate
        self.rate = MOD_RATE

    def setTest(self):
        # Set test flag
        self.test = True

    def lowestHeight(self):
        lowest = self.torso.position[1]
        for i in range(self.numSegs):
            if self.segs[i].position[1] < lowest:
                lowest = self.segs[i].position[1]
        return lowest

    def computeFactor(self, val, idx):
        # Parameters list:
        # - torso width and height
        # - segment width and height (one pair for each segment)
        # - joint angular ranges (two for each joint)
        # - torso angle
        # - segment angles
        # - torso density
        # - segment density (one for each segment)
        factor = 0.0
        if idx < self.nsizes:
            # Segment sizes
            # Body sizes (width and height of segments) are scaled depending on the parameter
            # (1 + p * rate)
            factor = (val * (1.0 + np.tanh(self.params[idx]) * self.rate))
        elif idx >= (self.nsizes + self.nangleranges + self.nangles):
            # Segment densities
            # Density factors are computed as tanh(p), where p is the density parameter
            factor = (val + (0.8 * np.tanh(self.params[idx])))
        else:
            # Joint ranges and segment angles
            # Angle factors must belong to range [-1,1]. Values
            # outside boundaries are cut
            param = self.params[idx]
            if param < -1.0:
                param = -1.0
            if param > 1.0:
                param = 1.0
            factor = (param * val)
        return factor

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0

        self.timer = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        # Scale factor
        U = 1.0 / SCALE # 1/30

        # Segment width and height (inverted!!!)
        seg_w = self.segWidth*U # 0.2
        seg_h = self.segHeight*U # 0.8
        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        max_h = seg_h
        init_y = TERRAIN_HEIGHT + (self.initHeightFactor * max_h)

        self.segs = []
        self.joints = []

        # Torso sizes
        torsoWidth = self.computeFactor(seg_w, 0)
        torsoHeight = self.computeFactor(seg_h, 1)

        torsoAngle = self.computeFactor((3.0 * math.pi / 4.0), (self.nsizes + self.nangleranges))
        torsoAngle = setAngleInRange(torsoAngle)
        # Create first segment (i.e., torso) out of the for loop
        self.torso = self.world.CreateDynamicBody(
                position = (init_x, init_y), # (4.6667, 3.3333)
                angle = torsoAngle, # 0: vertical, 90: horizontal
                fixtures = fixtureDef(
                            shape=polygonShape(box=(torsoWidth/2, torsoHeight/2)), # Width and height inverted (see rendering)
                            density=self.computeFactor(self.segDensity, self.nsizes + self.nangleranges + self.nangles), # Density can be increased/decreased up to 80%
                            restitution=0.0,
                            categoryBits=0x0020,
                            maskBits=0x001)
                )
        self.torso.color1 = (0.5,0.4,0.9)
        self.torso.color2 = (0.3,0.3,0.5)
        self.torso.ground_contact = False

        # Create remaining segments
        half_segs = int(float(self.numSegs - 1) / 2.0)
        prevSeg = self.torso # Previous segment
        seg = None
        rjd = None
        # Indices for parameters/modifications
        k = 2 # Body sizes
        h = self.nsizes # Joint ranges
        z = (self.nsizes + self.nangleranges + 1) # Relative angles
        w = (self.nsizes + self.nangleranges + self.nangles) # Densities
        # Previous sizes
        prevWidth = torsoWidth
        prevHeight = torsoHeight
        prevAngle = torsoAngle
        # Previous point coords
        tmpX = init_x - torsoWidth / 2.0 * math.cos(torsoAngle)# - torsoHeight / 2.0 * math.sin(torsoAngle)
        tmpY = init_y - torsoWidth / 2.0 * math.sin(torsoAngle)# - torsoHeight / 2.0 * math.cos(torsoAngle)
        for i in range(self.numSegs):
            # New sizes
            currWidth = self.computeFactor(seg_w, k)
            currHeight = self.computeFactor(seg_h, k + 1)
            # New joint ranges
            currLowerAngle = self.computeFactor(self.jointRange, h)
            currUpperAngle = self.computeFactor(self.jointRange, h + 1)
            # Check lower and upper angles
            if currLowerAngle > currUpperAngle:
                # Switch lower and upper
                tmp = currLowerAngle
                currLowerAngle = currUpperAngle
                currUpperAngle = tmp
            currAngle = self.computeFactor((3.0 * math.pi / 4.0), z)
            currAngle = setAngleInRange(currAngle)
            # Body segment
            seg = self.world.CreateDynamicBody(
                    #position = (tmpX - currWidth / 2.0 * math.cos(currAngle) - currHeight / 2.0 * math.sin(currAngle), tmpY - currWidth / 2.0 * math.sin(currAngle) - currHeight / 2.0 * math.cos(currAngle)),
                    position = (tmpX - currWidth / 2.0 * math.cos(currAngle), tmpY - currWidth / 2.0 * math.sin(currAngle)),
                    angle = currAngle,
                    fixtures = fixtureDef(
                                shape=polygonShape(box=(currWidth/2, currHeight/2)),
                                density=self.computeFactor(self.segDensity, w),
                                restitution=0.0,
                                categoryBits=0x0020,
                                maskBits=0x001)
                     )
            if i % 2 == 0:
                seg.color1 = (1.0,0.1,0.2)
                seg.color2 = (0.8,0.2,0.6)
            else:
                seg.color1 = (0.5,0.4,0.9)
                seg.color2 = (0.3,0.3,0.5)
            # Joint
            rjd = revoluteJointDef(
                    bodyA=prevSeg,
                    bodyB=seg,
                    localAnchorA=(-prevWidth/2,0),#-prevWidth/2*math.sin(prevAngle)),
                    localAnchorB=(currWidth/2,0),#currWidth/2*math.cos(currAngle)),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=MOTORS_TORQUE,
                    motorSpeed = i,
                    lowerAngle = currLowerAngle,
                    upperAngle = currUpperAngle,
                   )
            seg.ground_contact = False
            # Append segment to list of segments
            self.segs.append(seg)
            # Append joint to list of joints
            self.joints.append(self.world.CreateJoint(rjd))
            # Update previous segment
            prevSeg = seg
            # Update previous sizes
            prevWidth = currWidth
            prevHeight = currHeight
            # Set point coords
            tmpX = tmpX - currWidth * math.cos(currAngle)# - currHeight * math.sin(currAngle)
            tmpY = tmpY - currWidth * math.sin(currAngle)# - currHeight * math.cos(currAngle)
            # Update indices
            k += 2
            h += 2
            z += 1
            w += 1
        # Check number of joints
        assert len(self.joints) == self.njoints

        # List of objects to display
        self.drawlist = self.terrain + self.segs + [self.torso]

        fakeAction = [0.0 for _ in range(self.njoints)]

        # Initialize first contact with ground flag
        self.firstContact = False

        return self.step(np.array(fakeAction))[0] # self._step

    def step(self, action): # _step
        # Apply actions
        j = 0
        for i in range(len(action)):
            self.joints[j].motorSpeed = float(self.segSpeed * np.sign(action[i]))
            self.joints[j].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[i]), 0, 1))
            j += 1
        self.world.Step(1.0/FPS, 6*30, 2*30)
        #import time
        #time.sleep(0.5)

        pos = self.torso.position
        vel = self.torso.linearVelocity

        # Add torso info
        state = [
            self.torso.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.torso.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            1.0 if self.torso.ground_contact else 0.0
            ]
        # Add joint info
        joint_state = []
        for i in range(self.njoints):
            joint_state += [(self.joints[i].angle)]# + j)]
            joint_state += [self.joints[i].speed / self.segSpeed]
        state += joint_state
        # Add contact info
        seg_contact_state = []
        for i in range(self.numSegs):
            seg_contact_state += [1.0 if self.segs[i].ground_contact else 0.0]
        state += seg_contact_state

        # Check state array size
        assert len(state) == (5 + (self.njoints * 2) + self.numSegs)

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        """
        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less
        """
        # Introduces a penalty for high velocities!!!
        """
        excess = 0.0
        for i in range(len(self.joints)):
            if abs(self.joints[i].speed) > segSpeed:
                excess += (abs(self.joints[i].speed) - segSpeed) / (2.0 * segSpeed)
        reward -= (PENALTY * excess * excess)
        """

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True

        if self.hardcore:
          if self.timer >= BIPED_HARDCORE_LIMIT:
            done = True
        else:
          if self.timer >= BIPED_LIMIT:
            done = True

        if TERRAIN_STEPNESS == 1.0:
            # Check whether or not the embryo has one segment below the terrain
            lowest = self.lowestHeight()
            if lowest < TERRAIN_HEIGHT:
                done = True

        self.timer += 1

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human', close=False): # _render
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

# Maybe can be eliminated
if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = customEnv()
    augment_vector = (1.0 + (np.random.rand(8)*2-1.0)*0.5)
    print("augment_vector", augment_vector)
    env.augment_env(augment_vector)
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break
