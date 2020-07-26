#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it

   renderWorld.py include methods for rendering the activations of neurons
   and methods for rendering the behavior Er environments
   to that described in:

   Requires pyglet

   Parts of the methods are taken from:
   https://github.com/adamwiggins/cocos2d/blob/master/tools/primitives.py
   copyright 2007 by Flavio Codeco Coelho
 
"""

from pyglet import font
from pyglet import clock
from pyglet import window
from pyglet import image
from pyglet.gl import *
from pyglet.window import mouse
from pyglet.window import event
from pyglet.window import key
import numpy as np


win = None


"""
# world data to be displayed
# type, x,y,dir/rad/x2, y2, red, green, blue, ?, ?
wobj = np.arange(1000, dtype=np.float64)

c = 0
#robots
for r in range(5):
    wobj[c] = 1
    wobj[c+1] = 0 + r*40
    wobj[c+2] = 0 + r*40
    wobj[c+3] = 10
    wobj[c+4] = 1.0
    wobj[c+5] = 1.0
    wobj[c+6] = 0.0
    wobj[c+7] = 0.0
    wobj[c+8] = 0.0
    wobj[c+9] = 0.0
    c=c+10
#walls
for w in range(4):
    wobj[c] = 2
    wobj[c+1] = 0 
    wobj[c+2] = 0 + w * 20
    wobj[c+3] = 200.1
    wobj[c+4] = 0 + w * 20
    wobj[c+5] = 0.0
    wobj[c+6] = 1.0
    wobj[c+7] = 0.0
    wobj[c+8] = 0.0
    wobj[c+9] = 0.0
    c=c+10
wobj[c] = 0
"""





class Base(object):
    """
    Basic attributes of all drawing primitives
    """
    def __init__(self, x, y, z=0, color=(0.,0.,0.,1.), stroke=0, rotation=0):
        try :
            self.rect
        except AttributeError:
            self.rect = Rect(x,y,1,1) # this inits x,y and loc as well
        self.visible = 1 #
        self.z = z 
        self.rotation = rotation
        self.stroke = stroke
        self.color = color
        self.q = gluNewQuadric()
    def setLoc(self, p) : self.rect.loc = p 
    def getLoc(self) : return self.rect.loc
    def setX(self, x) : self.rect.x = x 
    def getX(self) : return self.rect.x 
    def setY(self, y) : self.rect.y = y 
    def getY(self) : return self.rect.y
    loc = property(getLoc, setLoc)
    x = property(getX, setX)
    y = property(getY, setY)
    def setWidth(self,w) : self.rect.width = w
    def getWidth(self) : return self.rect.width
    def setHeight(self, h) : self.rect.height = h
    def getHeight(self) : return self.rect.height
    width = property(getWidth, setWidth)
    height = property(getHeight, setHeight)



class Pixel(Base):
    """ A pixel at a given x,y,z position and color.
        Pixel(x=12, y=100, z=900, color=(1,0,0,0.5))
    """
    def render(self):
        """
            Draws a pixel at a given x and y with given color .
            Color = 3 or 4 arg tuple. RGB values from 0 to 1 being 1 max value (1, 1, 1) would be white
        """
        glColor4f(*self.color)
##        glDisable(GL_TEXTURE_2D) # disable in case it was on
            
        glPushMatrix() # remember previous matrix state before translating, rotating
        glTranslatef(self.x, self.y, -self.z) # translate to point to draw

        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

        glPopMatrix() # back to previous matrix state

    def intersects(self, x,y):
        if x==self.x and y==self.y : return True

#display an empty circle
class Circle(Base):
    """ Circle class
        Circle(x=20, y=100, z=1, width=300, color=(1,1,0,0.3), stroke=5, rotation=0, style=GLU_FILL)
        style choices are : GLU_LINE, GLU_FILL, GLU_SILHOUETTE, GLU_POINT
    """
    def __init__(self, x=10, y=10, z=0, width=2, color=(0,0,0,1), stroke=0, rotation=0.0, style=GLU_SILHOUETTE):
        self.radius = width*0.5
        self.rect = Rect(x, y, width, width)
        self.style = style
        self.circleresolution = 60
        Base.__init__(self, x,y,z,color, stroke, rotation)
        
    def setWidth(self, w):
        self.radius = w*0.5
        self.rect.width = w
    width = property(Base.getWidth, setWidth)
        
    def render(self):
        """ Draw Circle
            x, y, z, width in pixel, rotation, color and line width in px
            style choices are : GLU_LINE, GLU_FILL, GLU_SILHOUETTE, GLU_POINT
            TO DO : textured circles
        """
        
        glColor4f(*self.color)
        glPushMatrix()

        glTranslatef(self.x, self.y, -self.z)
        glRotatef(self.rotation, 0, 0, 0.1)


        if self.radius < 1 : self.radius = 1

        if self.stroke :
            inner = self.radius - self.stroke # outline width
            if inner < 0: inner=0
        else :
             inner = 0 # filled
        
        gluQuadricDrawStyle(self.q, self.style)

        gluDisk(self.q, inner, self.radius, self.circleresolution, 1) # gluDisk(quad, inner, outer, slices, loops)
            
        glPopMatrix()


#display a full circle
class Disk(Base):
    """ Circle class
        Circle(x=20, y=100, z=1, width=300, color=(1,1,0,0.3), stroke=5, rotation=0, style=GLU_FILL)
        style choices are : GLU_LINE, GLU_FILL, GLU_SILHOUETTE, GLU_POINT
    """
    def __init__(self, x=10, y=10, z=0, width=2, color=(0,0,0,1), stroke=0, rotation=0.0, style=GLU_FILL):
        self.radius = width*0.5
        self.rect = Rect(x, y, width, width)
        self.style = style
        self.circleresolution = 60
        Base.__init__(self, x,y,z,color, stroke, rotation)
        
    def setWidth(self, w):
        self.radius = w*0.5
        self.rect.width = w
    width = property(Base.getWidth, setWidth)
        
    def render(self):
        """ Draw Circle
            x, y, z, width in pixel, rotation, color and line width in px
            style choices are : GLU_LINE, GLU_FILL, GLU_SILHOUETTE, GLU_POINT
            TO DO : textured circles
        """
        
        glColor4f(*self.color)
        glPushMatrix()

        glTranslatef(self.x, self.y, -self.z)
        glRotatef(self.rotation, 0, 0, 0.1)


        if self.radius < 1 : self.radius = 1

        if self.stroke :
            inner = self.radius - self.stroke # outline width
            if inner < 0: inner=0
        else :
             inner = 0 # filled
        
        gluQuadricDrawStyle(self.q, self.style)

        gluDisk(self.q, inner, self.radius, self.circleresolution, 1) # gluDisk(quad, inner, outer, slices, loops)
            
        glPopMatrix()


class Arc(Base):
    """ Arc class
        Arc(x=10, y=10, z=0, radius=1, start=0, sweep=1, color=(0,0,0,1), stroke=0, rotation=0.0, style=GLU_FILL)
        style choices are : GLU_LINE, GLU_FILL, GLU_SILHOUETTE, GLU_POINT
    """
    def __init__(self, x=10, y=10, z=0, radius=1, start=0, sweep=1, color=(0,0,0,1), stroke=0,
                 rotation=0.0, style=GLU_FILL):

        Base.__init__(self, x,y,z,color, stroke, rotation)
        self.radius = radius
        self.start = start
        self.sweep = sweep
        self.style = style
        self.circleresolution = 60
        
    def render(self):
        """
        Draws Arc
        """
        glColor4f(*self.color)
        glPushMatrix()

        glTranslatef(self.x, self.y, -self.z)
        glRotatef(self.rotation, 0, 0, 0.1)

        if self.stroke : 
            inner = self.radius - self.stroke
            if inner < 0: inner=0
        else :
            inner = 0 # full, no inner
        self.start -= 180
        
        gluQuadricDrawStyle(self.q, self.style)
        
        gluPartialDisk(self.q, inner, self.radius, self.circleresolution, 1, self.start, self.sweep)
        
        glPopMatrix()



class Polygon(Base):
    def __init__(self, v, z=0, color=(0,0,0,1), stroke=0, rotation=0.0, style=0):
        """ polygon class
            Polygon(vertexarray=[(0, 0), (29, 100), (30, 200)], z=100, color=(0,0.3,0.1,1), stroke=0, rotation=23)
            overwrites few methods from superclass as polygons are more complex, needs to update everyvertex.
        """
        self.v = v
        l, t, r, b = calcPolygonRect(v) # get the bounding rect
        self.rect = Rect(l+(r-l)*0.5, t+(b-t)*0.5, r-l, b-t)
        self.v2 = [(i[0] - self.rect.x, i[1] - self.rect.y) for i in v] #relative polygon

        self.style = style
        
        Base.__init__(self, self.rect.x, self.rect.y, z,color,stroke,rotation)

    def updateV(self):
        self.v = [(self.rect.x + n[0], self.rect.y + n[1]) for n in self.v2]

    def setLoc(self, p):
        self.rect.loc = p ; self.updateV()
    def setX(self, x):
        self.rect.x = x ; self.updateV()
    def setY(self, y):
        self.rect.y = y; self.updateV()
    x = property(Base.getX, setX)
    y = property(Base.getY, setY)
    loc = property(Base.getLoc, setLoc)
    
    def render(self):
        """ Draw Polygon
            v is an array with tuple points like [(x, y), (x2, y2), (x3, y3)]
            min vertex number to draw a polygon is 3
            stroke=0 to fil with color the inside of the shape or stroke=N just to draw N-px thick outline.
            Note. It doesnt work with non convex polygons, need to implement tesselation yet
        """
        l,t,r,b = calcPolygonRect(self.v)
        x,y = calcRectCenter(l,t,r,b)
        self.drawVertex(x, y, self.z, [(i[0] - x, i[1] - y) for i in self.v], self.color, self.stroke, self.rotation, self.style)

    
    def drawVertex(self, x, y,  z=0, v=(), color=(0,0,0,1), stroke=0, rotation=0.0,   style=0):
        
        glColor4f(*self.color)    
        glPushMatrix()

        glTranslatef(x, y, -z)
        glRotatef(self.rotation, 0, 0, 0.1)

        if self.style :
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(1, style)
##        else :
##            glDisable(GL_LINE_STIPPLE)
##            0xF0F0 # dashed line
##            0xF00F # long dashed line
##            0x8888 # dotted lines
##        glRect(x1,y,1,x1,x2)
##        glRectiv(v1,v2) # oposite vertex of rectangle
        # -- start drawing
        if self.stroke : # outlined polygon
            glLineWidth(self.stroke)
            glBegin(GL_LINE_LOOP)
        else: # filled polygon
            if   len(v) == 4 : glBegin(GL_QUADS)
            elif len(v)  > 4 : glBegin(GL_POLYGON)
            else :             glBegin(GL_TRIANGLES) # which type of polygon are we drawing?

        for p in v:
            glVertex3f(p[0], p[1],0)  # draw each vertex

        glEnd()
        # -- end drawing
        
        if self.style : glDisable(GL_LINE_STIPPLE)
        
        glPopMatrix()


class LineRel(Base):
    def __init__(self, x,y, a=(0,0), b=(0,0), z=0, color=(0,0,0,1), stroke=0, rotation=0.0, style=0):
        """ Draws a basic line given the begining and end point (tuples), color (tuple) and stroke
            (thickness of line)
            Line( x,y, a=(1,1), b=(100,100), z=0, color=(0.2,0,0,1), stroke=10, rotation=45)
        """
        w = (b[0] - a[0]) 
        h = (b[1] - a[1]) 
        x = abs(a[0] + w*0.5)
        y = abs(a[1] + h*0.5)
        self.a2 = abs(a[0]) - x, abs(a[1]) - y
        self.b2 = abs(b[0]) - x, abs(b[1]) - y
        self.a = x - w*0.5, y - w*0.5
        self.b = x + w*0.5, y + w*0.5
        self.rect = Rect(x, y, w, h)
        self.style = style
        Base.__init__(self, x, y, z,color,stroke,rotation)

    def render(self):
        """
        Draws Line
        """
        p1 = self.a2
        p2 = self.b2
        glColor4f(*self.color)
        color  = (GLfloat *4)(*self.color)
               
        glPushMatrix()

        glTranslatef(self.x, self.y, -self.z) # translate to GL loc ppint
        glRotatef(self.rotation, 0, 0, 0.1)

        if self.style :
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(1, self.style)
##        else :
##            glDisable(GL_LINE_STIPPLE)
            
        if self.stroke <= 0:
            self.stroke = 1
        glLineWidth(self.stroke)

        glBegin(GL_LINES)
        glVertex2i(int(p1[0]), int(p1[1])) # draw pixel points
        glVertex2i(int(p2[0]), int(p2[1]))
        glEnd()

        if self.style :
            glDisable(GL_LINE_STIPPLE)
        
        glPopMatrix()


    def updateAB(self):
        self.a = self.x + self.a[0], self.y + self.a[0]
        self.b = self.x + self.b[0], self.y + self.b[0]

    def setLoc(self, p):
        self.rect.loc = p ; self.updateAB()
    def setX(self, x):
        self.rect.x = x ; self.updateAB()
    def setY(self, y):
        self.rect.y = y; self.updateAB()
    x = property(Base.getX, setX)
    y = property(Base.getY, setY)
    loc = property(Base.getLoc, setLoc)


class Line(LineRel):
    def __init__(self, a=(0,0), b=(0,0), z=0, color=(0,0,0,1), stroke=0, rotation=0.0, style=0):
        """ Draws a basic line given the begining and end point (tuples), color (tuple) and stroke
            (thickness of line)
            Line( a=(1,1), b=(100,100), z=20, color=(0.2,0,0,1), stroke=10, rotation=45)
        """
        w = (b[0] - a[0]) 
        h = (b[1] - a[1]) 
        x = abs(a[0] + w*0.5) # abs x,y
        y = abs(a[1] + h*0.5)
        a = x-w*0.5, y-h*0.5 # relative a,b
        b = x+w*0.5, y+h*0.5
        LineRel.__init__(self, x, y, a, b, z, color, stroke, rotation, style)

        

# intersectable shapes (draggable)

class Rect(object):
    def __init__(self, x=0,y=0,w=0,h=0):
        """ rect(self, x=0,y=0,w=0,h=0)
            x,y,loc, width, height
            left,top,right,bottom
            quad ->
            --------------------
            topleft = quad[0]
            topright = quad[1]
            bottomright = quad[2]
            bottomleft = quad[3]
        """
        self.rect = x,y,w,h

    def setRect(self, r):
        self.__x = r[0]
        self.__y = r[1]
        self.__width = r[2]
        self.__height = r[3]
        w = r[2]*0.5 ; h = r[3]*0.5
        self.__rect = r[0]-w, r[1]-h, r[0]+w, r[1]+h # l t r b
    def getRect(self):
        return self.__rect
    rect = property(getRect, setRect)

    def setQuad(self, q): # [ q[0][0], q[0][1], q[1][0], q[2][1] ] # l t r b
        self.rect = q[0][0]+(q[1][0]-q[0][0])*0.5, q[0][1]+(q[2][1]-q[0][1])*0.5, q[1][0]-q[0][0], q[2][1]-q[0][1] 
    def getQuad(self):
        return [(self.rect[0], self.rect[1]),(self.rect[2], self.rect[1]),(self.rect[2], self.rect[3]),(self.rect[0], self.rect[3])] # tl tr br bl
    quad = property(getQuad, setQuad)

    def setX(self, x):
        self.rect = x, self.y, self.width, self.height
    def getX(self) : return self.__x
    x = property(getX, setX)

    def setY(self, y):
        self.rect = self.x, y, self.width, self.height
    def getY(self) : return self.__y
    y = property(getY, setY)

    def setLoc(self, p):
        self.rect = p[0], p[1], self.width, self.height
    def getLoc(self) : return self.__x, self.__y # self.x, self.y
    loc = property(getLoc, setLoc)

    def setWidth(self, w):
        self.rect = self.x, self.y, w, self.height
    def getWidth(self): return self.__width
    width = property(getWidth, setWidth)
    
    def setHeight(self, h):
        self.rect = self.x, self.y, self.width, h
    def getHeight(self): return self.__height
    height = property(getHeight, setHeight)

    def setLeft(self,x):
        self.rect = x+self.width*0.5, self.y, self.width, self.height
    def getLeft(self): return self.rect[0]
    left = property(getLeft, setLeft)
    
    def setTop(self,y):
        self.rect = self.x, y+self.width*0.5, self.width, self.height
    def getTop(self): return self.rect[1]
    top = property(getTop, setTop)
    
    def setRight(self,x):
        self.rect = x-self.width*0.5, self.y, self.width, self.height
    def getRight(self): return self.rect[2]
    right = property(getRight, setRight)
    
    def setBottom(self,x):
        self.rect = self.x, y-self.width*0.5, self.width, self.height
    def getBottom(self): return self.rect[3]
    bottom = property(getBottom, setBottom)


def calcPolygonRect(pointArray):
    """ receives a point list and returns the rect that contains them as a tupple -> tuple left, top, right, bottom
    """
    # init to ridiculously big values. not very elegant or eficient
    l, t, r, b = 10000000, 10000000, -10000000, -10000000
##    l = pointArray[0]
##    t = pointArray[1]
##    r = l
##    b = t

    for n in pointArray: # calc bounding rectangle rect
        if n[0] < l : l = n[0]
        if n[0] > r : r = n[0]
        if n[1] < t : t = n[1]
        if n[1] > b : b = n[1]

    return l, t, r, b


def calcRectCenter(l,t,r,b):#,v=()):
    """ returns rect center point -> x,y
        calcRectCenter(l,t,r,b)
    """
##    if len(v) : l,t,r,b = v[0],v[1],v[2],v[3]
    return l+((r-l)*0.5), t+((b-t)*0.5)

def update(wobj, info, ob, ac, nact):

    global win
    global robot
    global x
    global y
    global wobjects
    global nobjects

    if (win == None):
        win = window.Window()
        win.set_size(600+10, 400+10+100)
        glEnable(GL_BLEND)
    
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #p = Pixel(10,10)
        #a = Arc(150,150,radius=100,color=(1.,0.,0.,1.),sweep=90,style=GLU_FILL)
        #P = Polygon([(0, 0), (50, 200), (80, 200),(60,100),(100,5)],color=(.3,0.2,0.5,.7))
        sx = 10
        sy = 10
        x = 100
        y = 100
        #dir = 0.0
        # draw label


        """        
        # draw objects
        wobjects = []
        c = 0
        while (wobj[c] > 0.0):
            # robot
            if (wobj[c] == 1.0):
                o = Circle(wobj[c+1],wobj[c+2],width=wobj[c+3]*2.0,color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0))
                wobjects.append(o)
            # wall
            if (wobj[c] == 2.0):
                o = Line((wobj[c+1],wobj[c+2]),(wobj[c+3],wobj[c+4]),stroke=1, color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0))
                wobjects.append(o)
            # cylinder
            if (wobj[c] == 3.0):
                o = Disk(wobj[c+1],wobj[c+2],width=wobj[c+3]*2.0,color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0))
                wobjects.append(o)
            c = c + 10
        nobjects = []
        """


    win.dispatch_events()
    #clear window in white
    glClearColor(255,255,255,0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # render objects


    c = 0
    nobjects = 0
    while (wobj[c] > 0.0):
        # circular robot
        if (wobj[c] == 1.0):
            o = Circle(wobj[c+1],wobj[c+2],width=wobj[c+3]*2.0,color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0)) 
            o.render()
            o = Line((wobj[c+1],wobj[c+2]),(wobj[c+8],wobj[c+9]),stroke=1, color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0))
            o.render()
        # line
        if (wobj[c] == 2.0):
            o = Line((wobj[c+1],wobj[c+2]),(wobj[c+3],wobj[c+4]),stroke=1, color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0))
            o.render()
        # circle
        if (wobj[c] == 3.0):
            o = Disk(wobj[c+1],wobj[c+2],width=wobj[c+3]*2.0,color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0))
            o.render()
        # polygon
        if (wobj[c] == 4.0):
            o = Polygon([(wobj[c+1],wobj[c+2]), (wobj[c+3],wobj[c+2]), (wobj[c+3],wobj[c+4]), (wobj[c+1],wobj[c+4])], color=(wobj[c+5],wobj[c+6],wobj[c+7],1.0))
            o.render()
        c = c + 10
        nobjects += 1

        
    #rend neurons
    x = 0
    y = 450
    ni = 0
    no = 0
    d = 20
    dx = np.clip(400 / (len(ob) + len(ac)), 1, 20)
    for a in ob:
        dd = a * float(d)
        o = Polygon([(x, y), (x+dx, y), (x+dx, y+dd),(x,y+dd)],color=(1.0,0.0,0.0,1.0))
        o.render()
        x = x + dx
    for a in ac:
        dd = a * float(d)
        o = Polygon([(x, y), (x+dx, y), (x+dx, y+dd),(x,y+dd)],color=(0.0,0.0,1.0,1.0))
        o.render()
        x = x + dx
        
    x = 0
    y = 425
    if (nobjects==0):
        y = 350
    d = 20
    dx = np.clip(400 / len(nact), 1, 20)
    for a in nact:
        dd = a * float(d)
        o = Polygon([(x, y), (x+dx, y), (x+dx, y+dd),(x,y+dd)],color=(0.3,0.3,0.3,1.0))
        o.render()
        x = x + dx
        
    # rend text info
    label = pyglet.text.Label(info,
                          font_name='Arial',
                          font_size=10,
                          color=(0, 0, 0, 255),
                          x=0, y=490)

    # rend activation
    label.draw()
    win.flip()

    """
    while not win.has_exit:
        win.dispatch_events()
        #col = [random.random() for i in range(3)]+[1]
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        robot.setX(x)
        robot.setY(y)
        robot.color = [1,0,0,1]
        for wo in wobjects:
            wo.render()

        #robot.render()
        x = x + 1
        #p.render()
        #a.render()
        #a.rotation+=1
        #P.render()
        #l.render()
        win.flip()
    """

"""
if __name__=="__main__":
    for step in range(100):
        updateEnv()
"""
