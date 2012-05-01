"""
Simple particle rendering without shaders.
"""
from OpenGL.GL import *

def render_particles(position_vbo, N):
    """
    Draw particle vertices.
    """
    position_vbo.bind()
    glVertexPointer(3, GL_FLOAT, position_vbo.data.strides[0], position_vbo)
    #glVertexPointer(4, GL_FLOAT, position_vbo.data.strides[0], position_vbo)
    glEnableClientState(GL_VERTEX_ARRAY)
    glDrawArrays(GL_POINTS, 0, N)
    position_vbo.unbind()
    glDisableClientState(GL_VERTEX_ARRAY)

def render_points(position_vbo, N):
    glEnable(GL_POINT_SMOOTH)
    glPointSize(3)
    glColor3f(0,0,0.8)
    render_particles(position_vbo, N)