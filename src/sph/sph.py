import pyopencl as cl

import sys, os, struct
from math import log, ceil

import numpy as np

from .radix_sort import RadixSort

class FluidSimulator(object):
    def __init__(self, N, boxsize=(10,10,10), gl_interop=False):

        # modify N here such that it is useful to set positions (see initialize_positions()).
        # in this example, we make sure to have the right number of particles 
        # to build one large and two smaller cubes (using not more particles than the user requested).
        assert N > 270, "N must be larger than 270."
        a = 0
        while a**3 * 1.25 <= N: 
            a += 2
        N = ((a-2)**3 * 5) // 4
        self.N = N
        self.gl_interop = gl_interop

        # set p2 to be the smallest power of 2 larger or equals than N
        # needed for the current implementation of radix sort, which requires the arrays to be sorted to have
        # lengths of powers of two.
        self.p2 = 2**int(ceil(log(N)/log(2)))

        # gas constant
        self.k = 1000
        # fluid viscosity. if it is too high, the simulation might explode.
        # in that case, also increase k or decrease the timestep dt.
        self.viscosity = 250
        self.boxsize = boxsize

        ## can switch off those two for debugging
        # if False, uses bruteforce neighbour search
        self.use_grid = True
        # if False, positions/velocities won't be reordered for better memory coherency.
        self.reorder = True 

        # put all particles in a cube of length 0.5
        self.spacing0 = 0.5*max(boxsize) / (N**(1/3.))
        self.dt = 0.5 * self.spacing0 / self.k**0.5;
        self.h = 2.000001 * self.spacing0; # h=2*intial spacing
        
        # water rest density
        self.density0 = 1000
        # particle mass
        self.mass = self.spacing0**3 * self.density0

        # for grid based neighbour search
        self.number_of_cells = tuple([int(ceil(boxsize/self.h)) for boxsize in self.boxsize]); # 1h is support of kernel
        self.total_number_of_cells = self.number_of_cells[0] * self.number_of_cells[1] * self.number_of_cells[2]

        # initialize positions on the cpu and later move them to the gpu
        self.position = np.ndarray((N, 4), dtype=np.float32)
        self.velocity = np.zeros((N, 4), dtype=np.float32)

        # work group size for hashing computations
        self.wg_size = 64

        # caclulate local and global sizes for the kernel invocations
        local_size = 128
        grid_size = int(ceil(self.N/float(local_size)))
        global_size = grid_size*local_size
        self.local_size = (local_size, )
        self.global_size = (global_size, )

        assert len(self.local_size) == 1 and len(self.global_size) == 1, "need 1D indices"

        self.initialize_positions()

        print('%i particles' % self.N)
        print('initial density: %s, mass: %s, gas constant k: %s, timestep: %s' % (self.density0, self.mass, self.k, self.dt))
        print('%s %s cells' % (self.number_of_cells, self.total_number_of_cells))

    def initialize_positions(self):
        """
        Initalize the initial positions/velocities of the particles.
        Using self.N particles.
        """
        i = 0
        position = self.position
        spacing0 = self.spacing0

        # arrange particles in one large and two small cubes
        # (N needs to be a cubic number)
        n = int((self.N/1.25)**(1/3.)+0.5)
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    position[i, 0] = (x + 0.5) * spacing0
                    position[i, 1] = (y + 0.5) * spacing0 + 0.01*self.boxsize[1]
                    position[i, 2] = self.boxsize[2] - (z + 0.5) * spacing0
                    position[i, 3] = 0
                    i += 1

        for x in range(n//2):
            for y in range(n//2):
                for z in range(n//2):
                    position[i, 0] = (x + 0.5) * spacing0 + 0.01*self.boxsize[0]
                    position[i, 1] = self.boxsize[1] - (y + 0.5) * spacing0 
                    position[i, 2] = self.boxsize[2] - (z + 0.5) * spacing0
                    position[i, 3] = 0
                    i += 1

        for x in range(n//2):
            for y in range(n//2):
                for z in range(n//2):
                    position[i, 0] = self.boxsize[0] - (x + 0.5) * spacing0
                    position[i, 1] = (y + 0.5) * spacing0 + 0.15*self.boxsize[1]
                    position[i, 2] = (z + 0.5) * spacing0
                    position[i, 3] = 0
                    i += 1

    def cl_pick_device(self):
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                if device.type == cl.device_type.GPU:
                    return device
        raise Exception("No suitable device found")

    def cl_init_context(self):
        """
        Define context and queue.
        """
        device = self.cl_pick_device()
        platform = device.platform
        additional_properties = []
        if self.gl_interop:
            from pyopencl.tools import get_gl_sharing_context_properties
            additional_properties = get_gl_sharing_context_properties()
        # Some OSs prefer clCreateContextFromType, some prefer clCreateContext. Try both.
        try: 
            self.ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + additional_properties)
        except:
            self.ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + additional_properties, devices=[device])
        self.queue = cl.CommandQueue(self.ctx, device=device, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

    def cl_init(self):
        """
        Initialize opencl context, queue and device buffers.
        """
        if self.gl_interop:
            # create a opengl vertex buffer object we can can share the particle positions
            # with opengl (for rendering).
            from OpenGL.arrays import vbo
            from OpenGL import GL
            self.position_vbo = vbo.VBO(data=self.position, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)

        if self.gl_interop:
            # need to bind positions here, not sure why
            self.position_vbo.bind()
            self.cl_init_context()
            self.position_vbo.unbind()
        else:
            self.cl_init_context()
        # load opencl code
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        from mako.template import Template
        with open('%s/sph.cl' % cur_dir) as f:

            code = str(Template(f.read()).render(
                # constant parameters, replaced in code before
                # it is compiled
                use_grid=self.use_grid,
                reorder=self.use_grid and self.reorder,
                local_hash_size=self.wg_size+1,
                h=self.h,
                number_of_cells=self.number_of_cells,
                boxsize=self.boxsize,
                density0=self.density0,
                k=self.k,
                viscosity=self.viscosity,
                ))
            self.prg = cl.Program(self.ctx, code).build()

        self.radix_sort = RadixSort(self.ctx, self.queue, self.p2, np.uint32)

        self.cl_init_data()


    def cl_init_data(self):
        """
        Create and populate opencl device buffers.
        """
        N = self.N

        mf = cl.mem_flags
        ctx = self.ctx

        # sizes of float32, uint32
        sf = np.nbytes[np.float32]
        si = np.nbytes[np.uint32]

        # constant params made available to the kernels
        # can be updated though by copying a new set of parameters to the device 
        params = struct.pack('Iff',
                             np.uint32(N),
                             np.float32(self.dt),
                             np.float32(self.mass))
        self.params_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=params)


        if self.gl_interop:
            # share positions with opengl (for rendering)
            # use shared gl buffer
            self.position_vbo.bind() # added to test on ATI
            try:
                self.position_cl = cl.GLBuffer(ctx, mf.READ_WRITE, int(self.position_vbo.buffers[0]))
            except AttributeError: # pyopengl-accelerate is installed, only single buffer available
                self.position_cl = cl.GLBuffer(ctx, mf.READ_WRITE, int(self.position_vbo.buffer))
            self.position_vbo.unbind() # added to test on ATI
            self.cl_gl_objects = [self.position_cl]
        else:
            # create new device buffer
            self.position_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.position)

        self.position_sorted_cl = cl.Buffer(ctx, mf.READ_WRITE, size=4*N*sf)

        self.density_cl = cl.Buffer(ctx, mf.READ_ONLY, size=N*sf)
        self.pressure_cl = cl.Buffer(ctx, mf.READ_WRITE, size=N*sf)

        self.velocity_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.velocity_sorted_cl = cl.Buffer(ctx, mf.READ_WRITE, size=4*N*sf)
        self.acceleration_cl = cl.Buffer(ctx, mf.READ_WRITE, size=4*N*sf)

        p2 = self.p2

        # grid_hash and grid_index will be sorted, and the sort algorithm can only sort arrays that have lengths that are powers of two.
        # prefill grid_hash_cl with the largest uint32 (=unused) so that the ascending sort will make the unused elements end up at the end.
        grid_hash_preset = np.zeros((p2, 1), dtype=np.uint32)
        grid_hash_preset.fill(-1) # largest uint32
        self.grid_hash_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=grid_hash_preset)
        
        self.grid_index_cl = cl.Buffer(ctx, mf.READ_WRITE, size=p2*si)
        self.cell_start_cl = cl.Buffer(ctx, mf.READ_WRITE, size=self.total_number_of_cells*si)
        self.cell_end_cl = cl.Buffer(ctx, mf.READ_WRITE, size=self.total_number_of_cells*si)
        
        self.queue.finish()


    def step(self):
        """
        Advance simulation.
        """
        prg = self.prg
        queue = self.queue

        if self.gl_interop:
            cl.enqueue_acquire_gl_objects(queue, self.cl_gl_objects)

        ## one simulation step consists of four steps:
        ## step 1) assign particles to cells in the uniform grid
        ## step 2) compute densities
        ## step 3) compute accelerations from forces
        ## step 4) move particles according to accelerations.
        ## for now, simple collision detection is done in the last step.

        # step 1)
        if self.use_grid:
            self.assign_cells()

        global_size = self.global_size
        local_size = self.local_size

        # if we use a grid based neighbour search, we need to pass
        # some additional arguments.
        grid_args = [self.grid_index_cl,
                     self.cell_start_cl,
                     self.cell_end_cl] if self.use_grid else []

        # step 2)
        prg.stepDensity(queue, global_size, local_size, 
                        self.position_sorted_cl if self.use_grid and self.reorder else self.position_cl,
                        self.density_cl,
                        self.pressure_cl,
                        self.params_cl, *grid_args).wait()

        # step 3)
        prg.stepForces(queue, global_size, local_size,
                       self.position_sorted_cl if self.use_grid and self.reorder else self.position_cl,
                       self.velocity_sorted_cl if self.use_grid and self.reorder else self.velocity_cl,
                       self.acceleration_cl,
                       self.density_cl,
                       self.pressure_cl,
                       self.params_cl, *grid_args).wait()

        # step 4)
        prg.stepMove(queue, global_size, local_size,
                     self.position_cl,
                     self.velocity_cl,
                     self.acceleration_cl,
                     self.params_cl).wait()

        if self.gl_interop:
            cl.enqueue_release_gl_objects(queue, self.cl_gl_objects)

    def assign_cells(self):
        """
        Do some work so we can efficiently find neighbours using a grid.
        """
        prg = self.prg
        queue = self.queue

        # compute hashes

        prg.computeHash(queue, (self.N,), None,
                        self.position_cl,
                        self.grid_hash_cl,
                        self.grid_index_cl,
                        self.params_cl).wait()
        
        # sort particles based on hash, ascending.
        self.radix_sort.sort(self.grid_hash_cl, self.grid_index_cl, self.p2)

        
        # find cell start / cell end
        global_size = (self.p2 - self.p2 % self.wg_size, )
        local_size = (self.wg_size, )

        prg.memset(queue, global_size, local_size,
                   self.cell_start_cl,
                   np.uint32(-1),
                   np.uint32(self.total_number_of_cells)).wait()
        
        prg.reorderDataAndFindCellStart(queue, global_size, local_size,
                                        self.cell_start_cl,
                                        self.cell_end_cl,
                                        self.grid_hash_cl,
                                        self.grid_index_cl,
                                        self.position_cl,
                                        self.position_sorted_cl,
                                        self.velocity_cl,
                                        self.velocity_sorted_cl,
                                        self.params_cl).wait()

    def get_position(self):
        """
        Copies the position buffer from device to host.
        """
        assert not self.gl_interop, "currently not working with gl interop"
        cl.enqueue_copy(self.queue, self.position, self.position_cl)
        return self.position

    def set_positions(self):
        """
        Copies the positions (and velocities) from host to device
        """
        self.position_vbo.set_array(self.position)
        cl.enqueue_copy(self.queue, self.velocity_cl, self.velocity).wait()
        
if __name__ == '__main__':
    N = 10**3
    fluid_simulator = FluidSimulator(N)
    fluid_simulator.cl_init()
    track_particle = 123
    position = fluid_simulator.get_position()
    print('position of particle %i before simulation: %s' % (track_particle, position[track_particle,0:3]))
    
    # do 100 simulation steps
    for i in range(100):
        fluid_simulator.step()
    
    position = fluid_simulator.get_position()
    print('position of particle %i after simulation: %s' % (track_particle, position[track_particle,0:3]))
