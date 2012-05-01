#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct Params {
  uint N;
  float dt;
  float mass;
  int numberOfCells;
} Params;

__constant int3 NC = (int3)(${'%i,%i,%i' % number_of_cells});
__constant float h = ${h};
__constant float h2 = ${h*h};
__constant float h3 = ${h**3};
__constant float h5 = ${h**5};

<%def name="loopneighbours()">
  % if not use_grid:
  // brutefoce, go through all particles
  for(uint j = 0; j < params->N; ++j) {
    ${caller.body()}
  }
  % else :
  // loop through neighbors efficiently using the grid
  const int3 gridPos = getGridPosition(x);
  const int zi_min = max(0, gridPos.z-1), zi_max = min(NC.z-1, gridPos.z+1);
  const int yi_min = max(0, gridPos.y-1), yi_max = min(NC.y-1, gridPos.y+1);
  const int xi_min = max(0, gridPos.x-1), xi_max = min(NC.x-1, gridPos.x+1);
  for(int zi = zi_min; zi <= zi_max; zi++) {
    for(int yi = yi_min; yi <= yi_max; yi++) {
      for(int xi = xi_min; xi <= xi_max; xi++) {
	const uint hash = getGridHash((int3)(xi,yi,zi));
	const uint startIndex = cellStart[hash];
	if (startIndex == (uint)-1) continue;
	const uint endIndex = cellEnd[hash];
	for(int index = startIndex; index < endIndex; ++index) {
	  % if reorder:
	  const int j = index;
	  % else:
	  const int j = gridIndex[index];
	  % endif

	  ${caller.body()}
	}
      }
    }
  }
% endif
</%def>


inline float vlen(const float3 x) {
  // length(x) is *much much* slower for me, must be a bug.
  //return length(x);
  return sqrt(dot(x,x));
}

inline float kernelM4(const float x) {
  const float q = x / h;
  if(q >= 1.f)
    return 0.f;
  
  const float factor = ${2.546479089470325472 / h**3};
  if(q < .5f)
    return factor * (1.f - 6.f * q * q * (1 - q));
  float a = 1.f - q;
  return factor * 2.f * a*a*a;
}

inline float kernelM4_d(const float x) {
  const float q = x / h;
  
  if(q >= 1.f)
    return 0.f;
  
  const float factor = ${2.546479089470325472 / h**5};
  if(q < .5f)
    return factor * (-12.f + 18.f * q);
  
  return factor * (-6.f * (1.f-q)*(1.f-q)/q);
}

// laplacian of Wviscosity as in Stefan Auer's thesis
inline float kernelVisc_dd(const float x) {
  const float q = x / h;
  if(q <= 1.f)
    return -45.f / (M_PI * h5)*(1-q);
  
  return 0.f;
}


% if use_grid:
// only need those functions if we are using a grid based search

inline int3 getGridPosition(float3 position) {
  position.x *= ${number_of_cells[0]/float(boxsize[0])}f;
  position.y *= ${number_of_cells[1]/float(boxsize[1])}f;
  position.z *= ${number_of_cells[2]/float(boxsize[2])}f;
  return (int3)(min((int)position.x, NC.x-1), min((int)position.y, NC.y-1), min((int)position.z, NC.z-1));
}

inline uint getGridHash(int3 gridPos) {
  return (gridPos.z * NC.z + gridPos.y) * NC.y + gridPos.x;
}

__kernel void computeHash(__global float4 *position,
			  __global uint *gridHash,
			  __global uint *gridIndex,
			  __constant Params *params
			  ) {
  uint i = get_global_id(0);

  float3 p = as_float3(position[i]);
  
  int3 gridPos = getGridPosition(p);
  gridHash[i] = getGridHash(gridPos);;
  gridIndex[i] = i;
}

__kernel void memset(__global uint *d_Data, uint val, uint N) {
  uint i = get_global_id(0);
  if(i < N)
    d_Data[i] = val;
}

__kernel void reorderDataAndFindCellStart(__global uint *cellStart,
					  __global uint *cellEnd,
					  __global uint *gridHash,
					  __global uint *gridIndex,
					  __global float4 *position,
					  __global float4 *positionSorted,
					  __global float4 *velocity,
					  __global float4 *velocitySorted,
					  __constant Params *params
					  ) {

  __local uint localHash[${local_hash_size}]; // get_group_size(0) + 1 elements

  uint hash;
  uint N = params->N;
  const uint index = get_global_id(0);
  
  // handle case when no. of particles not multiple of block size
  if(index < N) {
    hash = gridHash[index];
    
    // Load hash data into local memory so that we can look 
    // at neighboring particle's hash value without loading
    // two hash values per thread
    localHash[get_local_id(0) + 1] = hash;
    //First thread in block must load neighbor particle hash
    if(index > 0 && get_local_id(0) == 0)
      localHash[0] = gridHash[index - 1];

    barrier(CLK_LOCAL_MEM_FENCE);

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell.
    // As it isn't the first particle, it must also be the cell end of
    // the previous particle's cell

    if(index == 0 || hash != localHash[get_local_id(0)]) {
      cellStart[hash] = index;
      if(index > 0)
	cellEnd[localHash[get_local_id(0)]] = index;
    }

    if(index == N-1) {
      cellEnd[hash] = N;
    }
    
% if reorder:
    uint sortedIndex = gridIndex[index];
    positionSorted[index] = position[sortedIndex];
    velocitySorted[index] = velocity[sortedIndex];
% endif
  }
}

% endif // use_grid

__kernel void stepDensity(__global float4 *position
			  , __global float *density
			  , __global float *pressure
			  , __constant Params *params
% if use_grid:
			  , __global uint *gridIndex
			  , __global uint *cellStart
			  , __global uint *cellEnd
% endif
			  ) {
  uint i = get_global_id(0);
  if(i >= params->N) return;

  float3 x = as_float3(position[i]);
  float _density = 0.f;

  <%self:loopneighbours>
    float3 x_j = as_float3(position[j]);
    _density += kernelM4(vlen(x-x_j));
  </%self:loopneighbours>

  _density *= params->mass;
  
  density[i] = _density;
  
  pressure[i] = max(0.f, ${k} * (_density - ${density0}));
}

__kernel void stepForces(__global float4 *position
			 , __global float4 *velocity
			 , __global float4 *acceleration
			 , __global float *density
			 , __global float *pressure
			 , __constant Params *params
% if use_grid:
			 , __global uint *gridIndex
			 , __global uint *cellStart
			 , __global uint *cellEnd
% endif
			 ) {
  uint i = get_global_id(0);
  if(i >= params->N) return;
  float3 x = as_float3(position[i]);
  float3 v = as_float3(velocity[i]);
  float _density = density[i];
  float _pressure = pressure[i];
  float3 accel = (float3)(0.f, 0.f, 0.f);
  
  <%self:loopneighbours>
    float3 x_j = as_float3(position[j]);
    float3 x_diff = x - x_j;
    float len = vlen(x_diff);
    if(len != 0.f) {
      float3 v_j = as_float3(velocity[j]);
      float _density_j = density[j];
      float _pressure_j = pressure[j];
      
      // pressure force
      accel -= x_diff/len * (.5f * (_pressure + _pressure_j) / _density_j * kernelM4_d(len));
      
      // viscosity
      const float v_coeff = ${viscosity}.f;

      // laplacian of Wviscosity as in Stefan Auer's thesis.
      accel += (v - v_j) * (v_coeff / _density_j * kernelVisc_dd(len));
    }	
  </%self:loopneighbours>
      
  accel *= params->mass / _density;
  
  // gravity
  accel.y -= 9.81f;

  % if reorder:
  acceleration[gridIndex[i]] = as_float4(accel);
  % else:
  acceleration[i] = as_float4(accel);
  % endif
}

__kernel void stepMove(__global float4 *position, 
		       __global float4 *velocity, 
		       __global float4 *acceleration, 
		       __constant Params *params
		       ) {
  uint i = get_global_id(0);
  if(i >= params->N) return;
  float3 x = as_float3(position[i]);
  float3 v = as_float3(velocity[i]);
  float3 a = as_float3(acceleration[i]);
  
  v += a * params->dt;
  x += v * params->dt;

  const float damp = 0.4f;

  // collisions
  if(x.x < 0.f) {
    x.x = 0.f;
    v.x *= -damp;
  }
  if(x.y < 0.f) {
    x.y = 0.f;
    v.y *= -damp;
  }		
  if(x.z < 0.f) {
    x.z = 0.f;
    v.z *= -damp;
  }

  const int3 boxsize = (int3)(${'%i,%i,%i' % boxsize});
  if(x.x > boxsize.x) {
    x.x = boxsize.x;
    v.x *= -damp;
  }
  if(x.y > boxsize.y) {
    x.y = boxsize.y;
    v.y *= -damp;
  }
  if(x.z > boxsize.z) {
    x.z = boxsize.z;
    v.z *= -damp;
  }

  position[i] = as_float4(x);
  velocity[i] = as_float4(v);
}
