
__constant float width = ${window_size[0]};
__constant float height = ${window_size[1]};

__constant float w = ${projection_matrix[0,0]};
__constant float h = ${projection_matrix[1,1]};
__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


__constant int2 right = (int2)(1,0);
__constant int2 down = (int2)(0,1);


#define DEPTH(coords) read_imagef(depth_in, smp, (coords)).x

inline float2 coordsToUv(int2 coords) {
  return (float2)(coords.x/(width-1), coords.y/(height-1));
}

inline float3 uvToEye(int2 coords, float z) {
  float2 uv = coordsToUv(coords);
  uv = uv * 2 - 1;
  
  /* assuming this is the projection matrix (right handed):
     m = [w, 0, 0, 0;
     0, h, 0, 0;
     0, 0, f/(n-f), n*f/(n-f);
     0, 0, -1, 0]
  */
 
  return ((float3)(-uv.x/w, -uv.y/h, 1)) * z;
}

float3 to3D(int2 coords, __read_only image2d_t depth_in) {
  float z = DEPTH(coords);
  return uvToEye(coords, z);
}

inline float diffZ(int2 coords, int2 offset, __read_only image2d_t depth_in) {
  // derivative of z(x,y), where x and y are screen space coordinates.

  const float h = 1.; // finite difference with 1 pixel difference
  float dp = DEPTH(coords+offset);
  float dm = DEPTH(coords-offset);
  if(dp == 0 || dm == 0) return 0;
  return (dp - dm)/(2*h);
}

inline float diffZ_2(int2 coords, int2 offset, __read_only image2d_t depth_in) {
  // second order central difference of z(x,y) (screen space coordinates).

  const float h = 1.; // finite difference with 1 pixel difference
  float dp = DEPTH(coords+offset);
  float d = DEPTH(coords);
  float dm = DEPTH(coords-offset);

  return (dp-2*d+dm)/(h*h);
}

inline float diffZ_xy(int2 coords, __read_only image2d_t depth_in) {
  int ox = 1, oy = 1;
  const float h = 1.; // finite difference with 1 pixel difference
  
  return (DEPTH(coords+right+down) - DEPTH(coords+right-down) - DEPTH(coords-right+down) + DEPTH(coords-right-down))/(4*h*h);
}


/* inline float3 computeNormal(int2 coords, __read_only image2d_t depth_in) { */
/*   float z = DEPTH(uv); */
/*   float z_x = diffZ(uv, (float2)(texelSize.x, 0), depth_in); */
/*   float z_y = diffZ(uv, (float2)(0, texelSize.y), depth_in); */
/*   float2 uv2 = 2*uv-1; */

/*   float Cx = -2/(width*w); */
/*   float Cy = -2/(height*h); */
/*   // sx, sy = screen space coordinates */
/*   float sx = floor(uv.x*(width-1)), sy = floor(uv.y*(height-1)); */
/*   float Wx=(width-2*sx)/(width*w); */
/*   float Wy=(height-2*sy)/(height*h); */

/*   // diff uvToEye(uv) w.r.t. screen space x */
/*   float3 _dx = (float3)(Cx*z+Wx*z_x*0, */
/* 			Wy*z_x, */
/* 			z_x); */

/*   // diff uvToEye(uv) w.r.t. screen space y */
/*   float3 _dy = (float3)(Wx*z_y, */
/* 			Cy*z+Wy*z_y, */
/* 			z_y); */

/*   float3 normal = normalize(cross(_dx,_dy)); */
/*   return normal; */

/* } */

inline float divN(int2 coords, __read_only image2d_t depth_in) {
  float z = DEPTH(coords);
  float z_x = diffZ(coords, right, depth_in);
  float z_y = diffZ(coords, down, depth_in);

  float Cx = -2/(width*w);
  float Cy = -2/(height*h);
  // sx, sy = screen space coordinates
  float sx = coords.x, sy = coords.y;
  float Wx=(width-2*sx)/(width*w);
  float Wy=(height-2*sy)/(height*h);


  float D = Cy*Cy*z_x*z_x + Cx*Cx*z_y*z_y+Cx*Cx*Cy*Cy*z*z;
  float z_xx = diffZ_2(coords, right, depth_in);
  float z_yy = diffZ_2(coords, down, depth_in);
  float z_xy = diffZ_xy(coords, depth_in);
  float D_x = 2*Cy*Cy*z_x*z_xx + 2*Cx*Cx*z_y*z_xy + 2*Cx*Cx*Cy*Cy*z*z_x;
  float D_y = 2*Cy*Cy*z_x*z_xy + 2*Cx*Cx*z_y*z_yy + 2*Cx*Cx*Cy*Cy*z*z_y;
  float Ex = 0.5*z_x*D_x-z_xx*D;
  float Ey = 0.5*z_y*D_y-z_yy*D;
  float H = (Cy*Ex+Cx*Ey)/(2*D*sqrt(D));
  return H;
}

__kernel void test(__read_only image2d_t depth_in, __write_only image2d_t test)  {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_global_size(0);
  int h = get_global_size(1);
  int2 coords = (int2)(x,y);
  float2 uv = (float2)((float)x/(w-1),(float)y/(h-1));

  float depth = DEPTH(coords);
  
  if(depth == 0) {
    write_imagef(test, coords, (float4)(0,0,0,1));
    return;
  }

  //write_imagef(test, coords, (float4)(1,0,0,1));
  /* write_imagef(test, (int2)(x,y), (float4)(DEPTH(uv),0,0,1)); */
  /* return; */

  write_imagef(test, (int2)(x,y), (float4)(divN(coords, depth_in),0,0,1));

  //write_imagef(test, (int2)(x,y), (float4)(read_imagef(depth_in, smpn, (uv)).y,0,0,1));
  
  /* float3 normal = computeNormal(uv, depth_in); */
  /* write_imagef(test, (int2)(x,y), (float4)(normal.x,normal.y,normal.z,1)); */
}

__kernel void mean_curvature(__read_only image2d_t depth_in, __write_only image2d_t out)  {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int2 coords = (int2)(x,y);

  float depth = DEPTH(coords);
  
  if(depth == 0) {
    write_imagef(out, coords, (float4)(0,0,0,1));
    return;
  }

  write_imagef(out, (int2)(x,y), (float4)(divN(coords, depth_in),0,0,1));
}

__kernel void curvatureFlow(__read_only image2d_t depth_in, __write_only image2d_t depth_out, const float dt, const float varying_z_contrib)  {
  
  int x = get_global_id(0);
  int y = get_global_id(1);
  int2 coords = (int2)(x,y);
  
  float depth = DEPTH(coords);

  if(depth == 0) {
    write_imagef(depth_out, coords, (float4)(depth, 0, 0, 1));
    return;
  }

  float z_x = diffZ(coords, right, depth_in);
  float z_y = diffZ(coords, down, depth_in);

  /* float diff_left	= fabs(DEPTH(coords-right) - depth); */
  /* float diff_right	= fabs(DEPTH(coords+right) - depth); */
  /* float diff_top	= fabs(DEPTH(coords-down) - depth); */
  /* float diff_bottom	= fabs(DEPTH(coords+down) - depth); */
  /* float depth_threshold = 1; */
  /* if(diff_left > depth_threshold || */
  /*    diff_right > depth_threshold || */
  /*    diff_top > depth_threshold || */
  /*    diff_bottom > depth_threshold) { */
  /*   write_imagef(depth_out, coords, (float4)(depth, 0, 0, 1)); */
  /*   return; */
  /* } */


  /* float z_xt = diffZ(coords, 3*right, depth_in); */
  /* float z_yt = diffZ(coords, 3*down, depth_in); */
  /* if(z_xt==0||z_yt==0||z_x==0||z_y==0) { */
  /*   write_imagef(depth_out, coords, (float4)(depth, 0, 0, 1)); */
  /*   return; */
  /* } */
  
  float mean_curv = divN(coords, depth_in); 
  // experimental hack:
  // I noticed that where the depth varies a lot (oblique view of the fluid),
  // the smoothing is not as good as where it does not vary as much and that at those places,
  // the smoothing is stable with a larger timestep.
  //const float varying_z_contrib = 40;

  depth += mean_curv * dt * (1 + (fabs(z_y)+fabs(z_x))*varying_z_contrib);


  write_imagef(depth_out, coords, (float4)(depth, 0, 0, 1));
  
}

