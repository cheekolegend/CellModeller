#define ISQRT2 ((float)(1.f/sqrt(2.f)))

float userAdhLogic(const float adh_str1, const float adh_str2)
{
    %s 
    //return min(adh_str1, adh_str2);
}

// This one is for 2020 version of CellModeller
__kernel void build_tangent_matrix(const int max_contacts,
                           __global const float4* centers,
                           __global const float4* dirs,
                           __global const float* lens,
                           __global const float* rads,
                           __global const float* adh_str, //******
                           __global const int* n_cts,
                           __global const int* frs,
                           __global const int* tos,
                           __global const float4* pts,
                           __global const float4* norms,
                           __global float8* ct_tangs_fr, //*****
                           __global float8* ct_tangs_to, //*****
                           __global float* ct_adh_str, //*******
                           __global float* overlap, //*******
                           __global float* stiff)
{
  int id = get_global_id(0);
  int ct = get_global_id(1);

  if (ct >= n_cts[id]) return;

  int i = id*max_contacts + ct;
  
  int a = frs[i];
  int b = tos[i];
  float4 r_a = pts[i]-centers[a];
  float4 r_b = pts[i]-centers[b];
  
  // k*dx
  if (b > 0) { // if a cell contact
    ct_adh_str[i] = overlap[i]*userAdhLogic(adh_str[a], adh_str[b]); // overlap[i] = 0 for walls! :(
  } else { // if a plane contact
    // For now, take the adhesion coefficient of the cell
    // For some reason overlap with the walls is negative, so take absolute value
    ct_adh_str[i] = fabs(overlap[i])*userAdhLogic(adh_str[a], adh_str[a]);
  }
  
  ///////////////////////// tangents to contacts

  float8 ct_tang_fr = 0.f;
  float4 dxr_a = 0.f;

  float4 d1 = 0.f;
  float4 z_axis = {0.f, 0.f, 1.f, 0.f};
  
  // Calculate tangent vector to contact - CHECKED!
  d1 = cross(norms[i], z_axis); //this is general for the 2d case, but not otherwise.
  d1 = normalize(d1);
  
  // Formulating B term (Eq. 3 in Anton Kan's 2018 paper SI)
  dxr_a = cross(d1, r_a);

  ct_tang_fr.s012 = d1.s012; //direction of displacement we are interested in
  ct_tang_fr.s345 = -dxr_a.s012; //h_hat cross point_r_a
  ct_tang_fr.s6 = dot(dirs[a], r_a) * dot(dirs[a], d1)/(lens[a]+2.f*rads[a]); //last term
  ct_tangs_fr[i] = ct_tang_fr * stiff[i];
  
  // plane and sphere contacts have no to_ent, and have negative indices
  if (b < 0) {
    ct_tangs_to[i] = 0.f;
    return;
  }
  float4 dxr_b = 0.f;
  float8 ct_tang_to = 0.f;
    
  dxr_b = cross(d1, r_b);
  
  ct_tang_to.s012 = d1.s012;
  ct_tang_to.s345 = -dxr_b.s012; 
  ct_tang_to.s6 = dot(dirs[b], r_b) * dot(dirs[b], d1)/(lens[b]+2.f*rads[b]);
  
  ct_tangs_to[i] = ct_tang_to * stiff[i];
}

__kernel void calculate_adhE(const int max_contacts,
                             __global const int* n_cts,
                             __global const int* n_cell_tos,
                             __global const int* cell_tos,
                             __global const float8* ct_tangs_fr,
                             __global const float8* ct_tangs_to,
                             __global const float* dist,
                             __global const float* ct_adh_str,
                             __global float8* adhE)
{
    int i = get_global_id(0);
    int base = i*max_contacts;
    float8 res = 0.f;
    for (int k = base; k < base+n_cts[i]; k++) {
        float8 oldres = res;
        res += ct_tangs_fr[k]*dist[k]*ct_adh_str[k]; 
    }
    
    for (int k = base; k < base+n_cell_tos[i]; k++) {
        int n = cell_tos[k];
        res -= ct_tangs_to[n]*dist[n]*ct_adh_str[n];
    }
    
    adhE[i] = res;
}

// Saved overlap distance with the wall in the "overlap" variable,
// otherwise same as find_plane_contacts
__kernel void find_plane_contacts_adh(const int max_cells,
                                  const int max_contacts,
                                  const int n_planes,
                                  __global const float4* plane_pts,
                                  __global const float4* plane_norms,
                                  __global const float* plane_coeffs,
                                  __global const float4* centers,
                                  __global const float4* dirs,
                                  __global const float* lens,
                                  __global const float* rads,
                                  __global int* n_cts,
                                  __global int* frs,
                                  __global int* tos,
                                  __global float* dists,
                                  __global float4* pts,
                                  __global float4* norms,
                                  __global float* reldists,
                                  __global float* stiff,
                                  __global float* overlap)
{
  int i = get_global_id(0);

  // collision count
  int k = n_cts[i]; //keep existing contacts

  float4 end1 = centers[i] - 0.5f*lens[i]*dirs[i]; // 'left' end of the cell
  float4 end2 = centers[i] + 0.5f*lens[i]*dirs[i]; // 'right' end of the cell

  for (int n = 0; n < n_planes; n++) { // loop through all planes
    int to1 = -2*n - 1; // 'to' if left end has contact with plane n
    int to2 = to1 - 1;  // 'to' if right end has contact with plane n

    float dist1 = pt_to_plane_dist(plane_pts[n], plane_norms[n], end1)-rads[i];
    float dist2 = pt_to_plane_dist(plane_pts[n], plane_norms[n], end2)-rads[i];

    // check for old contacts with this plane
    int cti1 = -1;
    int cti2 = -1;
    for (int m = i*max_contacts; m < i*max_contacts+n_cts[i]; m++) {
      if (tos[m] == to1) cti1 = m;
      else if (tos[m] == to2) cti2 = m;
    }

    // common to both ends
    float4 norm = -plane_norms[n];

    bool two_pts = ((cti1 >= 0) || (dist1<0.f) ) && ( (cti2 >= 0) || (dist2<0.f) );
    float stiffness = two_pts*ISQRT2 + (!two_pts)*1.0;

    // if we're in contact, or we were in contact, recompute
    if ((cti1 >= 0) || (dist1<0.f) ){
      // need to make a new contact
      if (cti1 < 0) {
        cti1 = i*max_contacts+k;
        k++;
      }

      frs[cti1] = i;
      tos[cti1] = to1;
      dists[cti1] = dist1;
      pts[cti1] = end1; // FIXME: not really the right point
      norms[cti1] = norm;
      reldists[cti1] = stiffness*plane_coeffs[n]*dist1;
      stiff[cti1] = stiffness*plane_coeffs[n];
      overlap[cti1] = dist1; //cti1 = 0 for a plane contact -AY
    }

    if ( (cti2 >= 0) || (dist2<0.f) ){
      if (cti2 < 0) {
        cti2 = i*max_contacts+k;
        k++;
      }

      frs[cti2] = i;
      tos[cti2] = to2;
      dists[cti2] = dist2;
      pts[cti2] = end2;
      norms[cti2] = norm;
      reldists[cti2] = stiffness*plane_coeffs[n]*dist2;
      stiff[cti2] = stiffness*plane_coeffs[n];
      overlap[cti2] = dist2; //-AY
    }
  }
  n_cts[i] = k;
}
