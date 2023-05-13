#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define BLOCK_SIZE 8

__global__ void pairwise_acceleration(vector3* pos, double* mass, vector3* accels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i == j) {
                 accels[i*NUMENTITIES+j][0] = accels[i*NUMENTITIES+j][1] = accels[i*NUMENTITIES+j][2] = 0.0;
            //FILL_VECTOR(accels[i*NUMENTITIES+j],0,0,0);
        }
        else {
                //__shared__ vector3 pos_i;
                //__shared__ double mass_j;
                //if (threadIdx.y == 0) {
                //pos_i = dPos[i];
                //mass_j = dMass[j];
                //}
                //__syncthreads();
                //vector3 distance = pos_i - dPos[j];
            vector3 distance;
            for (int k=0;k<3;k++) distance[k]=pos[i][k]-pos[j][k];
            __syncthreads();
            double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
            double magnitude=sqrt(magnitude_sq);
            double gc=GRAV_CONSTANT;
            double accelmag=-1* gc *mass[j]/magnitude_sq;

            accels[i*NUMENTITIES+j][0] = accelmag * distance[0] / magnitude;
                    accels[i*NUMENTITIES+j][1] = accelmag * distance[1] / magnitude;
                    accels[i*NUMENTITIES+j][2] = accelmag * distance[2] / magnitude;
            //FILL_VECTOR(accels[i*NUMENTITIES+j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
    }
}

__global__ void sum_rows(vector3* accels, double* mass, vector3* vel, vector3* pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES) {
        vector3 accel_sum={0,0,0};
        for (int j=0;j<NUMENTITIES;j++){
            for (int k=0;k<3;k++)
                accel_sum[k]+=accels[i*NUMENTITIES+j][k];
        }
        //compute the new velocity based on the acceleration and time interval
        //compute the new position based on the velocity and time interval
        for (int k=0;k<3;k++){
            vel[i][k]+=accel_sum[k]*INTERVAL;
            pos[i][k]+=vel[i][k]*INTERVAL;
        }
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity using CUDA
void compute() {


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((NUMENTITIES+BLOCK_SIZE-1)/BLOCK_SIZE, (NUMENTITIES+BLOCK_SIZE-1)/BLOCK_SIZE);

    pairwise_acceleration<<<dimGrid, dimBlock>>>(dPos, dMass, dAccels);
    sum_rows<<<(NUMENTITIES+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(dAccels, dMass, dVel, dPos);

    cudaMemcpy(hPos, dPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

}
