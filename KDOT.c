#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

#include "kdot_kernels.c"

BARRIER_INIT(my_barrier, NR_TASKLETS); //syncronization

//since block's dimentions might round up due to ceil we have
//to allocate based on that assumption
__mram T d_m1[(M1_ROWS/NR_DPUS1+1)*M1_COLS];
__mram T d_m2[(M2_ROWS/NR_DPUS2+2)*(M1_COLS)];
__mram T d_res[(M1_ROWS/NR_DPUS1+1)*(M2_ROWS/NR_DPUS2+2)];

__mram_ptr T* aux;

int main()
{
	barrier_t *barr_addr = &my_barrier;
	barrier_wait(barr_addr);
	//run kdot (matrix multiplication) in DPUs assuming second matrix
	//is column major
	aux = kDot_m1_m2T(NR_TASKLETS, d_m1, d_m2, d_res, Ceil((float)M1_ROWS/NR_DPUS1), M1_COLS, Ceil((((float)M2_ROWS/NR_DPUS2)/2))*2, barr_addr);

	//print stuff for debugging purposes
	//printf("\n\nceil val is: %d\n", Ceil((((T)M2_ROWS/NR_DPUS)/2))*2);
	/*for(int i = 0; i < (Ceil((float)M1_ROWS/NR_DPUS)); i++)
	{
		for(int j = 0; j < (Ceil((((T)M2_ROWS/NR_DPUS)/2))*2); j++)
		{
			printf("%.1f ", d_res[i*(Ceil((((T)M2_ROWS/NR_DPUS)/2))*2)+j]);
		}
		printf("\n");
	}*/
}
