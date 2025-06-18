#include <stdio.h>
#include <perfcounter.h>
#include <mram.h>
#include <defs.h>
#include <barrier.h>
#include "common/common.h"

//#include "math.h"

//macro for exponential calc since there is no math.h in Upmem
static union
{
    double d;
    struct {
      int j,i;
    } n;
 } _eco;

#define EXP_A 1512775
#define EXP_C 68243 /*see text for choice of values */
#define exponential(y) (_eco.n.i=EXP_A*(y)+(1072693248-EXP_C), _eco.d)


//ceil function because there is no math.h in Upmem
int Ceil(float a)
{
    return(a > (int) a) ? (a+1) : (a);
}

//min function because there is no math.h in Upmem
int min(int a, int b)
{
	return(a > b) ? b : a;
}



__mram_ptr T* kDot(const int n_threads, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{	
	int max_threads = m1_rows * m2_columns / 2;
	T* t_output = (T*) mem_alloc(max_threads * sizeof(T));
	int n_elem = m1_rows * m2_columns / n_threads;
	
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
 
	//print for debugging purposes
    /*if(me() == 0)
  	{	
  		for(int i = 0; i < m1_rows; i++)
		{
			for(int j = 0; j < m1_columns; j++)
			{
				printf("%.1f ", m1[i*m1_columns+j]);
			}
			printf("\n");
		}
	}*/
  
 
 	//sinc all threads
	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			//printf("m2 cols: %d\n",m2_columns);
		    int r = (int)i / m2_columns;
		    int c = i % m2_columns;
		    t_output[me()] = 0;
		    for( int k = 0; k < m1_columns; ++k )
		    {
	        	t_output[me()] = t_output[me()] + (m1[r * m1_columns + k] * m2[k * m2_columns + c]);
		       	
	        	//print for debugging purposes
		       	//printf("m1 = %.1f m2 = %.1f t_output = %.1f\n", m1[r * m1_columns + k], m2[k * m2_columns + c],t_output[me()]);
	        	//printf("%1f\n", m1[r * m1_columns + k] * m2[k * m2_columns + c]);
		        //printf(" m2 I'm thread %d and Value in column %d is %.10f\n",me(),k, m2[k * m2_columns + c]);
		        //printf("row is: %d\n", c);

		    }
		    output[i] = t_output[me()];
		    //printf("Stat is: %u. Value in %d is %.10f\n",check_stack(),i, output[i]);
		}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int j = 0; j < m1_rows; j++)
		{
			for(int k = 0; k < m2_columns; k++)
			{
				printf("%.4f ", output[j*m2_columns+k]);
			}
			printf("\n");
		}
	}
	*/
		
	return output;
}

__mram_ptr T* kDot_inference(const int n_threads, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, const int m1_rows, const int m1_columns, const int m2_columns, struct barrier_t * barrier_address)
{
	//KDot_inference difference to KDot is that it allows a 
	//single entry in the input matrix when doing inference
	//the only difference is: if(max_threads == 0)max_threads=1; 
	int max_threads = m1_rows * m2_columns / 2;
	T t_output[max_threads];
	int n_elem = m1_rows * m2_columns / n_threads;
	
	if(max_threads == 0)
		max_threads = 1;
	
	if(n_threads > max_threads)
	{
		n_elem = m1_rows * m2_columns / max_threads;
	}
 
	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			//printf("m2 cols: %d\n",m2_columns);
		    int r = (int)(i / m2_columns);
		    int c = i % m2_columns;
		    t_output[me()] = 0;

		    for( int k = 0; k < m1_columns; ++k )
		    {
	        	t_output[me()] += m1[r * m1_columns + k] * m2[k * m2_columns + c];
		       
	        	//print for debugging purposes
		        //printf(" m1 I'm thread %d and Value in column %d is %.10f\n",me(),k, m1[r * m1_columns + k]);
		        //printf("cachea I'm thread %d and Value in column %d is %.10f\n",me(),k, cachea[me()][0]);

		    }
		    output[i] = t_output[me()];
		    //printf("Stat is: %u. Value in column %d is %.10f\n",check_stack(),i, output[i]);
		}
	}
	barrier_wait(barrier_address);
	/*if(me() == 0)
	{
		for(int j = 0; j < m1_rows*m2_columns; j++)
		{
			printf("Value in column %d is %.10f\n",j, output[j]);
			//printf("%d\n",m1_columns);
		}
	}
	*/
		
	return output;
}

__mram_ptr T* kDot_m1_m2T(const int n_threads, __mram_ptr T const *m1, __mram_ptr T const *m2, __mram_ptr T *output, const int m1_rows, const int m1_columns, const int m2_rows, struct barrier_t * barrier_address)
{

	//This code implements matrix multiplication where the second matrix is transposed.
	int max_threads = m1_rows*m2_rows / 2;
	T t_output[max_threads];
	int n_elem = m1_rows*m2_rows / n_threads;
	if(n_threads > max_threads)
	{
		n_elem = m1_rows*m2_rows / max_threads;
	}


  	if(me() < max_threads)
	{
		for(int i = me()*n_elem; i < (me()+1)*n_elem; i+=1)
		{
			//printf("iter nr: %d\n",i);
		    int r = (int)i / m2_rows;
		    int c = i % m2_rows;
		    t_output[me()] = 0;
		    int id_T;

		    for( int k = 0; k < m1_columns; ++k )
		    {	
		    	id_T = c * m1_columns + k;
		        t_output[me()] += m1[r * m1_columns + k] * m2[id_T];
		        //printf("Value in column %d is %f\n",k, t_output[me()]);
		    }

		    output[i] = t_output[me()];
		    //printf("Value %d: %.1f\n", i, output[i]);
		}
	}
	barrier_wait(barrier_address);
	return output;
}













