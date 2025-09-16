#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 1e9
#define W 1.0/N

int main(){
	
	double sum = 0.0;
	double x, pi;
	double start, end;
	
	start = omp_get_wtime();
	
	#pragma omp parallel for reduction(+:sum)
	for(int i=1; i<=(int)N; i++){
		x = W * (i - 0.5);
		sum += 4.0/(1.0 + x*x);
	}
	
	pi = sum * W;
	
	end = omp_get_wtime();
	
	printf("The value of PI is %lf.\n", pi);
	printf("The time done is %.4lf seconds\n":, end - start);
	
	return 0;
}