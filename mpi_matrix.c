#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mkl.h"
#include "mkl_cblas.h"

#define SINGLE_PRECISION //Comment out to use double precision arithmetic
//#define DOUBLE_PRECISION

#ifdef SINGLE_PRECISION
	#define elem_t float
	#define blasGemm cblas_sgemm 
	#define MPI_ELEM_T MPI_FLOAT
#elif defined(DOUBLE_PRECISION)
	#define elem_t double
	#define blasGemm cblas_dgemm 
	#define MPI_ELEM_T MPI_DOUBLE
#endif

// C versions of utility functions (replacing the C++ template versions)
void allocateMatrixCPU_float(int M, int N, float** ptr) {
	*ptr = (float*)malloc(M * N * sizeof(float));
}

void allocateMatrixCPU_double(int M, int N, double** ptr) {
	*ptr = (double*)malloc(M * N * sizeof(double));
}

void freeMatrixCPU_float(int M, int N, float* ptr) {
	free(ptr);
}

void freeMatrixCPU_double(int M, int N, double* ptr) {
	free(ptr);
}

void initMatrixCPU_float(int M, int N, float* ptr, float val) {
	for (int i = 0; i < M * N; i++)
		ptr[i] = val;
}

void initMatrixCPU_double(int M, int N, double* ptr, double val) {
	for (int i = 0; i < M * N; i++)
		ptr[i] = val;
}

void initMatrixRandomCPU_float(int M, int N, float* ptr) {
	srand((unsigned int)time(NULL));
	for (int i = 0; i < M * N; i++)
		ptr[i] = (float)rand() / RAND_MAX;
}

void initMatrixRandomCPU_double(int M, int N, double* ptr) {
	srand((unsigned int)time(NULL));
	for (int i = 0; i < M * N; i++)
		ptr[i] = (double)rand() / RAND_MAX;
}

// FIXED: Function to build/extract B block from local B matrix
// Each process has local_K rows of B (the full width N)
// Each process extracts specific columns based on block_rank
void buildBBlock_float(int local_K, int N, float* local_B, int B_block_cols, int block_rank, float* B_block) {
	int start_col = block_rank * B_block_cols;
	
	// Extract columns [start_col : start_col + B_block_cols) from local_B
	// local_B has dimensions local_K × N (stored column-major)
	// B_block will have dimensions local_K × B_block_cols
	
	for (int col = 0; col < B_block_cols; col++) {
		for (int row = 0; row < local_K; row++) {
			// Source: local_B at (row, start_col + col)
			// In column-major: local_B[row + (start_col + col) * local_K]
			// Destination: B_block at (row, col)  
			// In column-major: B_block[row + col * local_K]
			B_block[row + col * local_K] = local_B[row + (start_col + col) * local_K];
		}
	}
}

void buildBBlock_double(int local_K, int N, double* local_B, int B_block_cols, int block_rank, double* B_block) {
	int start_col = block_rank * B_block_cols;
	
	for (int col = 0; col < B_block_cols; col++) {
		for (int row = 0; row < local_K; row++) {
			B_block[row + col * local_K] = local_B[row + (start_col + col) * local_K];
		}
	}
}

// Macro definitions to replace template calls
#ifdef SINGLE_PRECISION
	#define allocateMatrixCPU(M, N, ptr) allocateMatrixCPU_float(M, N, ptr)
	#define freeMatrixCPU(M, N, ptr) freeMatrixCPU_float(M, N, ptr)
	#define initMatrixCPU(M, N, ptr, val) initMatrixCPU_float(M, N, ptr, val)
	#define initMatrixRandomCPU(M, N, ptr) initMatrixRandomCPU_float(M, N, ptr)
	#define buildBBlock(local_K, N, local_B, block_cols, block_rank, B_block) buildBBlock_float(local_K, N, local_B, block_cols, block_rank, B_block)
#else
	#define allocateMatrixCPU(M, N, ptr) allocateMatrixCPU_double(M, N, ptr)
	#define freeMatrixCPU(M, N, ptr) freeMatrixCPU_double(M, N, ptr)
	#define initMatrixCPU(M, N, ptr, val) initMatrixCPU_double(M, N, ptr, val)
	#define initMatrixRandomCPU(M, N, ptr) initMatrixRandomCPU_double(M, N, ptr)
	#define buildBBlock(local_K, N, local_B, block_cols, block_rank, B_block) buildBBlock_double(local_K, N, local_B, block_cols, block_rank, B_block)
#endif

#ifndef GEMM_M
#define GEMM_M 4096
#endif
#ifndef GEMM_N
#define GEMM_N 4096
#endif
#ifndef GEMM_K
#define GEMM_K 4096
#endif
#ifndef WARMUPS
#define WARMUPS 1
#endif
#ifndef ITERS
#define ITERS 10
#endif

int main(int argc, char **argv)
{
	elem_t *C;  // Only root needs this for final results
	
	int M = GEMM_M;
	int K = GEMM_K;
	int N = GEMM_N;
	
	int rank, npes;
	
	// Matrix pointers for local data - ALL processes have these
	elem_t *local_A;      // Each process's rows of A
	elem_t *local_B;      // Each process's rows of B  
	elem_t *local_C;      // Each process's result rows
	elem_t *B_block;      // Extracted block from local_B
	elem_t *B_buffer;     // Buffer to receive all B blocks via AllGather
	
	// Timing variables
	double total_start, total_end;
	double comm_start, comm_end;
	double comp_start, comp_end;
	double total_time, comm_time, comp_time;
	double comm_time_allgather = 0.0;  // Track AllGather time across iterations
	
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	
	int local_M = M / npes;  // Rows of A per process
	int local_K = K / npes;  // Rows of B per process (K dimension split)
	int B_block_cols = N / npes;  // Columns per block
	
	// Phase 1: Memory Allocation
	// ALL processes allocate their local matrices
	allocateMatrixCPU(local_M, K, &local_A);           // Each process's rows of A
	allocateMatrixCPU(local_K, N, &local_B);           // Each process's rows of B (full width)
	allocateMatrixCPU(local_M, N, &local_C);           // Each process's result rows
	allocateMatrixCPU(local_K, B_block_cols, &B_block); // Extracted block from local_B
	allocateMatrixCPU(K, B_block_cols, &B_buffer);     // Buffer for all B blocks (corrected size)
	
	// Only ROOT allocates C for final result collection
	if (rank == 0) {
		allocateMatrixCPU(M, N, &C);
	}
	
	// Start total timing
	MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
	total_start = MPI_Wtime();
	
	// Phase 2: Data Initialization
	// Each process initializes its OWN chunk of A and B
	initMatrixRandomCPU(local_M, K, local_A);
	initMatrixRandomCPU(local_K, N, local_B);
	initMatrixCPU(local_M, N, local_C, 0.0);
	
	// Phase 3 & 4: Block iteration with communication and computation
	comp_start = MPI_Wtime();
	
	elem_t alpha = 1.0;
	elem_t beta = 0.0;  // First iteration overwrites, subsequent add
	
	for (int block = 0; block < npes; block++) {
		// Communication: Block Extraction and Distribution
		comm_start = MPI_Wtime();
		
		// Each process builds/extracts its designated block from its local_B
		// Pass N as the width parameter since local_B has full width N
		buildBBlock(local_K, N, local_B, B_block_cols, block, B_block);
		
		// AllGather - each process shares its B block with all others
		MPI_Allgather(B_block, local_K * B_block_cols, MPI_ELEM_T,
		              B_buffer, local_K * B_block_cols, MPI_ELEM_T,
		              MPI_COMM_WORLD);
		
		comm_end = MPI_Wtime();
		comm_time_allgather += (comm_end - comm_start);
		
		// Computation: Matrix multiplication for this block
		// Calculate which columns of C this B block affects
		int start_col = block * B_block_cols;
		
		// Pointer to corresponding columns in local_C
		elem_t *current_C_block = local_C + start_col * local_M;
		
		// Compute: local_A × B_buffer → current_C_block
		// B_buffer now contains the complete reconstructed B columns for this block
		blasGemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		         local_M, B_block_cols, K, alpha, 
		         local_A, local_M, 
		         B_buffer, K,  // Note: leading dimension is K (full height)
		         beta, current_C_block, M);
		
		// After first iteration, switch to accumulation mode
		if (block == 0) beta = 1.0;
	}
	
	comp_end = MPI_Wtime();
	comp_time = comp_end - comp_start - comm_time_allgather;  // Subtract communication from computation
	
	// Phase 5: Result Collection (Using MPI_Gather)
	comm_start = MPI_Wtime();
	
	// All processes participate in gather operation
	// Each process contributes their local_C, root receives in C
	MPI_Gather(local_C, local_M*N, MPI_ELEM_T, 
	           C, local_M*N, MPI_ELEM_T, 
	           0, MPI_COMM_WORLD);
	
	comm_end = MPI_Wtime();
	double comm_time_gather = comm_end - comm_start;
	
	// End total timing
	total_end = MPI_Wtime();
	total_time = total_end - total_start;
	comm_time = comm_time_allgather + comm_time_gather;
	
	// Phase 6: Performance Analysis and Output (Root only)
	if (rank == 0) {
		double flops = 2.0 * (double)M * (double)N * (double)K;
		double total_gflops = (flops / 1.0e9) / total_time;
		double comp_gflops = (flops / 1.0e9) / comp_time;
		double comm_percentage = (comm_time / total_time) * 100.0;
		
		printf("========================================\n");
		printf("   MPI BLOCK-CYCLIC GEMM RESULTS       \n");
		printf("========================================\n");
		printf("Matrix size: %dx%dx%d\n", M, N, K);
		printf("Number of processes: %d\n", npes);
		printf("Rows per process (A): %d\n", local_M);
		printf("Rows per process (B): %d\n", local_K);
		printf("B block columns per process: %d\n", B_block_cols);
		printf("Total FLOPS: %.2f GFLOP\n", flops / 1.0e9);
		printf("----------------------------------------\n");
		
		printf("TIMING BREAKDOWN:\n");
		printf("  Total time:         %8.4f ms\n", total_time * 1000);
		printf("  Communication time: %8.4f ms (%.1f%%)\n", 
		       comm_time * 1000, comm_percentage);
		printf("  Computation time:   %8.4f ms (%.1f%%)\n", 
		       comp_time * 1000, 100.0 - comm_percentage);
		printf("----------------------------------------\n");
		
		printf("PERFORMANCE:\n");
		printf("  Total performance:  %8.2f GFLOP/s\n", total_gflops);
		printf("  Pure computation:   %8.2f GFLOP/s\n", comp_gflops);
		printf("  Parallel efficiency: %7.1f%%\n", (total_gflops/comp_gflops)*100);
		printf("----------------------------------------\n");
		
		printf("COMMUNICATION ANALYSIS:\n");
		printf("  AllGather (total):   %8.4f ms\n", comm_time_allgather * 1000);
		printf("  Final gather:        %8.4f ms\n", comm_time_gather * 1000);
		printf("  Comp/Comm ratio:     %8.2f:1\n", comp_time/comm_time);
		
		if (comm_percentage < 5.0) {
			printf("\n✓ EXCELLENT: Communication overhead < 5%%!\n");
		} else if (comm_percentage < 15.0) {
			printf("\n→ GOOD: Communication overhead < 15%%\n");
		} else {
			printf("\n⚠ HIGH: Communication overhead > 15%%\n");
		}
		
		printf("========================================\n\n");
	}
	
	// Phase 7: Cleanup
	// All processes free their local matrices
	freeMatrixCPU(local_M, K, local_A);
	freeMatrixCPU(local_K, N, local_B);
	freeMatrixCPU(local_M, N, local_C);
	freeMatrixCPU(local_K, B_block_cols, B_block);
	freeMatrixCPU(K, B_block_cols, B_buffer);
	
	// Root frees its specific matrices
	if (rank == 0) {
		freeMatrixCPU(M, N, C);
	}
	
	MPI_Finalize();
	
	return 0;
}