#!/bin/bash
#SBATCH --job-name=pi_openmp_test
#SBATCH --output=pi_results_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --time=00:30:00
#SBATCH --account=TRAINING_T-7BPCJWTIUFG-DEFAULT-CPU

echo "========================================="
echo "Pi Calculation OpenMP Performance Test"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "========================================="

thread_count=(1 2 4 8 16 32 56)

for thread in "${thread_count[@]}"; do
        echo ""
        echo "Number of threads: $thread"
        echo "--------------------------"

        export OMP_NUM_THREADS=$thread

        ./pi_openMP
done

echo ""
echo "All tests completed!"