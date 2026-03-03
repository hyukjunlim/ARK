# Update the job name

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"

module purge

StartTime=$(date +%s)
# cd $SLURM_SUBMIT_DIR

# Run the specified Python script
sh scripts/train/oc20/s2ef/crack/run.sh

EndTime=$(date +%s)

# Calculate and display runtime
RUNTIME=$((EndTime - StartTime))
echo "Run time:"
echo "${RUNTIME} sec"
echo "$(echo "scale=2; ${RUNTIME}/60" | bc) min"
echo "$(echo "scale=2; ${RUNTIME}/3600" | bc) hour"
echo "Job completed at: $(date)"

