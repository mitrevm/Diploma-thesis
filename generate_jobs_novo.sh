#!/bin/bash

# Directory containing the config files
CONFIG_DIR="/d/hpc/home/mm5129/Diplomska/domains_only/cfgs"

# Directory where job files will be created
JOB_DIR="/d/hpc/home/mm5129/Diplomska/domains_only/jobs"

# Template for the job file
JOB_TEMPLATE='#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=haddock3
#SBATCH --cpus-per-task=50
#SBATCH --time=04:00:00

module load Anaconda3
source /d/hpc/home/mm5129/myenv/bin/activate
haddock3 CONFIG_FILE_PATH'

# Create jobs directory if it doesn't exist
mkdir -p "$JOB_DIR"

# Loop through each config file and create a corresponding job file
for config_file in "$CONFIG_DIR"/*.cfg; do
    if [ -f "$config_file" ]; then
        # Extract the filename without extension
        filename=$(basename "$config_file" .cfg)
        
        # Create job file name
        job_file="$JOB_DIR/job_${filename}.sh"
        
        # Create the job file with the config file path replaced
        echo "$JOB_TEMPLATE" | sed "s|CONFIG_FILE_PATH|$config_file|g" > "$job_file"
        
        # Make the job file executable
        chmod +x "$job_file"
        
        echo "Created job file: $job_file"
    fi
done

echo "Job generation complete!"