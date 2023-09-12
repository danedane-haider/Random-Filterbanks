#!/bin/bash
#SBATCH --job-name=slurm_synth
#SBATCH --output=jobs/slurm_synth/slurm/slurm_%j__slurm_synth.log
#SBATCH --time=12:00:00


#########################
#
# Job: slurm_synth
# Args: 
# {'job_id': 'slurm_synth'}
#
#########################




###### ( hop into the singularity o_O )
singularity exec  \
    --overlay /scratch/user/ic24_overlay-15GB-500K.ext3:ro \
    cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
echo "@: entered singularity container"

[[ -f /ext3/env.sh ]] && source /ext3/env.sh
[[ -f /ext3/env ]] && source /ext3/env








echo "Stability of Random Filterbanks under Gradient Descent"

python slurm_synth.py \
    --job_id=slurm_synth


echo "Many thanks to Bea Steers, author of SLURMJOBS."



###### (escape from the singularity @o@ )
echo "@: exiting singularity container"
exit 0;
EOF

