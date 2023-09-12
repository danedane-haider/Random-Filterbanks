import os
import slurmjobs

jobs = slurmjobs.Singularity(
    "python slurm_synth.py",
    f'/scratch/{os.getenv("USER")}/ic24_overlay-15GB-500K.ext3',
    "cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif",
    email='',
    sbatch=dict(time="12:00:00"),
    template="""{% extends 'job.singularity.j2' %}
      
{% block main %}
echo "Stability of Random Filterbanks under Gradient Descent"

{{ super() }}

echo "Many thanks to Bea Steers, author of SLURMJOBS."
{% endblock %}
    """,
)

# generate jobs across parameter grid
sav_dir = f'/scratch/{os.getenv("USER")}/icassp24_data'
run_script, job_paths = jobs.generate()

slurmjobs.util.summary(run_script, job_paths)
