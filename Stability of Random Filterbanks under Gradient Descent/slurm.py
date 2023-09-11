import os
import slurmjobs

jobs = slurmjobs.Singularity(
    "python main.py",
    f'/scratch/{os.getenv("USER")}/wa23_overlay-15GB-500K.ext3',
    "cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif",
    email='',
    sbatch=dict(time="2:00:00"),
    template="""{% extends 'job.singularity.j2' %}
      
{% block main %}
echo "Optimizing auditory filterbanks by gradient descent"

{{ super() }}

echo "Many thanks to Bea Steers, author of SLURMJOBS."
{% endblock %}
    """,
)

# generate jobs across parameter grid
sav_dir = f'/scratch/{os.getenv("USER")}/waspaa2023_filterbanks_data'
run_script, job_paths = jobs.generate(
    [
        ("domain", ["music", "speech", "synth", "urban"]),
        ("arch", ["TDFilterbank", "Gabor1D", "MuReNN"]),
        ("init_id", range(5)),
    ],
    batch_size=64,
    sav_dir=sav_dir,
)

slurmjobs.util.summary(run_script, job_paths)
