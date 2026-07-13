clone over the repos for each ViT you would like to test. I have mine set up to do APHQ-ViT, PTQ4ViT, and DynamicViT, as well as also testing the baseline of DeiT-S. Clone each into the models folder. For the checkpoints you will have to download them from each of their git repos. For example go to APHQ-ViTs git and download their checkpoints for DeiT-S into the checkpoints folder. To run it simply type
. run.sh
and it will clear the cache for a clean run and start benchmark.py. The results and time series data will be generated and saved in the results folder.
