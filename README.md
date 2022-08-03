# DT-Net
Explain network.
2 training approaches.
- "new" is unsupervised using anisotropic LCC loss using middle channel.
- "super" is supervised training using all 6 channels.

The folder "model" has the pretrained models with the corresponding loss functions.

Dataset description
- "/srv/beegfs02/scratch/sosrecon/data" path has 870 SoS maps with beamformed RF data for all 6 channels.
- "/srv/beegfs02/scratch/sosrecon/data/MS" has augmented dataset with RF data corresponding to only middle channel
 
Use norm_para.py script for finding the normalization parameters of the dataset.

Testing 
- "test_new" is used for testing results using LCC loss (unsupervised).
- "test_super" is used for supervised model (using pseudo displacement fields)
