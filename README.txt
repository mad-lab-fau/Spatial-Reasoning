This is an abridged example of how to run the experiments for tinyImagenet from the paper "From Patches to Objects: Exploiting Spatial Reasoning for Better Visual Representations"
It is meant as a way to briefly show the minimal changes necessary to implement the method and rerun experiments from the paper.
As written in the Paper, the implementation is based on the code from the
Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
see GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning for the full version.
To make things more readable we have removed code for other methods and datasets.
We plan to release the full and cleaned version of the code in the future.


This command can be used for unsupervised training:
python train_unsupervised.py --dataset="tiny" --method="patchbased" --backbone="resnet32" --seed=1 --data_size=128 --K=4 --gpu=0 --epochs=200 --num_workers 1 --patchsize 24 --patchcount 3 --id "pc3s1"
The standard patch count is 2.
For additive use, add the flag  --additive


This command is used for the linear evaluation:
python train_linear_evaluation.py --dataset="tiny" --method="patchbased" --patchcount=0 --backbone="resnet32" --seed=2 --data_size=32 --gpu=0 --patchsize 24 --id "pc3aevals2" --checkpoint "./examplePath.tar"
For additvely trained networks, set patchcount to 0.
The default patchcount is 9, other options, as discussed in the paper, are 1,3,5,7. Further options can easily be added.

When rerunning the experiments, please make sure to structure the train and val folder of tiny imagenet as necessary for the pytorch dataset.
See https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py for an examplary way of restructuring the validation set (downloadandcheck.ipynb should also work for the val set).

