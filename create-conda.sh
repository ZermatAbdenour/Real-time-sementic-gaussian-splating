conda create -n rtsgs python=3.11 -y
conda activate rtsgs
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
-c pytorch -c nvidia
