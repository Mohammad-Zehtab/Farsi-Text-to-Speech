# Farsi Text to Speech
Using a custom dataset (accessible from this link https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale) to train a tacotron2 model for text-to-speech conversion in the Farsi language by following the steps below for training.


conda env create -f environment.yml
conda activate tf_gpu

git clone --depth 1 --branch v0.9 https://github.com/TensorSpeech/TensorFlowTTS

rsync -a ./TensorFlowTTS/ ./mods/

cd TensorFlowTTS
pip install .

tensorflow-tts-preprocess --rootdir ./dataset --outdir ./dump --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech

tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech

CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./out/train.tacotron2.v1/ \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""
