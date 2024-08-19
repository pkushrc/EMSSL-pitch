# EMSSL
Train a model to do acoustic-to-articulatory inversion with EMSSL framework.  

Usage of EMSSL

------------------------------------------------------------------------------------------------------
Preparation:

0.1. Build gnuspeech_sa as synthesizer (at any location).

   git clone -b modify_io https://github.com/godlovesun/gnuspeech_sa.git

   cd gnuspeech_sa --> mkdir build --> cd build --> cmake ../ --> make 

   We refer to ./gnuspeech_sa/build as GNUSPEECH_DIR

0.2. Dependencies.

   python 3.7

   Pytorch 1.3.0
   
   Other dependencies are listed in requirements.txt

-----------------------------------------------------------------------------------------

1. Prepare moments for model (for data normalization).

   cd ./misc

   python spec_trm_statistics.py --moments_path example_path/moments.pkl

   We refer to example_path/moments.pkl as MOMENTS_PATH

2. Preprocess dataset.

   Assume a dataset with train/eval/test splits is available, each split directly contains mixed audios (.wav). We refer to the path of the raw audio dataset as RAW_AUDIO_DIR and the path of the processed dataset as PROCESSED_AUDIO_DIR.  Use the following commands to preprocess training data (similar for the evaluation and testing splits)

   cd ./misc

   python dataset.py --moments_path MOMENTS_PATH --source_dir RAW_AUDIO_DIR/train --target_dir PROCESSED_AUDIO_DIR/train

3. Modify ./misc/config.py with proper parameter settings (mainly configure the GLOBAL, TRAIN, ITERTRAIN, EVAL, UTIL options)

4. Train with EMSSL.

   cd ./train

   python  a2t_trainer.py

