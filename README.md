# MIR_SGPR
#Installation

Clone this repo.

git clone https://github.com/tongkusdxx/MIR_SGPR.git

#requirements:

pip install requirements.txt


#Training New Models

To train on the you dataset, for example.

python train.py --st_root=[the path of structure images] --de_root=[the path of ground truth images] --mask_root=[the path of mask images]

#Pre-trained weights and test model

You can download the pre-trained model 
https://drive.google.com/drive/folders/1WJQFExfjm6xI_r3PEaGIRlHumJGHDCvC


#test data:image_for_test

#test :

sbatch test.sh

or

python test.py --st_root="../MIR_SGPR/image_for_test/st/" --de_root="../MIR_SGPR/image_for_test/gt/" --mask_root="../MIR_SGPR/image_for_test/mask/"

