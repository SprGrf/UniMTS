# UniMTS: Unified Pre-training for Motion Time Series

Fork of the original repository (https://github.com/xiyuanzh/UniMTS), used for evaluation purposes during my Realistic HAR Master Thesis. To draw the attention and give credits to the original authors, I have removed most of the detailed implementation and usage instructions. For those, please follow the link to the original repository. 

Original paper citation:
```
@misc{zhang2024unimtsunifiedpretrainingmotion,
      title={UniMTS: Unified Pre-training for Motion Time Series}, 
      author={Xiyuan Zhang and Diyan Teng and Ranak Roy Chowdhury and Shuheng Li and Dezhi Hong and Rajesh K. Gupta and Jingbo Shang},
      year={2024},
      eprint={2410.19818},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2410.19818}, 
}
```
Original model weights at Hugging Face: https://huggingface.co/xiyuanz/UniMTS. Download and save in a _checkpoint_ folder inside the repository. Make sure you also include the pre-prosecced dataset files in a _UniMTS_data_ folder.

### Installation

```sh
conda create -n unimts python=3.9
conda activate unimts
pip install -r requirements.txt
```

Alternatively create a python virtual environment using the provided _requirements.txt_. 

### Fine-tune

Use the following command to finetune the model using all available samples for all 7 datasets. This will save the results in a _checkpoint_ folder, which you can then use to evaluate the different cases. 

```sh
python finetune_custom.py --padding_size 200 --batch_size 8  --checkpoint './checkpoint/UniMTS.pth' --config_path './label_dictionary.json' --joint_list 21 --original_sampling_rate 25 --num_class 7 --case_study cv
```

### Evaluation

Use the following command to evaluate the previously created and saved models for all datasets. Using the _case_study_ argument, choose between the cross validation (cv) and dataset to datasets (d2d) cases. This will save the results in a _results_ folder. 

```sh
python -u evaluate_custom.py --batch_size 64 --case_study d2d  --config_path './label_dictionary.json' --joint_list 21 --original_sampling_rate 25  --num_class 7
```


