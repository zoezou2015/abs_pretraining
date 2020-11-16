## STEP: Sequence-to-Sequence Transformer Pre-training for Document Summarization

## Installation
You need to install python3 and the dependencies in the ```requirements.txt```.
```
pip install -r requirements.txt
```

## CNN/DailyMail Dataset
You can download the datasets [here](https://microsoft-my.sharepoint.com/:u:/g/personal/xizhang_microsoft_com1/ERaYAOUGjwNCpkjfeg8J0QYBSHc9T-jWUOSmTIFJ6l9tPA?e=VkDCfx) <br>
The `cnndm_raw` dataset is the raw CNN/Dailymail dataset in text format, which is used for evaluation. <br>
The `cnndm_article512_summary256_roberta_large` dataset is in binary format and is used for finetuning.

## Model Output
TBD

## Trained Models with SR
* STEP_SR [download](https://xingxingzhang.blob.core.windows.net/share/step/Pre-trained_Model/SR/checkpoint30.pt)
 * `STEP_AbsSum_model_backup/Pre-trained_Model/SR/checkpoint30.pt` is the model after pre-training with the Sentence Reordering (SR) task on GIGA-CM dataset. Note that this stage only requires unlabeled documents.
 

## Trained Models with NSG
* STEP_NSG [download](https://xingxingzhang.blob.core.windows.net/share/step/Pre-trained_Model/NSG/checkpoint20.pt)
 * `STEP_AbsSum_model_backup/Pre-trained_Model/NSG/checkpoint20.pt` is the model after pre-training with the Next Sentence Generation (NSG) task on GIGA-CM dataset. Note that this stage only requires unlabeled documents.
 

## Trained Models with MDG
* STEP_MDG [download](https://xingxingzhang.blob.core.windows.net/share/step/Pre-trained_Model/MDG/checkpoint30.pt)
 * `STEP_AbsSum_model_backup/Pre-trained_Model/MDG/checkpoint30.pt` is the model after pre-training with the Masked Document Generation (MDG) task on GIGA-CM dataset. Note that this stage only requires unlabeled documents.
 

## Trained Models with ALL
* STEP_ALL [download](https://xingxingzhang.blob.core.windows.net/share/step/Pre-trained_Model/ALL/checkpoint20.pt)
 * `STEP_AbsSum_model_backup/Pre-trained_Model/ALL/checkpoint20.pt` is the model after pre-training with all the three tasks (ALL) on GIGA-CM dataset. Note that this stage only requires unlabeled documents.
 

## Open-domain Pre-training
The cost of open-domain pre-training is large and you can download the models after pre-training following the instructions in the previous section. <br>

The following is a script used for open-domain pre-training.
```
codedir=/mnt/yanyan/abstractive_summarization_pretraining
raw_data_dir=/mnt/yanyan/abstractive_sum_evaluate_zoe/cnndm_raw
raw_valid=$raw_data_dir/validation
raw_test=$raw_data_dir/test
evaldir=/mnt/yanyan/abstractive_sum_evaluate_zoe
data=giga_cnndm_nsp_roberta
datadir=/mnt/yanyan/abstractive_summarization_pretraining/data-bin/$data

curdir=`pwd`
pip install pytorch-transformers==1.1.0 --user
cd $curdir

lr=2e-5
dec_lr=1e-4
max_epoch=100
batch_size=2
dropout=0.1
dec_dropout=0.3
update_freq=64
bsz=1024
model_name=abs_sum_roberta_transformer_large
task=abstractive_summarization_roberta
modeldir=/mnt/yanyan/experiments/roberta_large/giga_ensemble_${model_name}_lr${lr}_declr${dec_lr}_bsz${bsz}_dp${dropout}_decdp_${dec_dropout}/ensemble_checkpoints
logdir=$modeldir/logs
max_update=1000000

mkdir -p $modeldir
mkdir -p  $logdir

python $codedir/scripts/backup_log.py $modeldir

python -u $codedir/train.py $datadir --left-pad-source False --ensemble-pretrain --min-mask-len 100 --max-mask-len 256  \
 --arch ${model_name} --task $task --sep-optim --dec-lr ${dec_lr} --decoder-dropout ${dec_dropout} --roberta-model roberta-large --param-name encoder.roberta \
 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --warmup-dec-updates 10000 --warmup-init-dec-lr 1e-07 \
 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 10000 \
 --lr $lr --min-lr 1e-09 --validate-interval 1 \
 --dropout ${dropout} --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
 --update-freq ${update_freq} \
 --max-update ${max_update} --save-interval-updates 2000 \
 --max-sentences ${batch_size} \
 --ddp-backend=no_c10d --save-dir ${modeldir} \
 --required-batch-size-multiple 1 \
 --max-source-positions 512 --max-target-positions 256 \
 --log-interval 1000 2>&1 | tee $modeldir/log.txt
```

## In-domain finetuning
Here is the script for doing finetuing on CNNDM dataset.
```
codedir=/mnt/yanyan/abstractive_summarization_pretraining
raw_data_dir=/mnt/yanyan/abstractive_sum_evaluate_zoe/cnndm_raw
raw_valid=$raw_data_dir/validation
raw_test=$raw_data_dir/test
evaldir=/mnt/yanyan/abstractive_sum_evaluate_zoe
data=cnndm_article512_summary256_roberta_large
datadir=/mnt/yanyan/abstractive_summarization_pretraining/data-bin/$data

curdir=`pwd`
pip install pytorch-transformers==1.1.0 --user
cd $curdir

lr=2e-5
dec_lr=1e-4
max_epoch=100
batch_size=2
dropout=0.1
dec_dropout=0.3
update_freq=16
bsz=256
model_name=abs_sum_roberta_transformer_large
task=abstractive_summarization_roberta
modeldir=/mnt/yanyan/experiments/roberta_large/giga_ensemble_abs_sum_roberta_transformer_large_lr2e-5_declr1e-4_bsz1024_dp0.1_decdp_0.3/ensemble_checkpoints
logdir=$modeldir/logs

ft_lr=2e-5
ft_max_epoch=30
ft_dec_lr=2e-5
ft_batch_size=2
ft_update_freq=48
ft_bsz=768
ft_modeldir=/mnt/yanyan/experiments/roberta_large/giga_ensemble_abs_sum_roberta_transformer_large_lr2e-5_declr1e-4_bsz1024_dp0.1_decdp_0.3/open_domain_ensemble_ft_lr${ft_lr}_declr${ft_dec_lr}_bsz${ft_bsz}_dp${dropout}_decdp${dec_dropout}
ft_logdir=${ft_modeldir}/logs

mkdir -p ${ft_modeldir}
mkdir -p  ${ft_logdir}

python $codedir/scripts/backup_log.py ${ft_modeldir}

python -u $codedir/train.py $datadir --left-pad-source False  --init-from-pretrained-model True --pretrained-model-path $modeldir/checkpoint21.pt  \
 --arch ${model_name} --task $task --sep-optim --dec-lr ${ft_dec_lr} --decoder-dropout ${dec_dropout} --roberta-model roberta-large --param-name encoder.roberta \
 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --warmup-dec-updates 4000 --warmup-init-dec-lr 1e-07 \
 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
 --lr ${ft_lr} --min-lr 1e-09 --validate-interval 1 \
 --dropout ${dropout} --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
 --update-freq ${ft_update_freq} \
 --max-epoch ${ft_max_epoch} \
 --save-dir ${ft_modeldir} \
 --max-sentences ${ft_batch_size} \
 --ddp-backend=no_c10d  \
 --required-batch-size-multiple 1 \
 --max-source-positions 512 --max-target-positions 256 \
 --log-interval 100 2>&1 | tee ${ft_modeldir}/log.txt
```
Here is the script for computing the ROUGE scores
```
START=10
END=30
for ((epoch=${START};epoch<=${END};epoch+=1)); do
       python3.6  $codedir/generate.py $datadir --task $task --isRoberta \
               --path ${ft_modeldir}/checkpoint$epoch.pt  --batch-size 64 --beam 5 --remove-bpe --min-len 60 \
               --gen-subset test  --no-repeat-ngram-size 3 \
               > ${ft_logdir}/epoch.$epoch.test.txt

       python3.6  $codedir/generate.py $datadir --task $task --isRoberta \
               --path ${ft_modeldir}/checkpoint$epoch.pt  --batch-size 64 --beam 5 --remove-bpe --min-len 60 \
               --gen-subset valid --no-repeat-ngram-size 3  \
               > ${ft_logdir}/epoch.$epoch.valid.txt

       python3.6 -u $evaldir/abstractive_sum_eval_pipe.py -ncpu 2 \
               -topk 3 -raw_valid $raw_valid -model_name roberta -lowercase False  \
               -raw_test $raw_test \
               -result_valid ${ft_logdir}/epoch.$epoch.valid.txt \
               -result_test ${ft_logdir}/epoch.$epoch.test.txt 
done
```
