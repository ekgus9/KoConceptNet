# Knowledge Graph-Augmented Korean Generative Commonsense Reasoning

### 1. Preparing data  
```
python gen_entity.py
python reorder_src.py
python conceptnet.py

```
### 2. Training Entity and Relation Embedding

```
python n-n.py
python train_transe_ConceptNet.py
python entity_onehot.py
```
### 3. Train and evaluate 
``` # cd KG-BART/KGBART/KGBART_training
python run_seq2seq.py --data_dir  ../../dataset/new_ko_t --output_dir ../output/KoBart_new_t --bart_model gogamza/kobart-base-v2 \
    --log_dir ../log/KoBart --fp16 False --local_rank -1\
    --max_seq_length 32 --max_position_embeddings 64 --max_len_a 32 --max_len_b 64 --max_pred 64 \
    --train_batch_size 60 --eval_batch_size 48 --gradient_accumulation_steps 6 --learning_rate 0.00001 \
    --warmup_proportion 0.1 --label_smoothing 0.1 --num_train_epochs 1
```
### 4. Test CommonGen
```
python decode_seq2seq.py --data_dir ../../dataset/new_ko_t --model_recover_path ../output/KoBart_new_t/best_model/model.best.bin \
 --input_file ../../dataset/new_ko_t/kommongen_test.src_alpha.txt --output_dir ../output/KoBart_new_t/best_model/Gen \
 --output_file model.best --split test --beam_size 1 --forbid_duplicate_ngrams True 
```
## 5. Evaluation
```
kobertscore git install 
python korean_commongen_evaluation_multi_ref.py --reference_file dataset/korean_commongen_official_test.txt --prediction_file baseline_results/$TASK_NAME$/$MODEL_GENERATE.TXT$ --model $MODEL_NAME$
```

## Citation

If you use or extend our work, please cite the following paper:

```
@misc{jung2023knowledge,
      title={Knowledge Graph-Augmented Korean Generative Commonsense Reasoning}, 
      author={Dahyun Jung and Jaehyung Seo and Jaewook Lee and Chanjun Park and Heuiseok Lim},
      year={2023},
      eprint={2306.14470},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
