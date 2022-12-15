import datasets
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab
from transformers import *
from korouge import *
from KoBERTScore import BERTScore


## For heatmap
data = {
    'bleu3':[],
    'bleu4':[],
    'rouge2':[],
    'rougeL':[],
    'meteor':[],
    'mbert':[],
    'kobert':[],
    'coverage':[]
}



bleu_metric = datasets.load_metric('bleu')
meteor_metric = datasets.load_metric('meteor')
rouge = Rouge()

def coverage_score(preds, concept_sets, tokenizer):
    covs = []
    for p, cs in tqdm(zip(preds, concept_sets), total=len(preds)):
        cs = set(cs)
        if '#' in cs:
            cs.remove('#')
        lemmas = set()
        for token in tokenizer(p):
            lemmas.add(token)
        cov = len(lemmas & cs) / len(cs)
        data['coverage'].append(cov)
        covs.append(cov)
    return sum(covs) / len(covs)


def scoring(preds, concept_sets, tokenizer):
    Coverage = coverage_score(preds, concept_sets, tokenizer)
    print(f"System level Concept Coverage: {Coverage * 100:.2f}")


def main(args):
    print("Start KommonGen Evaluation")
    mmodel = "bert-base-multilingual-cased"
    kmodel = "monologg/kobert"
    kbertscore = BERTScore(kmodel, best_layer=2)
    mbertscore = BERTScore(mmodel, best_layer=2)
    concept_sets = []

    rouge2 = []
    rougeL = []
    bleu_predictions = []
    bleu_references = []
    met_references = []
    met_predictions = []
    mecab_tokenizer = Mecab("/usr/local/lib/mecab/dic/mecab-ko-dic").morphs

    bert_prds= []
    bert_refs = []



    with open(args.reference_file, 'r', encoding='utf-8') as f, open(args.prediction_file, 'r', encoding='utf-8') as f1:
        refs_list = f.readlines()
        preds_list = f1.readlines()
        preds = [prd.strip() for prd in preds_list]
        refs = [ref.strip() for ref in refs_list]
        
        num = 0
        while True:
            # concept_set = mecab_tokenizer(ref.split(' = ')[0].replace('[SOS] ', '').strip())
            concept_set = mecab_tokenizer(refs[num].strip())
            concept_sets.append(concept_set)
            # print(refs[num:num+3])
            # print(preds[num])
            # if num > 20 : exit()

            # For BLEU score
            bleu_reference1 = [mecab_tokenizer(refs[num].strip())]
            bleu_reference2 = [mecab_tokenizer(refs[num+1].strip())]
            bleu_reference3 = [mecab_tokenizer(refs[num+2].strip())]

            bleu_references.extend([bleu_reference1, bleu_reference2, bleu_reference3])

            bleu_prediction = mecab_tokenizer(preds[num].strip())
            bleu_predictions.append(bleu_prediction)

            # For METEOR score
            rouge_met_reference1 = refs[num].strip()
            rouge_met_reference2 = refs[num+1].strip()
            rouge_met_reference3 = refs[num+2].strip()

            met_references.append(rouge_met_reference1)
            met_references.append(rouge_met_reference2)
            met_references.append(rouge_met_reference3)

            met_prediction = preds[num].strip()
            met_predictions.append(met_prediction)

            # For Rouge score
            concept_set = refs[num].strip()
            line1 = {"concept-set": concept_set, "scene": rouge_met_reference1}
            line2 = {"concept-set": concept_set, "scene": rouge_met_reference2}
            line3 = {"concept-set": concept_set, "scene": rouge_met_reference3}
            rouge_bert_ref = []

            rouge_bert_ref.append(str(line1)); rouge_bert_ref.append(str(line2)); rouge_bert_ref.append(str(line3))

            ref_lines = rouge_bert_ref
            total_rouge_2_score = []
            total_rouge_L_score = []
            for ref_line in ref_lines:
                tmp_dict = eval(ref_line)
                real_ref_line = tmp_dict['scene']

                cur_r2 = rouge.calc_score_2([preds[num]], [real_ref_line])
                cur_rl = rouge.calc_score_L([preds[num]], [real_ref_line])

                total_rouge_2_score.append(cur_r2)
                total_rouge_L_score.append(cur_rl)

                bert_prds.append(preds[num].strip())
                bert_refs.append(real_ref_line.strip())

            max_rouge_2_score = max(total_rouge_2_score)
            max_rouge_L_score = max(total_rouge_L_score)

            rouge2.append(max_rouge_2_score)
            rougeL.append(max_rouge_L_score)
            
            num += 3
            
            if num >= len(refs)-1 : break

        cur_mBS = mbertscore(bert_prds, bert_refs, batch_size=128)
        cur_kBS = kbertscore(bert_prds, bert_refs, batch_size=128)

        # BLEU 3/4 - max
        bleu3, bleu4, meteors = [], [], []
        for i in range(len(bleu_predictions)):
            #print(bleu_predictions[i]); print([bleu_references[3*i]])
            bleu_ref_first = bleu_metric.compute(predictions=[bleu_predictions[i]], references=[bleu_references[3*i]], max_order=4)
            bleu_ref_second = bleu_metric.compute(predictions=[bleu_predictions[i]], references=[bleu_references[3*i+1]], max_order=4)
            bleu_ref_third = bleu_metric.compute(predictions=[bleu_predictions[i]], references=[bleu_references[3*i+2]], max_order=4)

            meteor_ref_first = meteor_metric.compute(predictions=[met_predictions[i]], references=[met_references[3*i]])
            meteor_ref_second = meteor_metric.compute(predictions=[met_predictions[i]], references=[met_references[3*i+1]])
            meteor_ref_third = meteor_metric.compute(predictions=[met_predictions[i]], references=[met_references[3*i+2]])

            meteor_max = max(round(meteor_ref_first['meteor'], 4), round(meteor_ref_second['meteor'], 4), round(meteor_ref_third['meteor'], 4))
            meteors.append(meteor_max)

            bleu3_max = max(round(bleu_ref_first['precisions'][2], 4), round(bleu_ref_second['precisions'][2], 4), round(bleu_ref_third['precisions'][2], 4))
            bleu3.append(bleu3_max)
            bleu4_max = max(round(bleu_ref_first['precisions'][3], 4), round(bleu_ref_second['precisions'][3], 4), round(bleu_ref_third['precisions'][3], 4))
            bleu4.append(bleu4_max)

        print("BLEU 3: ", round(np.mean(bleu3),4))
        print("BLEU 4: ", round(np.mean(bleu4),4))

        print("ROUGE-2: ", round(np.mean(rouge2),4))
        print("ROUGE-L: ", round(np.mean(rougeL),4))

        print("METEOR: ", round(np.mean(meteors),4))

        print("mBERTScore: ", round(np.mean(cur_mBS),4))
        print("KoBERTScore: ", round(np.mean(cur_kBS),4))

        data['bleu3'] = bleu3
        data['bleu4'] = bleu4
        data['rouge2'] = rouge2
        data['rougeL'] = rougeL
        data['meteor'] = meteors

        ktmp = []
        mtmp = []
        cur_kBS_heatmap = []
        cur_mBS_heatmap = []
        for i, (k,m) in enumerate(zip(cur_kBS, cur_mBS)):
            if (i+1) % 3 == 0:
                kscore = np.mean(ktmp)
                mscore = np.mean(mtmp)
                cur_kBS_heatmap.append(kscore)
                cur_mBS_heatmap.append(mscore)
                ktmp = []
                mtmp = []
            else:
                ktmp.append(k)
                mtmp.append(m)

        data['kobert'] = cur_kBS_heatmap
        data['mbert'] = cur_mBS_heatmap

        scoring(preds, concept_sets, mecab_tokenizer)


        ## For heatmap
        # print(data)
        df = pd.DataFrame(data)
        df.to_csv('eval_multi_.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--prediction_file", type=str)
    # python eval.py --reference_file kommongen_test.tgt.txt --prediction_file model.best
    # /home/mnt/dhaabb55/Concept/KG-BART/evaluation
    args = parser.parse_args()
    main(args)
