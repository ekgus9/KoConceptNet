import json

data = ['train', 'dev', 'test']

for d in data:
    with open('../dataset/new_ko/korean_commongen_'+ d +'.json', 'r') as f:
        json_data = json.load(f)["concept_set"]
        for concept in json_data:
            # print(concept)
            if d == 'dev' or d == 'train' :
                src = concept["concept_set"].replace(' ','_').replace('#',' ')
                tgt1 = concept["reference_1"]
                
                with open('../dataset/new_ko/kommongen_'+ d +'.src_alpha.txt', 'a', encoding='utf-8') as k:
                    k.write(src + '\n')
                    
                with open('../dataset/new_ko/kommongen_'+ d +'.tgt.txt', 'a', encoding='utf-8') as k:
                    k.write(tgt1 + '\n')
                    
            else:

                src = concept["concept_set"]
                tgt1 = concept["reference_1"]
                tgt2 = concept["reference2"]
                tgt3 = concept["reference3"]
                
                with open('../dataset/new_ko/kommongen_'+ d +'.src_alpha.txt', 'a', encoding='utf-8') as k:
                    k.write(src + '\n')
                    k.write(src + '\n')
                    k.write(src + '\n')
                    
                with open('../dataset/new_ko/kommongen_'+ d +'.tgt.txt', 'a', encoding='utf-8') as k:
                    k.write(tgt1 + '\n')
                    k.write(tgt2 + '\n')
                    k.write(tgt3 + '\n')
                