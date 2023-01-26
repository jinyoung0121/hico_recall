import json
import numpy as np

recall_result={'R@1':[],'R@5':[],'R@10':[]}
recall={}
total_num=0
recall_1_count=0
recall_5_count=0
recall_10_count=0

with open('gt_triplet.json','r') as f:
    gts=json.load(f)

with open('image_per_gt.json','r') as f:
    image_per_gt=json.load(f)

with open('triplet_prediction_per_image_no_score.json','r') as f:
    triplet_prediction_per_image=json.load(f)

def recall_k(triplet_prediction_per_image,image_per_triplet,triplet,k):
    count=0
    for index,image in enumerate(image_per_triplet):
        length=min(k,len(triplet_prediction_per_image[image]))
        if triplet in triplet_prediction_per_image[image][:length]:
            count+=1
    return count

# for gt,num in gts.items():
#     recall_1=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,1)
#     recall_5=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,5)
#     recall_10=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,10)
#     recall[gt]={'R@1' : "%-6s" % round(recall_1/num,3)}
#     recall[gt]['R@5']="%-6s" % round(recall_5/num,3)
#     recall[gt]['R@10']="%-6s" % round(recall_10/num,3)
#     recall[gt]['type']='rare' if num<10 else 'non_rare'

# for key,value in recall.items():
#     print("%-13s : %s" % (key, value))

# -------------------------------------------------

for gt,num in gts.items():
    recall_1=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,1)
    recall_5=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,5)
    recall_10=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,10)
    recall_result['R@1'].append(round(recall_1/num,3))
    recall_result['R@5'].append(round(recall_5/num,3))
    recall_result['R@10'].append(round(recall_10/num,3))

print(np.mean(recall_result['R@1']))
print(np.mean(recall_result['R@5']))
print(np.mean(recall_result['R@10']))



for gt,num in gts.items():
    total_num+=num
    recall_1_count+=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,1)
    recall_5_count+=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,5)
    recall_10_count+=recall_k(triplet_prediction_per_image,image_per_gt[gt],gt,10)

print("R@1 : %.3f, R@5 : %.3f, R@10 : %.3f" % (recall_1_count/total_num,recall_5_count/total_num,recall_10_count/total_num))