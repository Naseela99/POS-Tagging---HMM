


import pandas as pd



df = pd.read_csv("./data/train", sep="\t", header=None)
df.columns = ["index", "word", "pos"]




counts = df.value_counts(subset="word", dropna=False)





counts = pd.DataFrame(counts, columns=["count"]).reset_index()




threshold = 1





counts.loc[counts["count"]<=threshold, "word"] = "<UNK>"





counts = (counts.groupby("word", as_index=False).sum().
          sort_values("count", ascending=False).
          sort_values("word", key=lambda x:x=='<UNK>', ascending=False, kind='mergesort' # to preserve last sort
                     )).reset_index(drop=True).reset_index()[["word", "index", "count"]]




counts.to_csv("vocab.txt", sep='\t', index=False, header=False)





print("Threshold:",threshold)





print("Total size of vocabulary",counts.shape[0])

print("Total occurrences of the special token < unk > after replacement:  ",counts.query("word=='<UNK>'").iloc[0, 2])


# ### Matrices



pos = df[["index", "pos"]]



uniq_pos = pos["pos"].unique().tolist()




uniq_pos.append("<S>")
uniq_pos.append("<E>")




idx_pos = dict(enumerate(uniq_pos))
pos_idx = {v:k for k,v in idx_pos.items()}




word_idx = counts.set_index("word").drop(labels="count", axis=1)["index"].to_dict()
idx_word = {v:k for k,v in word_idx.items()}



df["word"] = df["word"].apply(lambda x : idx_word[word_idx.get(x, 0)]) 




emission = df.groupby(["word", "pos"], as_index=False).count().pivot("word", "pos", "index").fillna(0)





emission = emission/emission.sum()





import numpy as np





emission_dict = pd.melt(emission, ignore_index=False).reset_index().set_index(["pos", "word"])["value"].to_dict()





emission_dict_str = {str(k):v for k, v  in emission_dict.items()}





data = {}






startindices = df[df["index"]==1].index
endindices = startindices-1

startindices = startindices.tolist()
endindices = endindices.tolist()
endindices.pop(0)
endindices.append(len(df)-1)

sentences = []
for st, en in zip(startindices, endindices):
    sentences.append(["<S>"]+df.loc[st:en, "pos"].values.tolist()+["<E>"])





from collections import defaultdict





count = defaultdict(int)
transition_freq = defaultdict(int)
for sentence in sentences:
    for i in range(len(sentence)-1):
        transition_freq[(sentence[i], sentence[i+1])] += 1
        count[sentence[i]] += 1





transition_dict = {}
for k, v in transition_freq.items():
    transition_dict[k] = v/count[k[0]]





transition_dict_str = {str(k):v for k, v  in transition_dict.items()}





data["transition"] = transition_dict_str
data["emission"] = emission_dict_str




import json





with open("hmm.json", "w") as f:
    
    json.dump(data, f, indent=4)





print("Number of transition parameters",len(transition_dict))
print("Number of emission parameters", len(emission_dict))


# ### Greedy Decoding




## convert transition_dict to transition_matrix
transition_matrix = np.zeros((len(uniq_pos), len(uniq_pos)))
for (pos1, pos2), prob  in transition_dict.items():
    transition_matrix[pos_idx[pos1], pos_idx[pos2]] = prob





## convert emission_dict to emission_matrix
emission_matrix = np.zeros((len(uniq_pos), len(word_idx)))
for (pos, word), prob in emission_dict.items():
    emission_matrix[pos_idx[pos], word_idx[word]] = prob





def greedy(sentence, transition, emission):
    tags = []
    current_tag = "<S>"
    for current_word in sentence:
        transition_probs = transition_matrix[pos_idx[current_tag]]
        emission_probs = emission_matrix[:, word_idx[current_word]]

        total_prob = transition_probs*emission_probs

        tag = idx_pos[np.argmax(total_prob)]
        tags.append(tag)
        current_tag = tag
    return tags





dev = pd.read_csv("./data/dev", sep='\t', header=None)





dev.columns = ["index", "word", "pos"]





startindices = dev[dev["index"]==1].index
endindices = startindices-1

startindices = startindices.tolist()
endindices = endindices.tolist()
endindices.pop(0)
endindices.append(len(dev)-1)

dev_sentences = []
for st, en in zip(startindices, endindices):
    dev_sentences.append(list(zip(*dev.loc[st:en, ["word", "pos"]].values.tolist())))





results = []
for sentence, tags in dev_sentences:
    sentence = [idx_word[word_idx.get(word, 0)] for word in sentence]
    pred_tags = greedy(sentence, transition_matrix, emission_matrix)
    results.extend(list(zip(pred_tags, tags)))
    
results = np.array(results)

print('Greedy Accuracy on dev data',np.mean(results[:, 0]==results[:, 1])*100)





test = pd.read_csv("./data/test",sep='\t', header=None)
test.columns = ["index", "word"]





startindices = test[test["index"]==1].index
endindices = startindices-1

startindices = startindices.tolist()
endindices = endindices.tolist()
endindices.pop(0)
endindices.append(len(test)-1)

test_sentences = []
for st, en in zip(startindices, endindices):
    test_sentences.append(test.loc[st:en, "word"].values.tolist())





results = []
for sentence in test_sentences:
    sentence = [idx_word[word_idx.get(word, 0)] for word in sentence]
    pred_tags = greedy(sentence, transition_matrix, emission_matrix)
    results.extend(pred_tags)
    





test["pred_pos"] = results

test = test.applymap(str)
val = test.values
res = ''

for v in val:
	if v[0] == '1':
		res +='\n'+"\t".join(v)
	else:
		res+="\t".join(v)
	res+='\n'
		


with open('greedy.out', 'w') as f:
	f.write(res[1:])




# ### Viterbi




def viterbi(sentence, transition, emission):
    global cur_probs, back
    
    dp = np.zeros((len(uniq_pos), len(sentence)+1))
    back = []
    
    word = sentence[0]
    t = transition_matrix[pos_idx["<S>"]]
    e = emission_matrix[:, word_idx[word]]
    cur_prob = t*e
    dp[:, 0] = cur_prob
    for idx, word in enumerate(sentence[1:], 1):
        cur_probs = []
        
        e = emission_matrix[:, word_idx[word]]
        for tag in uniq_pos:
            t = transition_matrix[pos_idx[tag]]
            prev_prob = dp[pos_idx[tag], idx-1]
            cur_prob = prev_prob*t*e
            cur_probs.append(cur_prob)
        cur_probs = np.array(cur_probs)
        probs = np.max(cur_probs, axis=-1)
        back.append(np.argmax(cur_probs, axis=-1))
        dp[:, idx]=probs

        
            
            
#         break
    return dp, back





def viterbi_decoding(sentence, transition_dict, emission_dict):


    dp = np.zeros((len(sentence), len(transition_dict)))
    back = np.zeros((len(sentence), len(transition_dict))).astype(int)

    for tag1, tag2 in transition_dict:
        if tag1 == "<S>":
            dp[0, pos_idx[tag2]] = transition_dict.get((tag1, tag2), 0) * emission_dict.get((tag2, sentence[0]), 0)
            back[0, pos_idx[tag2]] = 0

    for i in range(1, len(sentence)):
        for tag2 in pos_idx:
            max_prob = 0
            for tag1 in pos_idx:
                prob = dp[i-1, pos_idx[tag1]] * transition_dict.get((tag1, tag2), 0) * emission_dict.get((tag2, sentence[i]), 0)
                if prob > max_prob:
                    max_prob = prob
                    back[i, pos_idx[tag2]] = pos_idx[tag1]
            dp[i, pos_idx[tag2]] = max_prob

    best_path = []
    best_path.append(np.argmax(dp[-1, :]))
    for i in range(len(sentence)-1, 0, -1):
        
        best_path.append(back[i, best_path[-1]])
  
    return reversed([idx_pos[idx] for idx in best_path])






results = []
for sentence, tags in dev_sentences:
    sentence = [idx_word[word_idx.get(word, 0)] for word in sentence]
    pred_tags = viterbi_decoding(sentence, transition_dict, emission_dict)
    results.extend(list(zip(pred_tags, tags)))
    
results = np.array(results)

print('Viterbi decoding accuracy on dev data',np.mean(results[:, 0]==results[:, 1])*100)





results = []
for sentence in test_sentences:
    sentence = [idx_word[word_idx.get(word, 0)] for word in sentence]
    pred_tags = viterbi_decoding(sentence, transition_dict, emission_dict)
    results.extend(pred_tags)
    





test["pred_pos"] = results


test = test.applymap(str)
val = test.values
res = ''

for v in val:
	if v[0] == '1':
		res +='\n'+"\t".join(v)
	else:
		res+="\t".join(v)
	res+='\n'
		


with open('viterbi.out', 'w') as f:
	f.write(res[1:])




