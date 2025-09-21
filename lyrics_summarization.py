from openai import OpenAI
from sklearn.cluster import DBSCAN
import numpy as np

def local_summary_lyrics(lyrics, window, step):
    sentences = lyrics.split("\n")
    sentences = [sentence for sentence in sentences if len(sentence)!=0]
    start = 0
    llm = OpenAI()
    
    local_summaries = []
    flag = True
    
    while flag:
        local_sum_prompt = """너는 유능한 가사 요약 에이전트야. 너에게는 가사를 이루고 있는 여러 문장들이 주어질거고 그것을 잘 요약해야해.
        
    [가사]
    
    """
        endpoint = start + window
        if endpoint >= len(sentences):
            endpoint = len(sentences)
            flag = False 
        local_sentences = "\n".join(sentences[start:endpoint])
        local_sum_prompt += local_sentences
        messages = [{"role": "user", "content": local_sum_prompt}]
        
        completion = llm.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        temperature = 0.1,
        )
        
        local_summary = completion.choices[0].message.content
        local_summaries.append(local_summary)
        start += step
    
    return local_summaries

def rouge_f1(sentence1, sentence2):
    words1 = sentence1.split()
    words2 = sentence2.split()
    overlap = len(set(words1) & set(words2))
    return (2 * overlap) / (len(words1) + len(words2)) if (len(words1)+len(words2)) > 0 else 0.0

def rouge1_distance(sentence1, sentence2):
    return 1 - rouge_f1(sentence1, sentence2)




def filter_noise_clustering(local_summaries, eps, min_samples):
    n = len(local_summaries)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                dist = rouge1_distance(local_summaries[i], local_summaries[j])
                dist_matrix[i][j] = dist_matrix[j][i] = dist
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = dbscan.fit_predict(dist_matrix)
    filtered_local_summaries = [[] for _ in range(max(labels) + 1)]
 
    for i in range(len(local_summaries)):
        if labels[i] == -1:
            continue

        filtered_local_summaries[labels[i]].append(local_summaries[i])
 
    return filtered_local_summaries

def select_sentence(filtered_local_summaries):
    selected_sentences = []
    llm = OpenAI()
    
    for i in range(len(filtered_local_summaries)):
        curCluster = filtered_local_summaries[i]
        prompt = f"""너는 유능한 문장 분류 에이전트야. 너에게는 몇 개의 문장이 주어질 것이고, 그것들을 올바르게 각 카테고리로 나눠서 분류해야해.
        
        같은 카테고리에 속한 문장들은 같은 사실만을 말해야 하고, 다른 카테고리에 속한 문장은 다른 semantics를 가지고 있어야 해.  
        
        **전부 같은 카테고리에 속할 수도 있어**
        
        **출력할 때 근거는 출력하지말고, 출력 예의 형태만을 갖춘 답변을 출력해야해**
        
        [출력 예 1]
        
        1, 2 / 3, 4 / 5
        
        [출력 예 2]
        
        1, 2, 3, 4, 5
        
        [출력 예 3]
        
        1, 2, 3, 4 / 5
        
        [출력 예 4]
        
        1, 3, 5 / 2, 4
        
        [주어진 문장]
        
        {curCluster}
        
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        completion = llm.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        temperature = 0.1,
        )
        
        res = completion.choices[0].message.content
        
        selected_sentences.append(res)
    
    return selected_sentences


def final_sum(selected_sentences, filtered_local_summaries):
    total_context = ""
    llm = OpenAI()
    
    for i in range(len(selected_sentences)):
        l = selected_sentences[i].split(" / ")
        maxlength = 0
        maxClass = None
        for j in range(len(l)):
            nums = l[j].split(", ")
            if len(nums) > maxlength:
                maxClass = nums
                maxlength = len(nums)
               
        total_context += filtered_local_summaries[i][int(maxClass[-1]) - 1]
    
    
    modes = ["사실적 해석 모드", "상징적 해석 모드", "균형적 해석 모드", "정서적 해석 모드"]
    
    results = []
    
    for mode in modes:    
        prompt = f"""너는 유능한 가사 해석 에이전트야. 한 노래의 가사에 대한 묘사가 주어질 것이고, 그것들을 잘 해석해서 하나의 주제로 압축 요약해야해.
    
    너에게 가사를 해석하는 모드가 주어질 것이고, 해당 모드에 따라 가사를 엄밀하게 해석하여 하나의 주제로 요약해주어야 해.
    1. 사실적 해석 모드: 주어진 묘사들을 그대로 매끄럽게 연결하여 요약
    2. 상징적 해석 모드: 주어진 묘사들을 바탕으로 상징/비유/메타포 관점에서 자유롭게 해석 후 요약 (텍스트에서는 드러나지 않는 은유적, 사회적, 심리적 함의를 추론)
    3. 균형적 해석 모드: 주어진 묘사들에서 크게 벗어나지 않는 방향으로 상징적 의미를 추론 후 요약 (사실적 + 상징적)
    4. 정서적 해석 모드: 주어진 묘사들에서 표현하고 있는 감정 및 정서에 집중하여 해당 부분을 중심으로 요약
    
    [해석 모드] 
    {mode}
    
    [가사 묘사]
    {total_context}
    
        """
        messages = [{"role": "user", "content": prompt}]
        temperature = 0.9
        
        if mode == "사실적 해석 모드":
            temperature = 0.1
            
        completion = llm.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        temperature = temperature,
        )
        
        res = completion.choices[0].message.content
        
        results.append(res)
    
    return results
        


lyrics = """**[후렴]**

닭! 닭! 닭갈비~ 불판 위에 Party!

양념에 빠져버린 나 오늘 행복하지 🎉

배~ 배~ 배부르다 춤을 추자 같이

닭갈비 리듬 타며 모두 소리 질러 와!

**[1절]**

불판 위에 불꽃이 팡!

고소한 향기 맴돌아 쫙~

치즈 쭉 늘어나

한입 베어 물면 세상 다 가졌다!

**[프리-코러스]**

매콤달콤 에너지 업!

젓가락 멈출 수가 없어

배는 점점 불러오는데

행복 지수 터진다 Yeah!

**[후렴 반복]**

닭! 닭! 닭갈비~ 불판 위에 Party!

양념에 빠져버린 나 오늘 행복하지 🎉

배~ 배~ 배부르다 춤을 추자 같이

닭갈비 리듬 타며 모두 소리 질러 와!

**[브릿지]**

친구들과 웃고 떠들다 보면

배는 빵빵! 마음도 꽉 찼어

오늘 밤은 닭갈비의 노래

다 같이 불러봐!"""

l = local_summary_lyrics(lyrics, 10, 2)
l = filter_noise_clustering(l, 0.6, 2)
s = select_sentence(l)

print(final_sum(s, l))