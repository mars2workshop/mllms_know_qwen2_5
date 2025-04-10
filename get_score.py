import os
import json
import argparse
import difflib
import pandas as pd
import re
from tqdm import tqdm

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                        "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                        "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                        "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                        "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                        "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                        "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                        "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                        "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                        "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                        "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                        "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                        "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                        "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                        "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                        "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                        "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                        "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                        "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                        "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                        "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap    = { 'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
articles     = ['a', 'an', 'the']

periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def process_text(text):
    text = processPunctuation(text)
    text = processDigitArticle(text)
    return text

def get_acc(pred, gts):
    pred = process_text(pred)
    gts = [process_text(gt) for gt in gts]
    same_num = sum([1 if pred == gt else 0 for gt in gts])
    return 100*min(0.3*same_num, 1)

def get_acc_gqa(pred, gts):
    pred = process_text(pred)
    gts = [process_text(gt) for gt in gts]
    same_num = sum([1 if gt in pred else 0 for gt in gts])
    return 100*same_num

def str_simi(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def match_mcq(ans, candidates):
    simis = [str_simi(ans, c) for c in candidates]
    return candidates[simis.index(max(simis))]

def get_index(ans, candidates):
    simis = [str_simi(ans, c) for c in candidates]
    return simis.index(max(simis))

def evaluate_textvqa(datas):
    raw_accs = []
    crop_accs = []
    for data in tqdm(datas):
        raw_answer = data['original_answer']
        crop_answer = data['crop_answer']
        answers = data['labels']

        raw_acc = get_acc(raw_answer, answers)
        crop_acc = get_acc(crop_answer, answers)

        raw_accs.append(raw_acc)
        crop_accs.append(crop_acc)

    return sum(raw_accs) / len(raw_accs), sum(crop_accs) / len(crop_accs)

def evaluate_vstar(datas):
    raw_accs = []
    crop_accs = []

    for data in tqdm(datas):
        candidates = data['question'].split('\n')[1:-1]
        label = data['labels']
        ori = data['original_answer']
        crop = data['crop_answer']

        ori_answer = 'ABCD'[get_index(ori, candidates)]
        crop_answer = 'ABCD'[get_index(crop, candidates)]

        raw_acc = get_acc(ori_answer, label)
        crop_acc = get_acc(crop_answer, label)

        raw_accs.append(raw_acc)
        crop_accs.append(crop_acc)
    
    return sum(raw_accs) / len(raw_accs), sum(crop_accs) / len(crop_accs)


def evaluate_pope(datas):
    ori_acc = []
    crop_acc = []

    for data in tqdm(datas):
        label = data['labels']
        ori = data['original_answer']
        crop = data['crop_answer']

        ori_answer = match_mcq(ori, ['yes', 'no'])
        crop_answer = match_mcq(crop, ['yes', 'no'])

        ori_acc.append(ori_answer == label)
        crop_acc.append(crop_answer == label)

    return 100 * sum(ori_acc) / len(ori_acc), 100 * sum(crop_acc) / len(crop_acc)

def evaluate_aokvqa(datas):
    raw_accs = []
    crop_accs = []
    for data in tqdm(datas):
        raw_answer = data['original_answer']
        crop_answer = data['crop_answer']
        answers = data['labels']

        raw_acc = get_acc(raw_answer, answers)
        crop_acc = get_acc(crop_answer, answers)

        raw_accs.append(raw_acc)
        crop_accs.append(crop_acc)

    return sum(raw_accs) / len(raw_accs), sum(crop_accs) / len(crop_accs)

def evaluate_vqav2(datas):
    raw_accs = []
    crop_accs = []
    for data in tqdm(datas):
        raw_answer = data['original_answer']
        crop_answer = data['crop_answer']
        answers = data['labels']

        raw_acc = get_acc(raw_answer, answers)
        crop_acc = get_acc(crop_answer, answers)

        raw_accs.append(raw_acc)
        crop_accs.append(crop_acc)

    return sum(raw_accs) / len(raw_accs), sum(crop_accs) / len(crop_accs)

def evaluate_gqa(datas):
    raw_accs = []
    crop_accs = []
    for data in tqdm(datas):
        raw_answer = data['original_answer']
        crop_answer = data['crop_answer']
        answers = data['labels']

        raw_acc = get_acc_gqa(raw_answer, answers)
        crop_acc = get_acc_gqa(crop_answer, answers)

        raw_accs.append(raw_acc)
        crop_accs.append(crop_acc)

    return sum(raw_accs) / len(raw_accs), sum(crop_accs) / len(crop_accs)

def evaluate_docvqa(datas):
    raw_accs = []
    crop_accs = []
    for data in tqdm(datas):
        raw_answer = data['original_answer']
        crop_answer = data['crop_answer']
        answers = data['labels']

        raw_acc = 1 if sum([raw_answer in answer or answer in raw_answer for answer in answers]) > 0 else 0
        crop_acc = 1 if sum([crop_answer in answer or answer in crop_answer for answer in answers]) > 0 else 0

        raw_accs.append(raw_acc)
        crop_accs.append(crop_acc)

    return 100 * sum(raw_accs) / len(raw_accs), 100 * sum(crop_accs) / len(crop_accs)

def main(args):
    results = []

    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]

    json_files = [f for f in json_files if sum([task in f for task in args.tasks]) > 0]

    json_files = sorted(json_files)

    for json_file in json_files:
        filepath = os.path.join(args.data_dir, json_file)
        model_name, task, method = json_file.replace('.json', '').split('-')

        with open(filepath, 'r') as f:
            datas = json.load(f)

        if task == 'textvqa':
            raw_acc, crop_acc = evaluate_textvqa(datas)
        elif task == 'vstar':
            raw_acc, crop_acc = evaluate_vstar(datas)
        elif task == 'pope':
            raw_acc, crop_acc = evaluate_pope(datas)
        elif task == 'aokvqa':
            raw_acc, crop_acc = evaluate_aokvqa(datas)
        elif task == 'vqav2':
            raw_acc, crop_acc = evaluate_vqav2(datas)
        elif task == 'gqa':
            raw_acc, crop_acc = evaluate_gqa(datas)
        elif task == 'docvqa':
            raw_acc, crop_acc = evaluate_docvqa(datas)
        else:
            continue

        results.append({
            'model_name': model_name,
            'task': task,
            'method': method,
            'raw_acc': raw_acc,
            'crop_acc': crop_acc
        })


    report_path = os.path.join(args.save_path, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)

    df = pd.DataFrame(results)
    print(df)


    rows = [f"{model}_{method}" for model in args.models for method in args.methods]
    columns = args.tasks

    df1 = pd.DataFrame('', index=rows, columns=columns)

    for data in results:
        model = data['model_name']
        task = data['task']
        method = data['method']
        raw_acc = data['raw_acc']
        crop_acc = data['crop_acc']

        row_nocrop = f"{model}_nocrop"
        if df1.loc[row_nocrop, task] == '':
            df1.loc[row_nocrop, task] = f"{raw_acc:.2f}"

        row = f"{model}_{method}"
        if df1.loc[row, task] == '':
            df1.loc[row, task] = f"{crop_acc:.2f}"

    csv_path = os.path.join(args.save_path, 'evaluation_report.csv')
    df1.to_csv(csv_path, sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing JSON files", default="./data/results")
    parser.add_argument("--save_path", type=str, help="Path to save the evaluation report", default="./")
    args = parser.parse_args()

    args.models = ['llava', 'blip', 'qwen2_5']

    args.methods = ['nocrop', 'rel_att', 'grad_att', 'grad', 'rel_att_high', 'grad_att_high', 'grad_high']

    args.tasks = ['textvqa', 'vstar', 'gqa', 'pope', 'aokvqa', 'docvqa', 'chartqa', 'infoqa']

    main(args)