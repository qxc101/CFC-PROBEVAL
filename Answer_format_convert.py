import json
import argparse
from pprint import pprint
import fasttext
import fasttext.util

def CFC2ProtoQA(input_path_prediction, input_path_embeddings, output_path):
    # Step 1: Load Data
    with open(input_path_prediction, 'r') as file:
        prediction_data = json.load(file)
    
    with open(input_path_embeddings, 'r') as file:
        embedding_data = json.load(file)
    
    # Step 2: Extract and Organize Data
    output_data = []
    
    for question_id, item in embedding_data.items():
        # Extract ranked answers from the prediction data
        ranked_answers = prediction_data["model_ranked_answers"].get(question_id, [])
        
        # Combine ranked answers into a single item
        data_item = {
            question_id: ranked_answers
        }
        
        output_data.append(data_item)

    # Step 3: Save Data
    with open(output_path, 'w') as outfile:
        for item in output_data:
            json.dump(item, outfile)
            outfile.write('\n')

def average_embedding(phrase, model):
    words = phrase.split()
    print(phrase)
    embeddings = [model[word] for word in words]
    try:
        avg_embedding = sum(embeddings) / len(embeddings)
    except:
        print(phrase)
        print(avg_embedding)
    return avg_embedding


def ProtoQA2CFC(input_path, output_path_prediction, output_path_embeddings, reference_file, question2id=False):
    # Step 0: Load your FastText model (replace 'path/to/model' with the actual path to your model file)
    ft_model = fasttext.load_model('cc.en.300.bin')
    ft_model = fasttext.util.reduce_model(ft_model, 100)

    # Step 1: Load Data
    with open(input_path, 'r') as file:
        data = [json.loads(line) for line in file]

    # Step 2: Calculate Counts & Step 3: Rank Answers
    model_answers = {}
    model_ranked_answers = {}
    model_answers_with_count = {}
    x = 0
    for item in data:
        if question2id:
            qid = "r4q" + str(x + 1)
        for question_id, answers in item.items():
            if question2id:
                question_id = qid
            answers = [answer.lower().strip() for answer in answers if answer.strip() != ""]
            unique_answers = list(set(answers))  # Get unique answers
            answer_counts = [[answer, answers.count(answer)] for answer in unique_answers]  # Get counts
            ranked_answers = [answer for answer, count in
                              sorted(answer_counts, key=lambda x: x[1], reverse=True)]  # Get ranked answers

            model_answers[question_id] = unique_answers
            model_ranked_answers[question_id] = ranked_answers
            model_answers_with_count[question_id] = sorted(answer_counts, key=lambda x: x[1], reverse=True)
        x += 1

    # Step 4: Format Data
    output_data = {
        "model_answers": model_answers,
        "model_ranked_answers": model_ranked_answers,
        "model_answers_with_count": model_answers_with_count,
    }

    # Step 5: Save Data
    with open(output_path_prediction, 'w') as outfile:
        json.dump(output_data, outfile)

    # with open(output_path_prediction, 'w') as outfile:
    #     json.dump(output_data, outfile, ensure_ascii=False, indent=4)
    # Step 6: Load Reference Data (replace 'path/to/reference_file.jsonl' with the actual path to your reference file)
    reference_data = {}
    with open(reference_file, 'r') as ref_file:
        for line in ref_file:
            data = json.loads(line.strip())
            reference_data.update(data)
            
    
    output_data_embeddings = {}
    for question_id, item in reference_data.items():
        print(question_id)
        # Map true_answers, raw_answers, etc. directly from the reference data
        # print("--------------------------------------")
        # print(item['question_string'])
        # print(item['true_answers'])
        # print(model_answers.get(question_id, []),)
        # print(item['pred_answers'])
        output_data_embeddings[question_id] = {
            'true_answers': item['true_answers'],
            'raw_answers': item['raw_answers'],
            'question_string': item['question_string'],
            'true_answer_vectors': {answer: average_embedding(answer, ft_model).tolist() for answer in item['true_answers']},
            
            # Get predicted answers from the model answers we computed earlier
            'pred_answers': model_answers.get(question_id, []),
            
            # Calculate pred_answer_vectors using FastText model
            'pred_answer_vectors': {answer: average_embedding(answer, ft_model).tolist() for answer in model_answers.get(question_id, [])}
        }

    # Step 7: Save Data (Embeddings)
    # with open(output_path_embeddings, 'w') as outfile:
    #     for question_id, data in output_data_embeddings.items():
    #         json.dump({question_id: data}, outfile)
    #         outfile.write('\n')
    # Step 7: Save Data (Embeddings)
    with open(output_path_embeddings, 'w') as outfile:
        json.dump(output_data_embeddings, outfile, indent=4)


def CFC2ProtoQA(input_path_prediction, input_path_embeddings, output_path):
    # Step 1: Load Data
    with open(input_path_prediction, 'r') as file:
        prediction_data = json.load(file)
    
    with open(input_path_embeddings, 'r') as file:
        embedding_data = json.load(file)
    
    # Step 2: Extract and Organize Data
    output_data = []
    
    for question_id, item in embedding_data.items():
        # Extract ranked answers from the prediction data
        ranked_answers = prediction_data["model_ranked_answers"].get(question_id, [])
        
        # Combine ranked answers into a single item
        data_item = {
            question_id: ranked_answers
        }
        
        output_data.append(data_item)

    # Step 3: Save Data
    with open(output_path, 'w') as outfile:
        for item in output_data:
            json.dump(item, outfile)
            outfile.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple script.")
    parser.add_argument('-pf', '--protoQA_file', help='path containing ground truth answer clusters',
                        default="./ProtoQA_evaluator_data/original_q_gpt-4.jsonl")
    parser.add_argument('-cfp', '--cfc_prediction', help='Name of gpt filter model',
                        default="./CFC_evaluator_data/Other_Predictions/0915modelgpt4_prediction.json")
    parser.add_argument('-cfe', '--cfc_embedding', help='Name of gpt filter model',
                        default="./CFC_evaluator_data/Other_Predictions/0915modelgpt4_embeddings.jsonl")
    args = parser.parse_args()
    
    with open("CFC_evaluator_data/GPT2_Predictions/0715modelft_grad8lr5e-06sample200_temp1.0_embeddings_100.jsonl", 'r') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]

    ProtoQA2CFC(args.protoQA_file, args.cfc_prediction, args.cfc_embedding, "/home/qic69/Desktop/CFC/CFC_evaluator_data/GPT2_Predictions/0715modelft_grad8lr5e-06sample200_temp1.0_embeddings_100.jsonl", question2id=True)
    # CFC2ProtoQA(args.cfc_prediction, args.cfc_embedding, args.protoQA_file)
