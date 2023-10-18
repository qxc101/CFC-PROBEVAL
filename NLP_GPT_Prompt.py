import random
import openai
import json
import argparse
from llama_cpp import Llama
# openai.api_key = "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"
openai.api_key = "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"

def answer_llama2(role, user_prompt, model="llama-2-7b"):
    if model == "llama-2-7b":
        LLM = Llama(model_path="/home/qic69/Desktop/CFC/llama-2-7b-chat.ggmlv3.q8_0.bin", n_ctx=2048)
    else:
        LLM = Llama(model_path="/home/qic69/Desktop/CFC/llama-2-7b-chat.ggmlv3.q8_0.bin", n_ctx=2048)
    prompt = role + "Q: " + user_prompt + " A:"
    output = LLM(prompt)
    return output["choices"][0]["text"]

def answer_gpt(role, user_prompt, temperature=None, model="gpt-4"):
    print(user_prompt)
    if temperature != None:
        response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
    else:
        response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )

    message = response['choices'][0]['message']['content']
    # print(message)
    return message


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple script.")
    parser.add_argument('-m', '--model', help='Name of LLM. Can be any GPT models, or llama-2-7b', default="gpt-3.5-turbo")
    parser.add_argument('-i', '--input_file', help='Name of input file', default="/home/qic69/Desktop/CFC/ProtoQA_evaluator_data/original_q.txt")
    args = parser.parse_args()

    role1 = "You are a human-like AI that elaborates a situation based on a context. "
    role2 = "You are a human-like AI that answers the context questions with single word or short phrase. "
    # prompt1 = ("For the given context sentence, extend it to describe a real life situation. Make the description general. Add information as you see fit. Make it under 100 words.")
    prompt1 = ("For the given context sentence, extend it to include more information. Make the extension general (not including unnecessary detail). Add information as you see fit. Make it under 100 words.")
    prompt2 = ("Answer the following questions given the context description.")

    with open(args.input_file, 'r') as file:
        questions = file.readlines()
    questions = [line.strip() for line in questions]
    N_SAMPLES = 1
    # TODO Find best temperature on dev set
    BEST_TEMPERATURE = 0.5
    
    predictions = {}
    random.shuffle(questions)
    for idx, q in enumerate(questions):
        combined_answers = []
        for _ in range(N_SAMPLES):
            if args.model == "llama-2-7b":
                answers = answer_llama2(role, prompt + q.strip(), model= args.model).split(';')
            else:
                print("****************************")
                print("CQ Pair: ", q)
                print("----------Original Answer-------------")
                t = answer_gpt(role2, prompt2 + "\nSituation: " + q.split(". ")[0].strip() + "\nQuestion: " + q.split(". ")[1].strip() + "\nAnswer: ", BEST_TEMPERATURE, model= args.model).split(';')
                print(t)
                print("----------Round 1 -------------")
                answers = answer_gpt(role1, prompt1 + "\nContext sentence: " + q.split(". ")[0].strip(), model= args.model)
                print(answers)
                print("----------Round 2 -------------")
                answers = answer_gpt(role2, prompt2 + "\nSituation: " + q.split(". ")[0].strip() + "\nContext:" + answers + "\nQuestion: " + q.split(". ")[1].strip() + "\nAnswer: ", BEST_TEMPERATURE, model= args.model).split(';')
                print(answers)
            combined_answers.extend(answers)

        question_id = q
        predictions[question_id] = combined_answers
    
    # Save results to predictions.jsonl
    with open(args.input_file.replace(".txt", "NLP_" + args.model + '.jsonl'), 'w') as outfile:
        for key, value in predictions.items():
            outfile.write(json.dumps({key: value}) + '\n')



