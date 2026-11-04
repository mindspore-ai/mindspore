from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from tqdm import tqdm
import time
import os
import json
import argparse

class GPTBatcher:
    """
    Borrow from https://github.com/fengsxy/gpt_batch

    Parameters:
        api_key (str): API key for authenticating requests to the OpenAI API.
        model_name (str, optional): Specifies the GPT model version. Default is 'gpt-3.5-turbo-0125'.
        system_prompt (str, optional): Initial text or question to seed the model with. Default is empty.
        temperature (float, optional): Sets the creativity of the responses. Default is 1.
        num_workers (int, optional): Number of parallel workers for request handling. Default is 64.
        timeout_duration (int, optional): Timeout for API responses in seconds. Default is 60.
        retry_attempts (int, optional): How many times to retry a failed request. Default is 2.
    """

    def __init__(self, 
                 api_key, 
                 model_name="gpt-3.5-turbo", 
                 system_prompt="",
                 temperature=0, 
                 num_workers=32,
                 timeout_duration=60,
                 retry_attempts=2,
                 api_base_url=None):
        
        self.client = OpenAI(api_key=api_key, base_url = api_base_url)
        self.model_name = model_name
        self.system_prompt = "You are an impartial judge tasked with evaluating the quality of predicted text provided by autonomous driving AI assistant. You will compare this prediction text to a reference text, focusing on the description of objects that influence the driving behavior of ego car, and the explanation of why these objects impact. Your evaluation criteria should include accuracy(checking if the predicted text correctly identifies objects mentioned the reference text), suppression hallucination(ensuring that objects not mentioned in the reference text are not erroneously included in the predicted text), correlation(sessing if the reasons for the objects' impact on the ego car's driving behavior are consistent between the reference and predicted text). Be as objective as possible. Do not allow the length of the predicted text to influence your evaluation. Maximize your text comprehension capabilities to freely match objects with high similarity, appropriately ignoring the relative positions and color attributes of the objects. After providing your short explanation, you must rate the response on a scale from 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[10]]\"."
        self.temperature = temperature
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index =[]
        if api_base_url:
            self.client.base_url = api_base_url


    def create_messages(self, message):
        ret = []
        # system prompt
        ret.append({
            "role": "system",
            "content": self.system_prompt
        })

        # few shot example
        few_shot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_few_shot")
        with open(os.path.join(few_shot, "high.json")) as f:
            high_data = json.load(f)
        with open(os.path.join(few_shot, "low.json")) as f:
            low_data = json.load(f)

        template = "[The Start of Reference Text]\n{}\n[The End of Reference Text]\n\n[The Start of Prediction Text]\n{}\n[The End of Prediction Text]"

        # high example
        ret.append({
            "role": "user", 
            "content": template.format(high_data["reference"], high_data["prediction"])
        })
        ret.append({
            "role": "assistant", 
            "content": high_data["response"]
        })

        # low example
        ret.append({
            "role": "user", 
            "content": template.format(low_data["reference"], low_data["prediction"])
        })
        ret.append({
            "role": "assistant", 
            "content": low_data["response"]
        })

        ret.append({
            "role": "user", 
            "content": template.format(message["reference"], message["prediction"])
        })
        return ret

    def get_attitude(self, ask_text):
        index, ask_text = ask_text
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=ask_text,
                temperature=self.temperature,
            )
            return (index, completion.choices[0].message.content)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.miss_index.append(index)
            return (index, None)

    def process_attitude(self, message_list):
        new_list = []
        num_workers = self.num_workers
        timeout_duration = self.timeout_duration
        retry_attempts = self.retry_attempts
    
        executor = ThreadPoolExecutor(max_workers=num_workers)
        message_chunks = list(self.chunk_list(message_list, num_workers))
        try:
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {executor.submit(self.get_attitude, message): message for message in chunk}
                for _ in range(retry_attempts):
                    done, not_done = wait(future_to_message.keys(), timeout=timeout_duration)
                    for future in not_done:
                        future.cancel()
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    future_to_message = {executor.submit(self.get_attitude, future_to_message[future]): future for future in not_done}
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            executor.shutdown(wait=False)
            return new_list

    def complete_attitude_list(self,attitude_list, max_length):
        completed_list = []
        current_index = 0
        for item in attitude_list:
            index, value = item
            # Fill in missing indices
            while current_index < index:
                completed_list.append((current_index, None))
                current_index += 1
            # Add the current element from the list
            completed_list.append(item)
            current_index = index + 1
        while current_index < max_length:
            print("Filling in missing index", current_index)
            self.miss_index.append(current_index)
            completed_list.append((current_index, None))
            current_index += 1
        return completed_list

    def chunk_list(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def handle_message_list(self,message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def get_miss_index(self):
        return self.miss_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, default="ann")
    parser.add_argument("--prediction_path", type=str, default="prediction/xx.jsonl")
    parser.add_argument("--save_path", type=str, default="eval_res/xx")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--retry_attempts", type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    json_list = sorted([each for each in os.listdir(args.reference_path) if each.endswith(".json")])
    answers = [json.loads(q) for q in open(os.path.expanduser(args.prediction_path), "r", encoding='utf-8')]
    answers.sort(key=lambda element: element['question_id'])
    assert len(json_list) == len(answers)
    batcher = GPTBatcher(
        api_key=args.api_key, 
        model_name=args.model_name, 
        num_workers=args.num_workers,
        retry_attempts=args.retry_attempts,
        api_base_url=args.api_base_url)
    
    all_score = []
    rets = []
    for idx, json_name in tqdm(enumerate(json_list)):
        assert idx == int(answers[idx]['question_id'])
        message= dict()
        message["prediction"] = answers[idx]['answer']
        
        with open(os.path.join(args.reference_path, json_name), "r", encoding='utf-8') as f:
            data = json.load(f)
        general_data = data["general_perception"]
        message["reference"] = general_data['description and explanation']
        ret = batcher.create_messages(message)
        rets.append(ret)
        
    results = batcher.handle_message_list(rets)
    for idx, json_name in tqdm(enumerate(json_list)):
        output = results[idx]
        txt_name = json_name.replace(".json", ".txt")
        if output == None:
            continue
            print(f"Missing {json_name} output")
            
        try:
            all_score.append(int(output.split("Rating: [[")[1].split("]]")[0]))
        except:
            try:
                all_score.append(int(output.split("rating is: [[")[1].split("]]")[0]))
            except:
                try:
                    all_score.append(int(output.split("[[")[1].split("]]")[0]))
                except:
                    print(f"Missing extract score from {txt_name}")    
        with open(os.path.join(args.save_path, txt_name), "w", encoding='utf-8') as f:
            f.write(output)
    
    # cal score
    print(f"Stage1_score: {sum(all_score)/len(all_score)}")
    with open(os.path.join(args.save_path, "all_score.txt"), "w", encoding='utf-8') as f:
        f.write(f"Stage1_score: {sum(all_score)/len(all_score)}")