from googletrans import Translator
import pandas as pd
from tqdm import tqdm
import time

# GOOGLE_TIME_TO_SLEEP = 1.5

# data = pd.read_csv('./process_song_data.csv')
# translator = Translator()

# li = []
# i=0
# for caption in tqdm(data['caption']):
#     i+=1
#     li.append(caption)
#     trans = translator.translate(caption, src='en', dest='ko')
#     li.append(trans.text)
#     time.sleep(GOOGLE_TIME_TO_SLEEP)
#     if i ==3:
#         break
# print(li)


# from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
# import torch

# import pandas as pd

# access_token ='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'
 
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
 
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)
# language_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=access_token)
 
# text_generation_pipeline = TextGenerationPipeline(
#     model=language_model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device=0,
# )

# def generate_text(prompt, max_length=400):
#     generated_sequences = text_generation_pipeline(
#         prompt,
#         do_sample=True,
#         top_k=10,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         max_length=max_length,
#         truncation= True,
#     )
 
#     return generated_sequences[0]["generated_text"].replace(prompt, "")

# data = pd.read_csv('./process_song_data.csv')

# li = []
# i=0
# for caption in tqdm(data['caption']):
#     base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
#     input = base_prompt.format(system_prompt = "Translate this into Korean. Answer format is 'answer : your response'",
#                                 user_prompt = caption)
#     text = f"'{input}' Translate this sentence into Korean"
#     recommendations = generate_text(text)
#     li.append(recommendations)
#     i+=1
#     if i ==3:
#         break
# print(li)