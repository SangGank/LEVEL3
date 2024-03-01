from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

import pandas as pd
data = pd.read_csv('./song_data.csv')

access_token ='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'
 
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)
language_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=access_token)
 
text_generation_pipeline = TextGenerationPipeline(
    model=language_model,
    tokenizer=tokenizer,
    # torch_dtype=torch.float16,
    device=0,
    torch_dtype=torch.float16,
    device_map="auto",
)

def generate_text(prompt, max_length=400):
    generated_sequences = text_generation_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        truncation= True,
    )
 
    return generated_sequences[0]["generated_text"].replace(prompt, "")

# base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
# text = f"Songs you want to listen to on a rainy day."
# input = base_prompt.format(system_prompt = " I'm going to create music that matches the sentence.Recommend the music genre, mood, tempo, and multiple instruments to be included in the song. The format of the answer should be 'genre : genre, mood : mood, tempo : tempo, instrument :  [inst1, inst2, inst3, ...]'.",
#                             user_prompt = text)
base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
text = f"Write one sentence describing about the melody of the song 'Bohemian Rhapsody' by 'Queen'"
input = base_prompt.format(system_prompt = "The format of the answer should be 'song_name : your response'. your response is noun form.",
                            user_prompt = text)
recommendations = generate_text(input)
print(recommendations)