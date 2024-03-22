from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
from tqdm import tqdm
import pandas as pd

data = pd.read_csv('./genre_11_tempo_4_no_world.csv')

access_token ='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'
 
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)
language_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=access_token)
 
text_generation_pipeline = TextGenerationPipeline(
    model=language_model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=device,
)


def make_when(singer, song_name):
    max_length = 400
    conversation = [ {'role': 'system', 'content': "Describe the situation in very detail in one sentence. Just explain situation. Sentence format must be [Result : {your sentence}]."},
                    {'role': 'user', 'content': f" Who do you want to listen to '{song_name}' by {singer} with?"} ] 
    
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
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

caption = []

data2=data.copy()
for i in tqdm(range(len(data2))):
    result = make_when(data2.iloc[i]['singer'],data2.iloc[i]['song_name'])
    caption.append(result.split(':')[-1])
    # caption.append(result)
data2['caption'] = caption

data2.to_csv('who_with_song_listen.csv',index=False)
