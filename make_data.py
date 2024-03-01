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
    torch_dtype=torch.float16,
    device=0,
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

def make_moode(album, song_name):
    base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
    text = f"Please tell me the emotion of the song '{song_name}' by '{album}'."
    input = base_prompt.format(system_prompt = "Respond with one of the following 13 emotions: ['Happiness', 'Sadness', 'Excitement', 'Calmness', 'Nostalgia', 'Fear', 'Anger', 'Anticipation', 'Moved', 'Love', 'Loneliness', 'Nostalgia', 'Gratitude']. The format of the answer should be 'answer: emotion'.",
                               user_prompt = text)
    recommendations = generate_text(input)

    return recommendations

def make_caption(album, song_name):
    base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
    text = f"Write one sentence describing about the melody of the song '{song_name}' by '{album}'."
    input = base_prompt.format(system_prompt = "The format of the answer should be 'song_name : your response'. your response is noun form.",
                               user_prompt = text)
    recommendations = generate_text(input)
    return recommendations


data['index'] =data.index
data['emotion'] = data['index'].apply(lambda x: make_moode(data.album.loc[x],data.song_name.loc[x]))
data.to_csv('add_emotion.csv',index=False)
print('emotion is finish!!')
data['caption'] = data['index'].apply(lambda x: make_caption(data.album.loc[x],data.song_name.loc[x]))
data.to_csv('add_caption.csv',index=False)
print('caption is finish!!')




