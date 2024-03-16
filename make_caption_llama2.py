from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
import pickle
import pandas as pd
from tqdm import tqdm

# data = pd.read_csv('./song_data.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

access_token ='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'
 
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)
language_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=access_token).to(device)
 
text_generation_pipeline = TextGenerationPipeline(
    model=language_model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=device,
    
)

def generate_text(prompt, max_length=2048):
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


def make_moode(mood, tempo, genre):
    base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
    text = f"Make 50 sentence. the genre of the song is 'f{genre}', the tempo of the song is 'f{tempo}', the mood of the song is 'f{mood}'. As for the form of the result, put 50 sentences in the list"
    system_text = "Please print out 50 sentences that make the genre of the song I designated, the tempo of the song, and the mood of the song."
    input = base_prompt.format(system_prompt = system_text,
                               user_prompt = text)
    recommendations = generate_text(input)

    return recommendations

label_data_path ='./labels.pkl'
with open(label_data_path,'rb') as f:
    emotion_labels=pickle.load(f)
    tempo_labels=pickle.load(f)
    genre_labels=pickle.load(f)


lis = []
columns = ['genre','emotion','tempo(category)','texts']
for genre in tqdm(genre_labels[:2]):
    for emotion in emotion_labels[:2]:
        for tempo in tempo_labels[:2]:
            result = make_moode(emotion, tempo, genre)
            lis.append([genre,emotion,tempo, result])
data = pd.DataFrame(lis,columns = columns)
data.to_csv('./make_caption.csv',index=False)
