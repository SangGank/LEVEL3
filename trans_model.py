from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
from tqdm import tqdm
import torch

article_hi = "The melody of this song is catchy and upbeat, with a memorable acoustic guitar riff and a soaring vocal performance that captures the carefree spirit of the 1960s counterculture movement."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Hindi to French
tokenizer.src_lang = "en_XX"
encoded_hi = tokenizer(article_hi, return_tensors="pt").to(device)
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"]
)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

data = pd.read_csv('./process_song_data.csv')
li = []
for i in tqdm(data.caption):
    encoded_hi = tokenizer(i, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"]
    )
    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    li.append(output)
data['trans_caption'] = li
data.to_csv('./trans.csv')