import ChatTTS
import torch
import time
import torchaudio

SET_SPK = True

###################################
# Sample a speaker from Gaussian.
chat = ChatTTS.Chat()
chat.load(compile=False)  # Set to True for better performance

if SET_SPK:
    rand_spk = torch.load("spk/spk_man2.pth")
else:
    rand_spk = chat.sample_random_speaker()
    print(rand_spk)  # save it for later timbre recovery
    torch.save(rand_spk, "spk/spk_man1.pth")

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=0.00000001,  # using custom temperatureSS
    top_P=0.7,  # top P decode
    top_K=20,  # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7)
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt="[oral_9][laugh_2][break_1][speed_5]",
)

###################################
# For word level manual control.

start = time.time()
# text = 'What is [uv_break]your favorite english food?[lbreak][laugh][laugh]'
# text = '今天我們來討論人工智慧技術的發展及其應用。'
# text = '今天天氣真好，我們出去散步巴。'
text = "我個人認為，義大利麵就應該拌四十二號混泥土，因為這個螺絲釘的長度很容易直接影響到挖掘機的扭矩你往裡砸的時候，一瞬間他就會產生大量的高能蛋白，俗稱 UFO，會嚴重影響經濟的發展，以至於對整個太平洋和充電器的核污染再或者說，透過這勾股定理，很容易推斷出人工飼養的東條英機，他是可以捕獲野生的三角函數所以說，不管這秦始皇的切面是否具有放射性，川普的 N 次方是否有沉澱物，都不會影響到沃爾瑪跟維爾康在南極匯合"
wavs = chat.infer(
    text,
    skip_refine_text=True,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)
end = time.time()
print("inference time:", end - start)
"""
In some versions of torchaudio, the first line works but in other versions, so does the second line.
"""
try:
    torchaudio.save(
        "word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000
    )
except:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)
