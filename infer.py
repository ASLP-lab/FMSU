import os
import sys
import json
import torch
import warnings

# ========== 环境设置 ==========
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7" #参考qwen3omni

warnings.filterwarnings('ignore')

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams

# ========== 模型加载函数 ==========
def load_model_processor(model_path):
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个 GPU，设置 tensor_parallel_size 为 {num_gpus}")

    model = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=num_gpus,
        max_num_seqs=4,
        max_model_len=32768,
        seed=1234,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor

# ========== 单条音频推理函数 ==========
def single_inference(model, processor, audio_path):
    # 构造 Prompt
    prompt_text = (
        "对这段音频进行多维度声学属性分析，以json格式输出text_and_paralanguage（带副语言标签的文本转录），"
        "language（语言），background_sound（背景音），environment（声学环境），gender（性别），age（年龄），"
        "pitch（音高），speed（语速），emotion（情绪），emotion_level（情绪强度），accent（口音），"
        "tone（语气），rhythm（节奏/韵律），texture（音质），pronunciation（发音），"
        "paralinguistic（副语言事件），contextual_inference（语境推理）和caption（音频的综合摘要）。"
    )

    # 构造模型消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # 预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios_data, images_data, videos_data = process_mm_info(messages, use_audio_in_video=True)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {"use_audio_in_video": True}
    }

    if audios_data is not None:
        inputs['multi_modal_data']['audio'] = audios_data
    if images_data is not None:
        inputs['multi_modal_data']['image'] = images_data
    if videos_data is not None:
        inputs['multi_modal_data']['video'] = videos_data

    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.01, top_p=0.1, top_k=1, max_tokens=2048)

    # 执行推理
    outputs = model.generate(inputs, sampling_params=sampling_params)
    response = outputs[0].outputs[0].text

    return response

# ========== 主入口 ==========
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # ===== 修改为你的模型路径和音频路径 =====
    MODEL_PATH = "xxxx" #模型路径
    AUDIO_PATH = "xxx.wav"   # 请替换为实际音频路径

    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(AUDIO_PATH):
        print(f"❌ 音频文件不存在: {AUDIO_PATH}")
        sys.exit(1)

    print("🚀 正在加载模型...")
    model, processor = load_model_processor(MODEL_PATH)

    print(f"🎤 正在对音频进行推理: {AUDIO_PATH}")
    response = single_inference(model, processor, AUDIO_PATH)

    print("\n" + "="*50)
    print("📝 模型输出:")
    print(response)
    print("="*50)

    # 可选：尝试将输出解析为 JSON 并美化打印
    try:
        parsed = json.loads(response)
        print("\n✅ 解析后的 JSON 内容:")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("\n⚠️ 模型输出并非合法 JSON，以上为原始文本。")
