"""
Check available Gemini models with current API
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print("=" * 70)
print("可用的Gemini模型列表")
print("=" * 70)

try:
    models = genai.list_models()

    chat_models = []
    other_models = []

    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            chat_models.append(model)
        else:
            other_models.append(model)

    print(f"\n✅ 支持generateContent的模型（可用于聊天）: {len(chat_models)}个\n")
    for model in chat_models:
        print(f"📌 {model.name}")
        print(f"   - 显示名称: {model.display_name}")
        print(f"   - 支持方法: {', '.join(model.supported_generation_methods)}")
        print()

    print(f"\n其他模型: {len(other_models)}个\n")
    for model in other_models[:5]:  # 只显示前5个
        print(f"   - {model.name}: {', '.join(model.supported_generation_methods)}")

    # 推荐的模型
    print("\n" + "=" * 70)
    print("推荐使用的模型")
    print("=" * 70)

    recommended = [m for m in chat_models if 'gemini' in m.name.lower()]
    if recommended:
        latest = recommended[0]  # 通常第一个是最新的
        print(f"\n🎯 推荐模型: {latest.name}")
        print(f"   在代码中使用: model='{latest.name.replace('models/', '')}'")

except Exception as e:
    print(f"\n❌ 获取模型列表失败: {e}")
    print("\n可能的原因:")
    print("1. API密钥无效")
    print("2. 网络连接问题")
    print("3. API版本不兼容")
