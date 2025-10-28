# 快速开始指南 - RAG系统

## 🚀 推荐使用方式 (避免下载缓慢)

### 方法一：预下载数据 (推荐)
```bash
# 1. 预下载所有维基百科数据 
python predownload_data.py

# 2. 启动应用
streamlit run app_gemini.py
```

### 方法二：在线下载 (较慢)
```bash
# 直接启动应用，在界面中下载数据
streamlit run app_gemini.py
```

## 📋 预下载脚本功能

运行 `python predownload_data.py` 后会看到菜单：

1. **检查数据状态** - 查看当前本地数据情况
2. **下载所有数据** - 完整下载所有话题的文章 
3. **强制重新下载** - 清除现有数据重新下载
4. **退出**

## 📊 数据覆盖范围

预下载脚本会下载以下话题的文章：

### 核心AI话题
- Artificial Intelligence (5篇)
- Machine Learning (5篇) 
- Deep Learning (4篇)
- Neural Networks (3篇)

### 专门领域
- Natural Language Processing (4篇)
- Computer Vision (4篇)
- Robotics (3篇)
- Data Science (3篇)

### 相关技术
- Python Programming (3篇)
- Data Mining (2篇)
- Big Data (2篇)
- Cloud Computing (2篇)

### 应用领域
- Autonomous Vehicles (2篇)
- Healthcare AI (2篇)
- Fintech (2篇)
- Cybersecurity (2篇)

**总计**: 约40篇文章，处理后生成数百个文档块

## 🔧 环境要求

1. 创建 `.env` 文件并添加：
```
GOOGLE_API_KEY=your_api_key_here
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 💡 优势对比

| 方式 | 启动时间 | 数据量 | 网络依赖 |
|------|---------|--------|----------|
| 预下载 | 快速 (~10秒) | 丰富 (~40篇) | 仅预下载时 |
| 在线下载 | 慢 (2-5分钟) | 少量 (~6篇) | 每次启动 |

## 🗂️ 生成的文件

预下载完成后会生成：
- `data/processed_wikipedia_articles.json` - 处理后的文档数据
- `data/data_stats.json` - 数据统计信息
- `predownload.log` - 下载日志

## ❓ 常见问题

### Q: 预下载失败怎么办？
A: 检查网络连接，查看 `predownload.log` 日志文件

### Q: 如何更新数据？
A: 运行预下载脚本选择"强制重新下载"

### Q: 数据存储在哪里？
A: 所有数据存储在 `data/` 目录下

## 🚨 注意事项

1. 预下载过程需要稳定的网络连接
2. 首次下载可能需要5-10分钟
3. 生成的数据文件约几十MB
4. 建议在网络环境良好时进行预下载