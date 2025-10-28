#!/usr/bin/env python3
"""
自动下载数据脚本
Auto download Wikipedia data without user input
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector import WikipediaDataCollector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """自动下载数据"""
    print("🚀 开始自动下载维基百科数据...")
    
    # 检查是否已有数据
    articles_file = "data/processed_wikipedia_articles.json"
    if os.path.exists(articles_file):
        with open(articles_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"✅ 发现已有数据: {len(existing_data)} 个文档块")
        print("如需重新下载，请删除 data/ 目录")
        return
    
    # 创建数据收集器
    collector = WikipediaDataCollector("data")
    
    # 定义话题
    topics = [
        "Artificial Intelligence",
        "Machine Learning", 
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Robotics"
    ]
    
    try:
        logger.info("开始收集维基百科文章...")
        
        # 收集文章
        articles = collector.collect_articles_by_topics(topics, articles_per_topic=3)
        logger.info(f"收集了 {len(articles)} 篇文章")
        
        # 预处理
        processed_articles = collector.preprocess_content(articles)
        logger.info(f"生成了 {len(processed_articles)} 个文档块")
        
        # 保存
        saved_file = collector.save_articles(processed_articles, "processed_wikipedia_articles.json")
        
        print(f"✅ 数据下载完成！")
        print(f"📁 文件位置: {saved_file}")
        print(f"📊 总计: {len(processed_articles)} 个文档块")
        print("🚀 现在可以运行: streamlit run app_gemini.py")
        
    except Exception as e:
        logger.error(f"下载失败: {e}")
        print(f"❌ 下载失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()