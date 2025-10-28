#!/usr/bin/env python3
"""
预下载维基百科数据脚本
Pre-download Wikipedia data for RAG system
运行此脚本预先下载所有需要的数据，避免在启动应用时下载
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predownload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreloader:
    """数据预加载器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.collector = WikipediaDataCollector(data_dir)
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 预定义的话题和文章数量配置
        self.topic_config = {
            # 核心AI话题 - 更多文章
            "Artificial Intelligence": 5,
            "Machine Learning": 5,
            "Deep Learning": 4,
            "Neural Networks": 3,
            
            # 专门领域
            "Natural Language Processing": 4,
            "Computer Vision": 4,
            "Robotics": 3,
            "Data Science": 3,
            
            # 相关技术
            "Python Programming": 3,
            "Data Mining": 2,
            "Big Data": 2,
            "Cloud Computing": 2,
            
            # 应用领域
            "Autonomous Vehicles": 2,
            "Healthcare AI": 2,
            "Fintech": 2,
            "Cybersecurity": 2
        }
    
    def download_all_data(self, skip_existing: bool = True) -> bool:
        """下载所有数据"""
        try:
            # 检查是否已有数据
            processed_file = os.path.join(self.data_dir, "processed_wikipedia_articles.json")
            if skip_existing and os.path.exists(processed_file):
                with open(processed_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                logger.info(f"发现已有数据: {len(existing_data)} 个文档块")
                
                response = input("发现已有数据，是否跳过下载？(y/n): ").lower()
                if response == 'y':
                    return True
            
            logger.info("开始下载维基百科数据...")
            logger.info(f"将下载 {len(self.topic_config)} 个话题的文章")
            
            # 显示将要下载的内容
            total_articles = sum(self.topic_config.values())
            logger.info(f"预计下载约 {total_articles} 篇文章")
            
            # 分批次收集数据
            all_articles = []
            
            for i, (topic, count) in enumerate(self.topic_config.items(), 1):
                logger.info(f"正在处理话题 {i}/{len(self.topic_config)}: {topic} (目标: {count}篇)")
                
                try:
                    # 搜索并获取文章
                    article_titles = self.collector.search_articles(topic, count + 2)  # 多搜索几个以防失败
                    
                    topic_articles = []
                    for title in article_titles[:count]:  # 只取指定数量
                        article = self.collector.get_article_content(title)
                        if article:
                            article["topic"] = topic
                            topic_articles.append(article)
                            logger.info(f"  ✓ 获取文章: {title}")
                        else:
                            logger.warning(f"  ✗ 无法获取文章: {title}")
                    
                    all_articles.extend(topic_articles)
                    logger.info(f"话题 '{topic}' 完成: 获取了 {len(topic_articles)} 篇文章")
                    
                    # 保存中间进度
                    if len(all_articles) % 10 == 0:
                        self._save_progress(all_articles, "temp_articles.json")
                    
                except Exception as e:
                    logger.error(f"处理话题 '{topic}' 时出错: {e}")
                    continue
            
            logger.info(f"所有文章收集完成！总共获取了 {len(all_articles)} 篇文章")
            
            # 预处理内容
            logger.info("正在预处理文章内容...")
            processed_articles = self.collector.preprocess_content(all_articles)
            logger.info(f"处理完成！生成了 {len(processed_articles)} 个文档块")
            
            # 保存处理后的数据
            final_file = self.collector.save_articles(processed_articles, "processed_wikipedia_articles.json")
            
            # 生成数据统计
            self._generate_data_stats(processed_articles)
            
            # 清理临时文件
            temp_file = os.path.join(self.data_dir, "temp_articles.json")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            logger.info("🎉 数据预下载完成！")
            logger.info(f"数据文件: {final_file}")
            logger.info("现在可以启动应用而无需等待数据下载")
            
            return True
            
        except Exception as e:
            logger.error(f"预下载过程中出现错误: {e}")
            return False
    
    def _save_progress(self, articles: List[Dict[str, Any]], filename: str):
        """保存中间进度"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"进度已保存: {len(articles)} 篇文章")
    
    def _generate_data_stats(self, processed_articles: List[Dict[str, Any]]):
        """生成数据统计信息"""
        stats = {
            "total_chunks": len(processed_articles),
            "unique_articles": len(set(article["title"] for article in processed_articles)),
            "topics": {},
            "generated_at": datetime.now().isoformat()
        }
        
        # 统计每个话题的文章数
        for article in processed_articles:
            topic = article.get("topic", "unknown")
            if topic not in stats["topics"]:
                stats["topics"][topic] = {"chunks": 0, "articles": set()}
            stats["topics"][topic]["chunks"] += 1
            stats["topics"][topic]["articles"].add(article["title"])
        
        # 转换set为list以便JSON序列化
        for topic in stats["topics"]:
            stats["topics"][topic]["unique_articles"] = len(stats["topics"][topic]["articles"])
            stats["topics"][topic]["articles"] = list(stats["topics"][topic]["articles"])
        
        # 保存统计信息
        stats_file = os.path.join(self.data_dir, "data_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("=== 数据统计 ===")
        logger.info(f"文档块总数: {stats['total_chunks']}")
        logger.info(f"独特文章数: {stats['unique_articles']}")
        logger.info("各话题分布:")
        for topic, info in stats["topics"].items():
            logger.info(f"  {topic}: {info['unique_articles']} 篇文章, {info['chunks']} 个块")
    
    def check_data_status(self):
        """检查当前数据状态"""
        logger.info("=== 数据状态检查 ===")
        
        # 检查原始数据
        processed_file = os.path.join(self.data_dir, "processed_wikipedia_articles.json")
        if os.path.exists(processed_file):
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ 已有处理后的数据: {len(data)} 个文档块")
        else:
            logger.info("❌ 没有找到处理后的数据")
        
        # 检查统计文件
        stats_file = os.path.join(self.data_dir, "data_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            logger.info(f"📊 数据统计可用 (生成于: {stats['generated_at']})")
        else:
            logger.info("📊 没有数据统计文件")
        
        # 检查向量存储
        vector_store_dir = "vector_store"
        if os.path.exists(os.path.join(vector_store_dir, "faiss_index.index")):
            logger.info("✅ 向量存储已存在")
        else:
            logger.info("⚠️ 向量存储尚未创建")

def main():
    """主函数"""
    print("🚀 维基百科数据预下载工具")
    print("=" * 50)
    
    preloader = DataPreloader()
    
    # 显示菜单
    while True:
        print("\n请选择操作:")
        print("1. 检查数据状态")
        print("2. 下载所有数据")
        print("3. 强制重新下载")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            preloader.check_data_status()
            
        elif choice == "2":
            success = preloader.download_all_data(skip_existing=True)
            if success:
                print("\n✅ 数据下载完成！现在可以运行: streamlit run app_gemini.py")
            else:
                print("\n❌ 数据下载失败，请检查网络连接和错误日志")
                
        elif choice == "3":
            print("⚠️ 这将删除现有数据并重新下载")
            confirm = input("确认继续？(y/n): ").lower()
            if confirm == 'y':
                success = preloader.download_all_data(skip_existing=False)
                if success:
                    print("\n✅ 数据重新下载完成！")
                else:
                    print("\n❌ 数据下载失败")
                    
        elif choice == "4":
            print("👋 再见！")
            break
            
        else:
            print("❌ 无效选择，请输入 1-4")

if __name__ == "__main__":
    main()