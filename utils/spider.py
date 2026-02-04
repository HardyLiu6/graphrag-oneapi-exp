"""
网页爬虫及 Markdown 转换模块

该模块使用 Scrapy 框架爬取网页内容，并将其转换为 Markdown 格式。
专注于提取页面主要内容，忽略导航栏、侧边栏、页脚等元素。

日期: 2026-01-27
作者: LiuJunDa

使用方法:
    python utils/spider.py --url <起始URL> --domain <允许的域名> --output <输出目录>

示例:
    python utils/spider.py --url https://example.com --domain example.com --output ./crawled_data
"""

import argparse
import os
import json
import logging
from urllib.parse import urlparse
from pathlib import Path

try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    from bs4 import BeautifulSoup
    import html2text
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请运行: pip install scrapy beautifulsoup4 html2text")
    exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ContentFocusedSpider(scrapy.Spider):
    """
    自定义的 Scrapy 爬虫类，继承自 Scrapy 的 Spider 基类
    
    功能: 爬取网页并将内容转换为 Markdown 格式
    
    属性:
        start_urls: 从命令行参数或默认值获取
        allowed_domains: 从命令行参数或默认值获取
        output_dir: 输出目录路径
    """
    name = 'content_focused_spider'

    def __init__(self, start_url=None, allowed_domain=None, output_dir=None, 
                 max_pages=100, ignore_links=False, ignore_images=True, *args, **kwargs):
        """
        初始化方法，用于设置爬虫在启动时的各种配置
        
        参数:
            start_url: 起始 URL
            allowed_domain: 允许的域名
            output_dir: 输出目录
            max_pages: 最大爬取页面数
            ignore_links: 是否忽略链接
            ignore_images: 是否忽略图片
        """
        super(ContentFocusedSpider, self).__init__(*args, **kwargs)
        
        # 设置起始 URL 和允许的域名
        self.start_urls = [start_url or 'https://crawl4ai.com/mkdocs/']
        
        if allowed_domain:
            self.allowed_domains = [allowed_domain]
        else:
            # 从 URL 中提取域名
            parsed = urlparse(self.start_urls[0])
            self.allowed_domains = [parsed.netloc]
        
        # 配置输出目录
        self.output_dir = Path(output_dir or '.data')
        self.markdown_dir = self.output_dir / 'markdown_files'
        
        # 爬取限制
        self.max_pages = int(max_pages)
        self.pages_crawled = 0
        
        # 创建一个 HTML2Text 对象，用于将 HTML 转换为 Markdown
        self.h = html2text.HTML2Text()
        self.h.ignore_links = ignore_links
        self.h.ignore_images = ignore_images
        self.h.ignore_emphasis = False  # 保留强调格式
        self.h.body_width = 0  # 不限制行宽
        self.h.unicode_snob = True  # 使用 Unicode
        
        # 初始化结果列表
        self.results = []
        self.visited_urls = set()
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"爬虫初始化完成: 起始URL={self.start_urls[0]}, 域名={self.allowed_domains}")

    def parse(self, response):
        """
        Scrapy 中的默认解析方法，用于处理每个响应（网页内容）
        
        参数:
            response: Scrapy 的响应对象
        """
        # 检查是否已达到最大页面数
        if self.pages_crawled >= self.max_pages:
            logging.info(f"已达到最大页面数限制: {self.max_pages}")
            return
        
        # 检查是否已访问过该 URL
        if response.url in self.visited_urls:
            return
        self.visited_urls.add(response.url)
        
        # 只处理 HTML 内容
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', errors='ignore')
        if 'text/html' not in content_type and not response.url.endswith(('.html', '.htm', '/')):
            return
        
        try:
            # 使用 BeautifulSoup 解析网页内容
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除不需要的元素
            for elem in soup(['nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript']):
                elem.decompose()

            # 尝试找到主要内容区域
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_='content') or
                soup.find('div', class_='main-content') or
                soup.find('div', id='content')
            )

            content = str(main_content) if main_content else str(soup.body or soup)

            # 转换为 Markdown
            markdown_content = self.h.handle(content)
            
            # 添加页面标题
            title = soup.find('title')
            if title:
                markdown_content = f"# {title.get_text().strip()}\n\n{markdown_content}"

            # 生成文件名
            parsed_url = urlparse(response.url)
            file_path = parsed_url.path.strip('/').replace('/', '_') or 'index'
            # 清理文件名中的非法字符
            file_path = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in file_path)
            markdown_filename = self.markdown_dir / f'{file_path}.md'

            # 写入文件
            with open(markdown_filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            result = {
                'url': response.url,
                'title': title.get_text().strip() if title else '',
                'markdown_file': str(markdown_filename),
            }
            self.results.append(result)
            self.pages_crawled += 1
            
            logging.info(f"[{self.pages_crawled}/{self.max_pages}] 已爬取: {response.url}")

            # 继续爬取页面中的链接
            if self.pages_crawled < self.max_pages:
                for link in response.css('a::attr(href)').getall():
                    if link and not link.startswith(('#', 'javascript:', 'mailto:')):
                        yield response.follow(link, self.parse)
                        
        except Exception as e:
            logging.error(f"处理页面时出错 {response.url}: {e}")

    def closed(self, reason):
        """
        Scrapy 爬虫结束时调用的方法，用于执行清理或保存最终结果的操作
        
        参数:
            reason: 爬虫关闭的原因
        """
        # 保存结果元数据
        results_file = self.output_dir / 'markdown_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        logging.info(f"爬取完成 (原因: {reason})")
        logging.info(f"总共爬取了 {len(self.results)} 个页面")
        logging.info(f"结果元数据保存在: {results_file}")
        logging.info(f"Markdown 文件保存在: {self.markdown_dir}")


def create_crawler_process(settings_override=None):
    """
    创建并配置 CrawlerProcess
    
    参数:
        settings_override: 覆盖默认设置的字典
    
    返回:
        配置好的 CrawlerProcess 实例
    """
    default_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 2,
        'DOWNLOAD_DELAY': 1.5,
        'COOKIES_ENABLED': False,
        'LOG_LEVEL': 'WARNING',
        'RETRY_TIMES': 3,
        'DOWNLOAD_TIMEOUT': 30,
    }
    
    if settings_override:
        default_settings.update(settings_override)
    
    return CrawlerProcess(settings=default_settings)


def main():
    """主函数，解析命令行参数并启动爬虫"""
    parser = argparse.ArgumentParser(
        description='网页爬虫 - 将网页内容转换为 Markdown 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python spider.py --url https://docs.example.com --max-pages 50
  python spider.py --url https://example.com --domain example.com --output ./output
        """
    )
    parser.add_argument('--url', '-u', type=str, default='https://crawl4ai.com/mkdocs/',
                        help='起始 URL (默认: https://crawl4ai.com/mkdocs/)')
    parser.add_argument('--domain', '-d', type=str, default=None,
                        help='允许的域名 (默认: 从 URL 中提取)')
    parser.add_argument('--output', '-o', type=str, default='.data',
                        help='输出目录 (默认: .data)')
    parser.add_argument('--max-pages', '-m', type=int, default=100,
                        help='最大爬取页面数 (默认: 100)')
    parser.add_argument('--delay', type=float, default=1.5,
                        help='请求延迟秒数 (默认: 1.5)')
    parser.add_argument('--keep-links', action='store_true',
                        help='保留 Markdown 中的链接')
    parser.add_argument('--keep-images', action='store_true',
                        help='保留 Markdown 中的图片')
    
    args = parser.parse_args()
    
    # 创建爬虫进程
    process = create_crawler_process({
        'DOWNLOAD_DELAY': args.delay,
    })
    
    # 启动爬虫
    process.crawl(
        ContentFocusedSpider,
        start_url=args.url,
        allowed_domain=args.domain,
        output_dir=args.output,
        max_pages=args.max_pages,
        ignore_links=not args.keep_links,
        ignore_images=not args.keep_images,
    )
    
    logging.info(f"开始爬取: {args.url}")
    process.start()


if __name__ == '__main__':
    main()