# Daily Planet 星球日报
<p align="center">
<img src="./img/报头.jpeg"  width="600" alignment="middle"/>
</p>
一个基于 LangChain 的 LLM 应用，用于每日抓取 arXiv 的新论文，判断是否属于指定主题，并总结其摘要。

* 每日从 arXiv 获取最新论文（requests+BeautifulSoup 爬虫实现）

* 使用 LLM 判断标题是否属于指定主题（如 “TOPIC”）

* arxiv 库获取文章metadata

* 对符合条件的论文摘要进行总结（目前仅总结翻译摘要）

#### 恒星/行星形成相关论文每日总结在微信公众号**宇宙哔哔机**。