import os
import re
import sys  
import time
import textwrap
import datetime as dt
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup  # type: ignore
import arxiv
from openai import OpenAI

import json


client = OpenAI(api_key="sk-psdwxdblqsanpckjwnkoebqnvpgszjdahyvqpgbexltvrkpx", 
                base_url="https://api.siliconflow.cn/v1",
                )

MODEL = "Qwen/Qwen2.5-32B-Instruct"  


ARXIV_NEW_URL = "https://arxiv.org/list/astro-ph/new"
BATCH_SIZE_API = 20      # arXiv API batch size
BATCH_SIZE_SUM = 10      # #abstracts per summarisation call
MAX_RETRIES = 3          # API retries
RETRY_WAIT = 4           # seconds between retries
MAX_PAPERS = 100

TOPIC = "Star formation / Planet formation: including star formation in the Milky Way, and the formation of protostellar/protoplanetary disks, planetary systems around stars. Both theoretical and observational papers."

# TOPIC = "All about astronomy."

OUTPUT_DIR = './astro_ph_daily_picks'

# ╭───────────────────────── prompt templates ──────────────────────────╮

classify_sysprompt = (
    "You are an expert astrophysics librarian.Your job is to determine whether a "
    "scientific paper is related to a given research topic based solely on its title.\n\n"
    "Instructions:\n"
    "- Use your domain knowledge in astrophysics to make a judgment.\n"
    "- Only respond with \"Yes\" or \"No\".\n"
    "- Be conservative: if the relation is unclear or indirect, respond \"No\".\n"
    "- Do not explain or elaborate."
)

classify_userprompt_tpl = (
    "Topic: {TOPIC}\nTitle: {title}\nDoes the title belong to the topic above?"
)

summarize_sysprompt = '''
You are an expert assistant for academic summarization and translation, 
specializing in astrophysics papers.

Your task is to read English abstracts and generate accurate, concise, and 
objective Chinese summaries.
'''

summarize_userprompt_header = '''
For each paper, produce a JSON object containing the following fields,:
- 'arxiv_id': the arXiv ID of the paper
- 'title': the paper title (keep original English)
- 'authors': a list of author names (keep original order and spelling)
- 'summary_zh': a 4–6 sentence objective Chinese summary
- 'keywords_zh': a list of 3–4 precise Chinese keywords

Guidelines:
1. The summary must be strictly objective. Do not include subjective or generic phrases like 
   “对……研究具有重要意义” or “为……提供了重要参考”.
2. Avoid vague or formulaic language. Be concise and precise in academic Chinese.
3. Do not omit key methods, findings, or research subjects.

Input format: for each paper, use the format:
### <arXiv ID>
Title: <title>
Authors: <comma-separated author names>
<abstract>

Return a JSON array containing one object per paper.
Return ONLY the raw JSON array. Do not include markdown formatting like ```json.
'''
# ───────────────────────────────────────────────

# ╭───────────────────────── helper functions ──────────────────────────╮

def fetch_new_submission_entries(url: str = ARXIV_NEW_URL) -> List[Dict[str, str]]:
    """Return list of dicts with 'id' and 'title' from *New submissions* section."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 检查日期
    date_text_tag = soup.find(string=re.compile(r"Showing new listings for"))
    if date_text_tag:
        date_match = re.search(r"\b\w+day, \d{1,2} \w{3,9} \d{4}", date_text_tag)
        if date_match:
            date_str = date_match.group(0)
            try:
                arxiv_date = dt.datetime.strptime(date_str, "%A, %d %B %Y").date()
                if arxiv_date != dt.date.today():
                    print(f"[INFO] 今日未更新。 预期日期: {arxiv_date}, 实际日期: {dt.date.today()}")
                    sys.exit(0)
            except Exception as e:
                print(f"[WARN] 日期解析失败: {e}")
                sys.exit(0)


    # 找 new submissions header and <dl> container

    header = soup.find("h3", string=lambda s: s and "new submissions" in s.lower())
    if not header:
        raise RuntimeError("Cannot locate 'New submissions' header – page structure changed.")

    dl = header.find_parent("dl") or header.find_next("dl")
    if not dl:
        raise RuntimeError("Cannot find <dl> container for new submissions.")

    id_regex = re.compile(r"\d{4}\.\d{4,5}")
    entries: List[Dict[str, str]] = []

    dt_tags = dl.find_all("dt", recursive=False)
    dd_tags = dl.find_all("dd", recursive=False)
    for dt_tag, dd_tag in zip(dt_tags, dd_tags):
        link = dt_tag.find("a", href=re.compile(r"/abs/"))
        if not (link and link["href"]):
            continue
        m = id_regex.search(link["href"])
        if not m:
            continue
        arxiv_id = m.group(0)

        title_div = dd_tag.find("div", class_=re.compile(r"list-title"))
        if not title_div:
            continue
        title_text = title_div.get_text(separator=" ", strip=True)
        title_text = title_text.replace("Title:", "").strip()

        entries.append({"id": arxiv_id, "title": title_text})
        if len(entries) >= MAX_PAPERS:
            break
    return entries


def classify_title(title: str) -> bool:
    """LLM Yes/No using conservative astrophysics librarian prompt."""
    user_prompt = classify_userprompt_tpl.format(TOPIC=TOPIC, title=title)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": classify_sysprompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower().startswith("y")


def classify_titles(entries: List[Dict[str, str]]) -> List[str]:
    """Loop over titles; return list of IDs whose titles get 'Yes'."""
    matched_ids: List[str] = []
    for entry in entries:
        if classify_title(entry["title"]):
            matched_ids.append(entry["id"])
    return matched_ids


def get_metadata_for_ids(ids: List[str]) -> List[arxiv.Result]:
    """Fetch metadata via arXiv API in batches with retry/back‑off."""
    if not ids:
        return []

    client = arxiv.Client()
    results: List[arxiv.Result] = []
    for start in range(0, len(ids), BATCH_SIZE_API):
        batch = ids[start : start + BATCH_SIZE_API]
        retries = MAX_RETRIES
        while retries:
            try:
                search = arxiv.Search(id_list=batch, max_results=len(batch))
                results.extend(list(client.results(search)))
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    print(f"[WARN] Failed batch {batch}: {e}")
                else:
                    wait = RETRY_WAIT * (MAX_RETRIES - retries)
                    print(f"[INFO] Retry in {wait}s … ({retries} left)")
                    time.sleep(wait)
    return results


def extract_json_array(text: str) -> list:
    """
    提取 LLM 响应中的第一个合法 JSON array（忽略前后 markdown 格式等）。
    """
    try:
        # 用正则匹配最外层 JSON 数组：[ {...}, {...} ]
        match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
        if not match:
            raise ValueError("无法找到 JSON 数组。")
        json_str = match.group(0)
        return json.loads(json_str)
    except Exception as e:
        print("[ERROR] JSON 提取失败:", e)
        raise

def batch_summarise(papers: List[arxiv.Result]) -> List[Dict[str, any]]:
    """Summarise abstracts and return list of structured dicts from JSON."""
    summaries: List[Dict[str, any]] = []
    for start in range(0, len(papers), BATCH_SIZE_SUM):
        batch = papers[start : start + BATCH_SIZE_SUM]

        parts = []
        for p in batch:
            part = f"### {p.get_short_id()}\n"
            part += f"Title: {p.title.strip()}\n"
            part += "Authors: " + ", ".join(a.name for a in p.authors) + "\n"
            part += p.summary.strip()
            parts.append(part)
        user_prompt = summarize_userprompt_header + "\n\n" + "\n\n".join(parts)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": summarize_sysprompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        # print(resp)

        try:
            json_block = resp.choices[0].message.content.strip()
            # print(json_block)
            parsed = extract_json_array(json_block)
            summaries.extend(parsed)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
    return summaries


def build_markdown(summaries: List[Dict[str, any]], date: dt.date) -> str:
    """Build markdown from structured summary objects."""
    lines = [f"# 星球日报 \n**{date}**\n"]
    if not summaries:
        lines.append("*(No matching papers today.)*\n")
    for item in summaries:
        lines.append(
            f"### [{item['title']}](https://arxiv.org/abs/{item['arxiv_id']})\n"
            f"**Authors**: {', '.join(item['authors'])}\n\n"
            f"**摘要**:\n{item['summary_zh']}\n\n"
            f"**关键词**: {', '.join(item['keywords_zh'])}\n\n"
            f"-----------------\n\n"
        )
    return "\n".join(lines)

def save_json(summaries: List[Dict[str, any]], path: str) -> None:
    """Save summaries to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


def main():
    target_date = datetime.now().date()

    print("[INFO] Scraping titles …")
    entries = fetch_new_submission_entries()
    print(f"[INFO] Got {len(entries)} entries")

    print("[INFO] Classifying titles …")
    matched_ids = classify_titles(entries)
    print(f"[INFO] {len(matched_ids)} titles match topic")

    print("[INFO] Fetching metadata …")
    papers = get_metadata_for_ids(matched_ids)

    print("[INFO] Batch summarising abstracts …")
    summaries = batch_summarise(papers)
    # print(summaries)
    


    json_path = os.path.join(OUTPUT_DIR, f"arxiv_summary_{target_date}.json")
    save_json(summaries, json_path)

    md = build_markdown(summaries, target_date)

    out_dir = Path(OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"astro_ph_{target_date}.md"
    outfile.write_text(md, encoding="utf-8")
    print(f"[INFO] Wrote {outfile} with {len(summaries)} summaries → {outfile}")


if __name__ == "__main__":
    main()
