import os
import re
import sys
import time
import json
import datetime as dt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional # Added Optional

import requests
from bs4 import BeautifulSoup # type: ignore
import arxiv

from prompt_template import *

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser 


# --- Configuration  ---
SILICONFLOW_API_KEY = os.getenv("SIL_API_KEY")
if not SILICONFLOW_API_KEY:
    print("[ERROR] Please set the SIL_API_KEY environment variable with your SiliconFlow API key.")
    sys.exit(1)
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

ARXIV_NEW_URL = "https://arxiv.org/list/astro-ph/new"
BATCH_SIZE_API = 20
BATCH_SIZE_SUM = 10
MAX_RETRIES = 3
RETRY_WAIT = 4
MAX_PAPERS = 100

TOPIC = "Star formation / Planet formation: star formation in the Milky Way, including the formation of protostellar/protoplanetary disks. Both theoretical and observational papers. Please exclude paper about comet, accretion disk around black hole..."
OUTPUT_DIR = './astro_ph_daily_picks' # Changed output dir

# --- Pydantic Models for summarization ---
class PaperSummary(BaseModel):
    arxiv_id: str = Field(description="The arXiv ID of the paper")
    title: str = Field(description="The paper title (original English)")
    authors: List[str] = Field(description="A list of author names (original order and spelling)")
    summary_zh: str = Field(description="A 4–6 sentence objective Chinese summary")
    keywords_zh: List[str] = Field(description="A list of 3–4 precise Chinese keywords")

# --- Langchain LLM Initialization ---
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    temperature=0
)

llm_summarize = ChatOpenAI(
    model=MODEL_NAME,
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
    temperature=0.3
)

# prompt templates for classification and summarization
classify_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(classify_system_template),
    HumanMessagePromptTemplate.from_template(classify_human_template)
])

summarize_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(summarize_system_template),
    HumanMessagePromptTemplate.from_template(summarize_human_template_header + "\n\n{papers_input_str}")
])

# --- Langchain Chains ---

# Classification Chain
classify_chain = classify_prompt | llm | StrOutputParser()

# Summarization Chain 
summarize_llm_chain = summarize_prompt | llm_summarize | StrOutputParser()


def sanitize_json_string(s: str) -> str:
    """
    Replaces backslashes that are not part of a valid JSON escape sequence
    with double backslashes to make them literal backslashes for JSON parsing.
    Valid JSON escapes: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
    This regex looks for a \ that is NOT followed by one of ", \, /, b, f, n, r, t, u.
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

# --- Robust JSON Parsing Helper Function ---
def extract_and_parse_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
    # Ensure the text starts with [ and ends with ] (strip any leading/trailing junk)
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    original_text_stripped = text.strip()
    
    # Attempt 1: Use specific regex to extract JSON array, then sanitize and parse.
    match_specific = re.search(r"\[\s*(\{.*?}(?:\s*,\s*\{.*?\})*\s*)?\]", original_text_stripped, re.DOTALL)
    
    if match_specific:
        json_candidate_str = match_specific.group(0)
        sanitized_candidate = sanitize_json_string(json_candidate_str)
        try:
            data = json.loads(sanitized_candidate)
            if isinstance(data, list):
                print("[INFO] Successfully parsed JSON using specific regex and sanitization.")
                return data
        except json.JSONDecodeError as e:
            print(f"[WARN] Attempt 1a (specific regex + sanitize) failed: {e}. Candidate (first 200 after sanitize): {sanitized_candidate[:200]}")
            # Fallback for Attempt 1: Try parsing the specific regex match *without* sanitization
            try:
                data_no_sanitize = json.loads(json_candidate_str)
                if isinstance(data_no_sanitize, list):
                    print("[INFO] Successfully parsed JSON using specific regex *without* sanitization (sanitization might have been the issue or not needed).")
                    return data_no_sanitize
            except json.JSONDecodeError as e_no_sanitize:
                print(f"[WARN] Attempt 1b (specific regex, no sanitize) also failed: {e_no_sanitize}. Candidate (first 200): {json_candidate_str[:200]}")
    else:
        print("[DEBUG] Specific regex did not find a JSON array structure.")

    # Attempt 2: Try to parse the whole original text after sanitization.
    # This helps if the regex failed to extract the correct part, or if the original text was already "clean" JSON that just needed sanitizing.
    sanitized_original_text = sanitize_json_string(original_text_stripped)
    try:
        data = json.loads(sanitized_original_text)
        if isinstance(data, list):
            print("[INFO] Successfully parsed JSON using full original text with sanitization.")
            return data
        else:
            print(f"[WARN] Expected a JSON list from full original text parse (sanitized), but got type {type(data)}. Content (first 200): {sanitized_original_text[:200]}")
            return None 
    except json.JSONDecodeError as e:
        print(f"[WARN] Attempt 2 (full original text + sanitize) failed: {e}. Text (first 200 after sanitize): {sanitized_original_text[:200]}")

    # Attempt 3: Try to parse the whole original text *without* any sanitization.
    # This is a last resort in case sanitization was detrimental or the error lies elsewhere.
    try:
        data_orig_no_sanitize = json.loads(original_text_stripped)
        if isinstance(data_orig_no_sanitize, list):
            print("[INFO] Successfully parsed JSON using full original text *without* sanitization.")
            return data_orig_no_sanitize
        else:
            print(f"[WARN] Expected a JSON list from full original text (no sanitize) parse, but got type {type(data_orig_no_sanitize)}. Content (first 200): {original_text_stripped[:200]}")
            return None
    except json.JSONDecodeError as e_orig_no_sanitize:
        print(f"[WARN] Attempt 3 (full original text, no sanitize) failed: {e_orig_no_sanitize}. Text (first 200): {original_text_stripped[:200]}")

    print(f"[ERROR] Failed to extract or parse JSON list from LLM output after all attempts: {original_text_stripped[:500]}...")
    return None

# --- Helper Functions ---
def fetch_new_submission_entries(url: str = ARXIV_NEW_URL) -> List[Dict[str, str]]:
    """Return list of dicts with 'id' and 'title' from *New submissions* section."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch arXiv page: {e}")
        sys.exit(1)

    soup = BeautifulSoup(resp.text, "html.parser")

    date_text_tag = soup.find(string=re.compile(r"Showing new listings for"))
    if date_text_tag:
        date_match = re.search(r"\b\w+day, \d{1,2} \w{3,9} \d{4}", date_text_tag.string)
        if date_match:
            date_str = date_match.group(0)
            try:
                arxiv_date = dt.datetime.strptime(date_str, "%A, %d %B %Y").date()
                if arxiv_date != dt.date.today():
                    print(f"[INFO] arXiv not updated for today. Expected: {arxiv_date}, Today: {dt.date.today()}")
                    sys.exit(0)
            except ValueError as e:
                print(f"[WARN] Date parsing failed: {e}")
        else:
            print("[WARN] Could not find date string in expected format.")
    else:
        print("[WARN] Could not find date information tag.")


    header = soup.find("h3", string=lambda s: s and "new submissions" in s.lower())
    if not header:
        print("[ERROR] Cannot locate 'New submissions' header – page structure might have changed.")
        return []

    dl = header.find_parent("dl") or header.find_next("dl")
    if not dl:
        print("[ERROR] Cannot find <dl> container for new submissions.")
        return []

    id_regex = re.compile(r"\d{4}\.\d{4,5}")
    entries: List[Dict[str, str]] = []

    dt_tags = dl.find_all("dt", recursive=False)
    dd_tags = dl.find_all("dd", recursive=False)
    for dt_tag, dd_tag in zip(dt_tags, dd_tags):
        link = dt_tag.find("a", href=re.compile(r"/abs/"))
        if not (link and link.get("href")):
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


def classify_title_langchain(title: str, topic: str) -> bool:
    """Classify a single title using Langchain chain."""
    try:
        response = classify_chain.invoke({"topic": topic, "title": title})
        return response.strip().lower().startswith("y")
    except Exception as e:
        print(f"[WARN] Error during classification for title '{title}': {e}")
        return False

def classify_titles_langchain(entries: List[Dict[str, str]], topic: str) -> List[str]:
    """Loop over titles; return list of IDs whose titles get 'Yes' using Langchain."""
    matched_ids: List[str] = []
    for i, entry in enumerate(entries):
        print(f"[INFO] Classifying title {i+1}/{len(entries)}: {entry['title'][:50]}...")
        if classify_title_langchain(entry["title"], topic):
            matched_ids.append(entry["id"])
            print(f"[INFO] ...Matched: {entry['id']}")
        time.sleep(0.2)
    return matched_ids


def get_metadata_for_ids(ids: List[str]) -> List[arxiv.Result]:
    """Fetch metadata via arXiv API in batches with retry/back-off."""
    if not ids:
        return []
    api_client = arxiv.Client(
        page_size = BATCH_SIZE_API,
        delay_seconds = RETRY_WAIT,
        num_retries = MAX_RETRIES
    )
    results: List[arxiv.Result] = []
    try:
        search = arxiv.Search(id_list=ids, max_results=len(ids))
        fetched_results = list(api_client.results(search))
        results.extend(fetched_results)
        print(f"[INFO] Successfully fetched metadata for {len(fetched_results)} IDs.")
    except Exception as e:
        print(f"[WARN] Error during arXiv API metadata fetch for IDs {ids}: {e}")
    return results

def fix_all_latex_backslashes_in_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace all `\` with `\\`.
    """
    for k, v in obj.items():
        if isinstance(v, str):
            obj[k] = v.replace("\\", "\\\\")
    return obj


def batch_summarise_langchain(papers: List[arxiv.Result], current_target_date: dt.date) -> List[Dict[str, Any]]:
    all_summaries_structured: List[Dict[str, Any]] = []
    for i in range(0, len(papers), BATCH_SIZE_SUM):
        batch = papers[i : i + BATCH_SIZE_SUM]
        print(f"[INFO] Summarizing batch {i//BATCH_SIZE_SUM + 1}/{(len(papers) + BATCH_SIZE_SUM - 1)//BATCH_SIZE_SUM}...")
        papers_input_parts = [
            f"### {p.get_short_id()}\nTitle: {p.title.strip()}\nAuthors: {', '.join(a.name for a in p.authors)}\n{p.summary.strip()}"
            for p in batch
        ]
        papers_input_str = "\n\n".join(papers_input_parts)
        try:
            raw_llm_output_str = summarize_llm_chain.invoke({"papers_input_str": papers_input_str})
            parsed_json_list = extract_and_parse_json_list(raw_llm_output_str)
            print('---------------- parse output ----------------')
            print(parsed_json_list)
            if parsed_json_list:
                batch_pydantic_objects: List[PaperSummary] = []
                for item_dict in parsed_json_list:
                    try:
                        # Do NOT fix LaTeX backslashes here; let JSON parsing handle it naturally
                        summary_obj = PaperSummary(**item_dict)
                        batch_pydantic_objects.append(summary_obj)
                    except Exception as e_pydantic:
                        print(f"[ERROR] Pydantic validation failed for item: {item_dict}. Error: {e_pydantic}")
                        print(f"[DEBUG] Raw LLM output for this batch (first 500 chars): {raw_llm_output_str[:500]}")
                all_summaries_structured.extend(p_obj.dict() for p_obj in batch_pydantic_objects)
                print(f"[INFO] ...Successfully parsed and validated {len(batch_pydantic_objects)} summaries in this batch.")
            else:
                print(f"[ERROR] Failed to get valid JSON list from LLM for batch. Skipping this batch.")
                debug_file = Path(OUTPUT_DIR) / f"failed_batch_{current_target_date.strftime('%Y-%m-%d')}_{i//BATCH_SIZE_SUM + 1}.txt"
                debug_file.write_text(f"---PROMPT SUBSET (first paper)---\n{papers_input_parts[0] if papers_input_parts else 'N/A'}\n\n---RAW_LLM_OUTPUT---\n{raw_llm_output_str}", encoding="utf-8")
                print(f"[DEBUG] Saved problematic input (subset) and output to {debug_file}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during summarization for a batch: {e}")
            print(f"[DEBUG] Problematic input (first paper in batch): {papers_input_parts[0] if papers_input_parts else 'N/A'}")
    return all_summaries_structured

def build_markdown(summaries: List[Dict[str, Any]], date: dt.date) -> str:
    """Build markdown from structured summary objects."""
    lines = [f"# 星球日报\n**{date.strftime('%Y-%m-%d')}**\n"] # Minor title change
    if not summaries:
        lines.append("*(今天没有匹配的论文 Our topic has no matching papers today.)*\n")
    for item in summaries:
        title = item.get('title', 'N/A')
        arxiv_id = item.get('arxiv_id', 'N/A')
        authors = item.get('authors', [])
        summary_zh = item.get('summary_zh', 'N/A')
        # Do NOT replace backslashes here! This preserves LaTeX commands like \rm, \sim, etc.
        keywords_zh = item.get('keywords_zh', [])
        print(summary_zh)
        lines.append(
            f"### [{title}](https://arxiv.org/abs/{arxiv_id})\n"
            f"**Authors**: {', '.join(authors)}\n\n"
            f"**摘要 (Summary)**:\n{summary_zh}\n\n"
            f"**关键词 (Keywords)**: {', '.join(keywords_zh)}\n\n"
            f"-----------------\n\n"
        )
    return "\n".join(lines)

def save_json(summaries: List[Dict[str, Any]], path: str) -> None:
    """Save summaries to a JSON file."""
    # This function remains the same as the original.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

# Global variable for target_date to be accessible in batch_summarise_langchain for debug file naming
target_date = datetime.now().date()

def main():
    global target_date # Allow main to set this global
    target_date = datetime.now().date()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Langchain Daily Planet Report (Robust Parsing) for {target_date.strftime('%Y-%m-%d')}")
    print(f"[INFO] Topic: {TOPIC}")

    print("\n[PHASE 1] Scraping titles from arXiv...")
    entries = fetch_new_submission_entries()
    if not entries:
        print("[INFO] No new entries found on arXiv. Exiting.")
        sys.exit(0)
    print(f"[INFO] Got {len(entries)} new entries from arXiv.")

    print("\n[PHASE 2] Classifying titles...")
    matched_ids = classify_titles_langchain(entries, TOPIC)
    if not matched_ids:
        print("[INFO] No titles matched the topic after classification. Exiting.")
        md_content = build_markdown([], target_date)
        outfile_md = Path(OUTPUT_DIR) / f"astro_ph_{target_date.strftime('%Y-%m-%d')}.md"
        outfile_md.write_text(md_content, encoding="utf-8")
        print(f"[INFO] Wrote empty report to {outfile_md}")
        json_path = Path(OUTPUT_DIR) / f"arxiv_summary_{target_date.strftime('%Y-%m-%d')}.json"
        save_json([], str(json_path))
        print(f"[INFO] Wrote empty JSON summary to {json_path}")
        sys.exit(0)
    print(f"[INFO] {len(matched_ids)} titles matched the topic: {', '.join(matched_ids)}")

    print("\n[PHASE 3] Fetching metadata for matched papers from arXiv API...")
    papers = get_metadata_for_ids(matched_ids)
    if not papers:
        print("[INFO] Failed to fetch metadata for any matched papers. Exiting.")
        sys.exit(1)
    print(f"[INFO] Successfully fetched metadata for {len(papers)} papers.")


    print("\n[PHASE 4] Batch summarising abstracts...")
    summaries = batch_summarise_langchain(papers, target_date)
    if not summaries:
        print("[INFO] No summaries were generated. This might indicate an issue with the summarization LLM or parsing.")
    else:
        print(f"[INFO] Successfully generated {len(summaries)} summaries.")

    json_path = Path(OUTPUT_DIR) / f"arxiv_summary_{target_date.strftime('%Y-%m-%d')}.json"
    save_json(summaries, str(json_path))
    print(f"[INFO] Saved {len(summaries)} summaries to {json_path}")

    print("\n[PHASE 5] Building Markdown report...")
    md_content = build_markdown(summaries, target_date)

    outfile_md = Path(OUTPUT_DIR) / f"astro_ph_{target_date.strftime('%Y-%m-%d')}.md"
    outfile_md.write_text(md_content, encoding="utf-8")
    print(f"[INFO] Wrote final report with {len(summaries)} summaries to {outfile_md}")
    print("\n[INFO] Process completed.")

if __name__ == "__main__":
    main()