# streamlit_concurrent_extractor_fixed.py
import streamlit as st
import pandas as pd
import os
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
from llama_cloud_services import LlamaExtract

# -----------------------
# Constants / config
# -----------------------
PERSIST_FILE = Path("extraction_results.jsonl")
MAX_EXTRACT_RETRIES = 3
BACKOFF_BASE = 1.8
DEFAULT_CONCURRENCY = 10

st.set_page_config(page_title="Concurrent Web Extractor", layout="wide")
st.title("🚀 Concurrent Web Extraction (URLs) — Fixed rerun behaviour")

# -----------------------
# Helpers: persistence / requests session
# -----------------------
def append_result_to_disk(res: dict):
    """Append a single JSON line to disk."""
    try:
        with PERSIST_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, default=str) + "\n")
    except Exception as e:
        # Don't crash the app if disk write fails; show message later
        st.error(f"Failed to persist result: {e}")

def load_persisted_results():
    results = []
    if PERSIST_FILE.exists():
        try:
            with PERSIST_FILE.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            pass
    return results

@st.cache_resource
def make_requests_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

# -----------------------
# Cached LlamaExtract agent
# -----------------------
@st.cache_resource
def get_extraction_agent(agent_name: str):
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
    if not api_key:
        return None
    try:
        extractor = LlamaExtract()
        agent = extractor.get_agent(name=agent_name)
        return agent
    except Exception:
        return None

# -----------------------
# Worker: process single URL (with retries/backoff)
# -----------------------
def backoff_sleep(attempt: int):
    jitter = random.uniform(0, 0.3)
    sleep_for = (BACKOFF_BASE ** (attempt - 1)) + jitter
    time.sleep(sleep_for)

def process_single_url(url: str, agent, timeout: int, retries: int = MAX_EXTRACT_RETRIES, session: Optional[requests.Session] = None, metadata: Optional[dict] = None):
    result = {
        "url": url,
        "status": "pending",
        "metadata": metadata or {},
        "started_at": time.time()
    }

    for attempt in range(1, retries + 1):
        try:
            extraction_result = agent.extract(url)
            result["status"] = "success"
            result["data"] = extraction_result.data if hasattr(extraction_result, "data") else str(extraction_result)
            break
        except Exception as e:
            if attempt == retries:
                # fallback: fetch page text and try again if possible
                if session is not None:
                    try:
                        resp = session.get(url, timeout=timeout)
                        resp.raise_for_status()
                        page_text = resp.text
                        try:
                            extraction_result = agent.extract(page_text)
                            result["status"] = "success"
                            result["data"] = extraction_result.data if hasattr(extraction_result, "data") else str(extraction_result)
                        except Exception as e2:
                            result["status"] = "error"
                            result["error"] = f"Extract failed after fallback: {str(e2)[:300]}"
                    except Exception as e3:
                        result["status"] = "error"
                        result["error"] = f"Extract + fallback failed: {str(e3)[:300]}"
                else:
                    result["status"] = "error"
                    result["error"] = f"Extract error: {str(e)[:300]}"
            else:
                backoff_sleep(attempt)
                continue

    result["finished_at"] = time.time()
    return result

# -----------------------
# Sidebar config
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    agent_name = st.text_input("Agent Name", value="Extraction agent1")
    api_key_input = st.text_input("LLAMA_CLOUD_API_KEY (or set env var)", type="password")
    if api_key_input:
        os.environ["LLAMA_CLOUD_API_KEY"] = api_key_input

    concurrency = st.slider("Concurrency (parallel workers)", min_value=1, max_value=50, value=DEFAULT_CONCURRENCY)
    delay_between_requests = st.slider("Delay between requests (s) per worker", min_value=0.0, max_value=3.0, value=0.1, step=0.05)
    timeout_seconds = st.number_input("Request timeout (s)", min_value=5, max_value=300, value=30)
    continue_on_error = st.checkbox("Continue on individual error", value=True)
    st.markdown("---")
    st.markdown("Persistence: results saved to `extraction_results.jsonl` so you can resume after interruptions.")

# -----------------------
# Upload CSV / load URLs
# -----------------------
st.subheader("📥 Upload CSV with URL column (url / link / URL / Link)")
uploaded_csv = st.file_uploader("CSV file", type=["csv"])

# Load persisted results into session_state once (resume)
if "persisted_loaded" not in st.session_state:
    persisted = load_persisted_results()
    if persisted:
        st.session_state.extraction_results = persisted.copy()
    else:
        st.session_state.extraction_results = []
    st.session_state.persisted_loaded = True

if "pdf_links" not in st.session_state:
    st.session_state.pdf_links = []

# Button handler: Load links
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        url_column = None
        for col in ["url", "link", "URL", "Link", "page_url"]:
            if col in df.columns:
                url_column = col
                break
        if url_column is None:
            st.error("CSV must contain a column named 'url' or 'link'")
        else:
            st.success(f"Found {len(df)} rows in column '{url_column}'")
            st.dataframe(df.head(10), use_container_width=True)
            if st.button("Load links into session"):
                # load links
                st.session_state.pdf_links = []
                # keep persisted results (so we can resume) but clear in-memory extraction results if desired
                st.session_state.extraction_results = st.session_state.get("extraction_results", [])
                for _, row in df.iterrows():
                    u = row[url_column]
                    if pd.notna(u):
                        metadata = {c: str(row[c]) for c in df.columns if c != url_column and pd.notna(row[c])}
                        st.session_state.pdf_links.append({"url": str(u).strip(), "metadata": metadata, "status": "pending"})
                st.success(f"Loaded {len(st.session_state.pdf_links)} links")
                # safe rerun inside button callback
                st.rerun()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# -----------------------
# Resume / clear persisted controls (button callbacks only)
# -----------------------
colA, colB = st.columns(2)
with colA:
    if st.button("Resume persisted results (reload from disk)"):
        st.session_state.extraction_results = load_persisted_results()
        st.success("Reloaded persisted results")
        st.rerun()
with colB:
    if st.button("Clear persisted results (delete file)"):
        try:
            if PERSIST_FILE.exists():
                PERSIST_FILE.unlink()
            st.session_state.extraction_results = []
            st.success("Deleted persisted file and cleared results")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to delete file: {e}")

# -----------------------
# Utility: processed URLs set
# -----------------------
def get_processed_urls():
    processed = set()
    for r in st.session_state.extraction_results:
        processed.add(r.get("url"))
    return processed

# -----------------------
# Processing controls & orchestration
# -----------------------
if st.session_state.get("pdf_links"):
    st.divider()
    st.subheader("🚀 Start concurrent extraction")

    total = len(st.session_state.pdf_links)
    processed_urls = get_processed_urls()
    completed_count = len([r for r in st.session_state.extraction_results if r.get("status") == "success"])
    failed_count = len([r for r in st.session_state.extraction_results if r.get("status") == "error"])
    pending_count = total - len(processed_urls)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Pending", pending_count)
    c3.metric("Completed", completed_count)
    c4.metric("Failed", failed_count)

    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Start button (safe rerun in callback)
    if not st.session_state.processing:
        if st.button("Start (concurrent)", type="primary"):
            st.session_state.processing = True
            st.rerun()

    # Stop button (safe rerun in callback)
    if st.session_state.processing:
        if st.button("Stop processing (stop after current tasks)"):
            st.session_state.processing = False
            st.warning("Processing will stop after current active tasks finish.")
            st.rerun()

    # Only run processing logic when processing flag is True
    if st.session_state.processing:
        # Initialize agent
        agent = get_extraction_agent(agent_name)
        if agent is None:
            st.error("Agent not initialized. Set LLAMA_CLOUD_API_KEY in sidebar and ensure agent name is correct.")
            st.session_state.processing = False
        else:
            processed_set = get_processed_urls()
            pending = [p for p in st.session_state.pdf_links if p["url"] not in processed_set]
            total_pending = len(pending)

            if total_pending == 0:
                st.success("No pending links to process.")
                st.session_state.processing = False
            else:
                st.info(f"Processing {total_pending} pending links with {concurrency} workers...")
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                session = make_requests_session()
                chunk_size = max(concurrency * 5, 50)
                submitted = 0

                try:
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        while submitted < total_pending and st.session_state.processing:
                            chunk = pending[submitted: submitted + chunk_size]
                            futures = {}
                            for item in chunk:
                                url = item["url"]
                                metadata = item.get("metadata", {})
                                future = executor.submit(process_single_url, url, agent, timeout_seconds, MAX_EXTRACT_RETRIES, session, metadata)
                                futures[future] = url
                                time.sleep(0.005)
                            for future in as_completed(futures):
                                url = futures[future]
                                try:
                                    res = future.result()
                                except Exception as e:
                                    res = {"url": url, "status": "error", "error": f"Worker exception: {e}", "metadata": {}}
                                # append to session state & persist
                                st.session_state.extraction_results.append(res)
                                append_result_to_disk(res)

                                # update UI metrics & progress
                                processed_urls = get_processed_urls()
                                completed_count = len([r for r in st.session_state.extraction_results if r.get("status") == "success"])
                                failed_count = len([r for r in st.session_state.extraction_results if r.get("status") == "error"])
                                processed_count = len(processed_urls)
                                overall_progress = processed_count / total if total > 0 else 1.0
                                progress_bar.progress(min(1.0, overall_progress))
                                status_text.text(f"Processed {processed_count}/{total} — Success: {completed_count} Fail: {failed_count}")

                                if delay_between_requests > 0:
                                    time.sleep(delay_between_requests)
                            submitted += len(chunk)
                            time.sleep(0.2)

                            # if user pressed Stop in another tab/button, break early
                            if not st.session_state.processing:
                                break

                    st.success("Processing loop finished (or stopped).")
                except Exception as e:
                    st.error(f"Processing loop failed: {e}")
                    st.session_state.processing = False

# allow clearing in non-processing state (in-memory only)
if not st.session_state.get("processing", False):
    if st.button("Clear session results (in-memory only)"):
        st.session_state.pdf_links = []
        st.session_state.extraction_results = []
        st.success("Cleared in-memory results")

# -----------------------
# Results display (merged persisted + in-memory)
# -----------------------
if st.session_state.extraction_results:
    st.divider()
    st.subheader("📊 Extraction Results (merged persisted + in-memory)")

    merged = {}
    persisted_list = load_persisted_results()
    for r in persisted_list:
        merged[r.get("url")] = r
    for r in st.session_state.extraction_results:
        merged[r.get("url")] = r
    results_list = list(merged.values())

    rows = []
    for r in results_list:
        row = {"URL": r.get("url"), "Status": "✅ Success" if r.get("status") == "success" else "❌ Error"}
        if r.get("status") == "success":
            data = r.get("data", {})
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        row[k] = str(v)[:300]
                    else:
                        row[k] = str(v)[:300]
            else:
                row["Extracted_Data"] = str(data)[:300]
        else:
            row["Error"] = r.get("error", "")[:300]
        meta = r.get("metadata", {})
        for mk, mv in meta.items():
            row[mk] = mv
        rows.append(row)

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, height=400)

    col1, col2, col3 = st.columns(3)
    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name=f"results_{int(time.time())}.csv", mime="text/csv")
    with col2:
        json_data = json.dumps(results_list, indent=2)
        st.download_button("Download JSON", json_data, file_name=f"results_{int(time.time())}.json", mime="application/json")
    with col3:
        failed_df = results_df[results_df["Status"] == "❌ Error"]
        if not failed_df.empty:
            failed_csv = failed_df[["URL", "Error"]].to_csv(index=False)
            st.download_button("Download Failed URLs", failed_csv, file_name=f"failed_{int(time.time())}.csv", mime="text/csv")
else:
    st.info("Upload CSV and load links to start.")

st.caption("Concurrent extractor (fixed). Uses cached LlamaExtract agent, ThreadPoolExecutor, retries, and persistence.")
