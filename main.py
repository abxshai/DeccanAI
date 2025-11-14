# streamlit_concurrent_extractor.py
import streamlit as st
import pandas as pd
import os
import time
import json
import tempfile
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
from llama_cloud_services import LlamaExtract

# -----------------------
# Config / constants
# -----------------------
PERSIST_FILE = Path("extraction_results.jsonl")  # append-only results for resume
MAX_DOWNLOAD_RETRIES = 3
MAX_EXTRACT_RETRIES = 3
BACKOFF_BASE = 1.8
DEFAULT_CONCURRENCY = 10
RESULTS_FLUSH_EVERY = 5  # flush to disk every N results (we append per result anyway)

st.set_page_config(page_title="Concurrent Web Extractor", layout="wide")
st.title("🚀 Concurrent Web Extraction (URLs) — Fast mode")

# -----------------------
# Utilities: persistence / requests session
# -----------------------
def append_result_to_disk(res: dict):
    """Append a single JSON line to disk (atomic-ish)."""
    try:
        with PERSIST_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, default=str) + "\n")
    except Exception as e:
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
    """
    Initialize and cache the LlamaExtract extractor/agent.
    This runs once per name.
    """
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
    if not api_key:
        # we don't raise here to allow sidebar to set env var first
        return None
    try:
        extractor = LlamaExtract()
        agent = extractor.get_agent(name=agent_name)
        return agent
    except Exception as e:
        # Return None and let caller handle error display
        return None

# -----------------------
# Worker: process single URL (with retries)
# -----------------------
def backoff_sleep(attempt: int):
    jitter = random.uniform(0, 0.3)
    sleep_for = (BACKOFF_BASE ** (attempt - 1)) + jitter
    time.sleep(sleep_for)

def process_single_url(url: str, agent, timeout: int, retries: int = MAX_EXTRACT_RETRIES, session: Optional[requests.Session] = None, metadata: Optional[dict] = None):
    """
    Attempts to call agent.extract(url). Retries extraction on exception with backoff.
    Returns a result dict that will be appended to results list/disk.
    """
    result = {
        "url": url,
        "status": "pending",
        "metadata": metadata or {},
        "started_at": time.time()
    }

    # First: try to call agent.extract(url) directly (most agents accept URL)
    for attempt in range(1, retries + 1):
        try:
            extraction_result = agent.extract(url)  # primary path
            # normalize result
            result["status"] = "success"
            result["data"] = extraction_result.data if hasattr(extraction_result, "data") else str(extraction_result)
            break
        except Exception as e:
            # If extraction fails and we still have retries, backoff and retry.
            if attempt == retries:
                # Final failure: attempt a fallback: fetch page ourselves and pass raw text if possible
                # Only attempt fallback if we have a requests session
                if session is not None:
                    try:
                        # Try to download page HTML/text and send as plain text to agent.extract
                        resp = session.get(url, timeout=timeout)
                        resp.raise_for_status()
                        page_text = resp.text
                        # Second fallback: try agent.extract on page_text (some agents accept plain text)
                        try:
                            extraction_result = agent.extract(page_text)
                            result["status"] = "success"
                            result["data"] = extraction_result.data if hasattr(extraction_result, "data") else str(extraction_result)
                        except Exception as e2:
                            result["status"] = "error"
                            result["error"] = f"Extract failed after fallbacks: {str(e2)[:300]}"
                    except Exception as e3:
                        result["status"] = "error"
                        result["error"] = f"Extract error and fallback failed: {str(e3)[:300]}"
                else:
                    result["status"] = "error"
                    result["error"] = f"Extract error: {str(e)[:300]}"
            else:
                backoff_sleep(attempt)
                continue

    result["finished_at"] = time.time()
    return result

# -----------------------
# Streamlit UI: Sidebar config
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

# Load persisted results (resume)
persisted = load_persisted_results()
if persisted and "resumed_once" not in st.session_state:
    # load persisted into session if not already loaded
    st.session_state.extraction_results = persisted
    st.session_state.resumed_once = True

if "extraction_results" not in st.session_state:
    st.session_state.extraction_results = []

if "pdf_links" not in st.session_state:
    st.session_state.pdf_links = []

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
                st.session_state.pdf_links = []
                st.session_state.extraction_results = []  # clear in-memory; persisted results remain
                for _, row in df.iterrows():
                    u = row[url_column]
                    if pd.notna(u):
                        metadata = {c: str(row[c]) for c in df.columns if c != url_column and pd.notna(row[c])}
                        st.session_state.pdf_links.append({"url": str(u).strip(), "metadata": metadata, "status": "pending"})
                st.success(f"Loaded {len(st.session_state.pdf_links)} links")
                st.experimental_rerun()
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Merge persisted results into display (dedupe by URL)
def get_processed_urls():
    processed = set()
    for r in st.session_state.extraction_results:
        processed.add(r.get("url"))
    return processed

# Buttons to resume persisted work
colA, colB = st.columns(2)
with colA:
    if st.button("Resume persisted results (reload from disk)"):
        st.session_state.extraction_results = load_persisted_results()
        st.success("Reloaded persisted results")
        st.experimental_rerun()
with colB:
    if st.button("Clear persisted results (delete file)"):
        try:
            if PERSIST_FILE.exists():
                PERSIST_FILE.unlink()
            st.success("Deleted persisted file")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to delete file: {e}")

# -----------------------
# Processing controls
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

    if not st.session_state.processing:
        if st.button("Start (concurrent)", type="primary"):
            st.session_state.processing = True
            st.experimental_rerun()

    if st.session_state.processing:
        if st.button("Stop processing (stop after current tasks)"):
            st.session_state.processing = False
            st.warning("Processing will stop after current active tasks finish.")
            st.experimental_rerun()

    # Only start worker orchestration if processing flag is set
    if st.session_state.processing:
        agent = get_extraction_agent(agent_name)
        if agent is None:
            st.error("Agent not initialized. Set LLAMA_CLOUD_API_KEY in sidebar and ensure agent name is correct.")
            st.session_state.processing = False
        else:
            # Build list of pending link dicts excluding already processed (persisted or in-memory)
            processed_set = get_processed_urls()
            pending = [p for p in st.session_state.pdf_links if p["url"] not in processed_set]
            total_pending = len(pending)

            if total_pending == 0:
                st.success("No pending links to process.")
                st.session_state.processing = False
                st.experimental_rerun()
            else:
                st.info(f"Processing {total_pending} pending links with {concurrency} workers...")
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                # create requests session for workers
                session = make_requests_session()

                # We'll submit tasks in chunks to avoid submitting all at once for very large lists.
                chunk_size = max(concurrency * 5, 50)  # tune: how many tasks we queue at once
                submitted = 0
                results_this_run = []

                try:
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        while submitted < total_pending and st.session_state.processing:
                            # prepare next chunk of tasks
                            chunk = pending[submitted: submitted + chunk_size]
                            futures = {}
                            for item in chunk:
                                url = item["url"]
                                metadata = item.get("metadata", {})
                                future = executor.submit(process_single_url, url, agent, timeout_seconds, MAX_EXTRACT_RETRIES, session, metadata)
                                futures[future] = url
                                # small jitter between task submissions to reduce bursts
                                time.sleep(0.005)
                            # wait for chunk results as they complete
                            for future in as_completed(futures):
                                url = futures[future]
                                try:
                                    res = future.result()
                                except Exception as e:
                                    res = {"url": url, "status": "error", "error": f"Worker exception: {e}", "metadata": {}}
                                # append to session state & persist
                                st.session_state.extraction_results.append(res)
                                append_result_to_disk(res)
                                results_this_run.append(res)

                                # update UI metrics & progress
                                processed_urls = get_processed_urls()
                                completed_count = len([r for r in st.session_state.extraction_results if r.get("status") == "success"])
                                failed_count = len([r for r in st.session_state.extraction_results if r.get("status") == "error"])
                                processed_count = len(processed_urls)
                                overall_progress = processed_count / total if total > 0 else 1.0
                                progress_bar.progress(min(1.0, overall_progress))
                                status_text.text(f"Processed {processed_count}/{total} — Success: {completed_count} Fail: {failed_count}")

                                # apply per-worker delay if configured
                                if delay_between_requests > 0:
                                    time.sleep(delay_between_requests)
                            submitted += len(chunk)

                            # small pause to allow interruption and avoid CPU spikes
                            time.sleep(0.2)

                            # stop early if user toggled processing flag off in another tab/button
                            if not st.session_state.processing:
                                break

                    # finished chunking loop
                    st.success("Batch run finished (or stopped).")
                except Exception as e:
                    st.error(f"Processing loop failed: {e}")
                    st.session_state.processing = False

                # Completed run - refresh UI
                st.experimental_rerun()

    # allow clearing in non-processing state
    if not st.session_state.processing:
        if st.button("Clear session results (in-memory only)"):
            st.session_state.pdf_links = []
            st.session_state.extraction_results = []
            st.experimental_rerun()

# -----------------------
# Results display (always shown, merged with persisted results)
# -----------------------
if st.session_state.extraction_results:
    st.divider()
    st.subheader("📊 Extraction Results (merged persisted + in-memory)")
    # Merge and dedupe by URL: prefer latest in-memory record
    merged = {}
    # load persisted first (so in-memory overrides)
    persisted_list = load_persisted_results()
    for r in persisted_list:
        merged[r.get("url")] = r
    for r in st.session_state.extraction_results:
        merged[r.get("url")] = r
    results_list = list(merged.values())

    # Build dataframe rows
    rows = []
    for r in results_list:
        row = {"URL": r.get("url"), "Status": "✅ Success" if r.get("status") == "success" else "❌ Error"}
        if r.get("status") == "success":
            data = r.get("data", {})
            if isinstance(data, dict):
                # flatten keys into row but limit long text
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        row[k] = str(v)[:300]
                    else:
                        row[k] = str(v)[:300]
            else:
                row["Extracted_Data"] = str(data)[:300]
        else:
            row["Error"] = r.get("error", "")[:300]
        # include metadata keys
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

st.caption("Concurrent extractor — uses same cached LlamaExtract agent, ThreadPoolExecutor, retries, and persistence.")
