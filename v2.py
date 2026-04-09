import streamlit as st
import pandas as pd
from llama_cloud import LlamaCloud
from pypdf import PdfReader, PdfWriter
import requests
import tempfile
import os
import time

st.set_page_config(
    page_title="Bulk PDF Extraction Platform",
    page_icon="📄",
    layout="wide"
)

if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = []
if 'pdf_links' not in st.session_state:
    st.session_state.pdf_links = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

st.title("📄 Bulk PDF Extraction Platform")
st.markdown("Extract structured data from thousands of PDF documents using LlamaExtract")

with st.sidebar:
    st.header("⚙️ Configuration")

    agent_name = st.text_input(
        "Agent Name",
        value="Extraction agent1",
        help="Name of your LlamaExtract configuration/agent"
    )

    api_key = st.text_input(
        "LLAMA_CLOUD_API_KEY",
        type="password",
        help="Your Llama Cloud API key (or set as environment variable)"
    )

    if api_key:
        st.session_state['llama_api_key'] = api_key
    elif 'llama_api_key' not in st.session_state:
        st.session_state['llama_api_key'] = os.environ.get('LLAMA_CLOUD_API_KEY', '')

    st.divider()
    st.subheader("🚀 Processing Settings")

    batch_size = st.slider(
        "Batch Size", min_value=1, max_value=50, value=5,
        help="Number of PDFs to process in each batch"
    )

    page_limit = st.slider(
        "Page Limit per PDF", min_value=1, max_value=50, value=3,
        help="Only process the first N pages of each PDF"
    )

    delay_between_requests = st.slider(
        "Delay Between Requests (seconds)",
        min_value=0.0, max_value=5.0, value=0.5, step=0.1,
        help="Add delay between extractions to avoid rate limits"
    )

    timeout_seconds = st.number_input(
        "Download Timeout (seconds)", min_value=10, max_value=300, value=30,
        help="Timeout for downloading each PDF"
    )

    continue_on_error = st.checkbox(
        "Continue on Error", value=True,
        help="Continue processing if individual PDFs fail"
    )

    st.divider()
    st.header("📋 CSV Format")
    st.markdown("""
    **Required column:** `url` or `link`

    Example CSV:
    ```
    url
    https://example.com/doc1.pdf
    https://example.com/doc2.pdf
    ```

    Optional columns: `id`, `name`, or any other metadata
    """)


def get_client_and_config(name, key):
    if not key:
        st.error("❌ LLAMA_CLOUD_API_KEY not set.")
        return None, None
    try:
        client = LlamaCloud(api_key=key)
        configs = client.configurations.list(name=name)
        config = None
        for c in configs:
            if c.name == name:
                config = c
                break
        if config is None:
            st.error(f'❌ Configuration "{name}" not found.')
            return None, None
        return client, config.id
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        return None, None


def truncate_pdf(pdf_path, max_pages):
    reader = PdfReader(pdf_path)
    if len(reader.pages) <= max_pages:
        return
    writer = PdfWriter()
    for i in range(max_pages):
        writer.add_page(reader.pages[i])
    with open(pdf_path, 'wb') as f:
        writer.write(f)


def process_single_pdf(url, client, config_id, timeout, max_pages=None, metadata=None):
    result = {'url': url, 'status': 'pending', 'metadata': metadata or {}}
    tmp_path = None

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file.close()
        tmp_path = tmp_file.name

        if max_pages:
            truncate_pdf(tmp_path, max_pages)

        with open(tmp_path, 'rb') as f:
            file_obj = client.files.create(file=f, purpose='extract')

        job = client.extract.run(
            file_input=file_obj.id,
            configuration_id=config_id
        )

        result['status'] = 'success'
        result['data'] = job.extract_result if job.extract_result else str(job)

    except requests.exceptions.Timeout:
        result['status'] = 'error'
        result['error'] = 'Download timeout'
    except requests.exceptions.RequestException as e:
        result['status'] = 'error'
        result['error'] = f'Download error: {str(e)[:200]}'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = f'Extraction error: {str(e)[:200]}'
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return result


st.subheader("📎 Upload CSV with PDF Links")

uploaded_csv = st.file_uploader(
    "Choose a CSV file with PDF URLs", type=['csv'],
    help="CSV should contain a column named 'url' or 'link' with PDF URLs"
)

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)

        url_column = None
        for col in ['url', 'link', 'URL', 'Link', 'pdf_url', 'pdf_link']:
            if col in df.columns:
                url_column = col
                break

        if url_column is None:
            st.error("❌ CSV must contain a column named 'url' or 'link'")
        else:
            st.success(f"✅ Found {len(df)} PDF links in column '{url_column}'")

            st.subheader("📋 CSV Preview")
            st.dataframe(df.head(10), width='stretch')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total PDFs", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Unique URLs", df[url_column].nunique())
            with col4:
                st.metric("Null URLs", df[url_column].isnull().sum())

            if st.button("📥 Load Links for Processing", type="primary", width='stretch'):
                st.session_state.pdf_links = []
                st.session_state.extraction_results = []
                for idx, row in df.iterrows():
                    url = row[url_column]
                    if pd.notna(url):
                        metadata = {c: str(row[c]) for c in df.columns if c != url_column and pd.notna(row[c])}
                        st.session_state.pdf_links.append({
                            'url': str(url), 'index': idx,
                            'metadata': metadata, 'status': 'pending'
                        })
                st.success(f"✅ Loaded {len(st.session_state.pdf_links)} valid URLs")
                st.rerun()
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

if st.session_state.pdf_links:
    st.divider()
    st.subheader("🚀 Process PDFs")

    total_links = len(st.session_state.pdf_links)
    completed_count = len([r for r in st.session_state.extraction_results if r['status'] == 'success'])
    failed_count = len([r for r in st.session_state.extraction_results if r['status'] == 'error'])
    processed_count = completed_count + failed_count
    pending_count = total_links - processed_count

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Links", total_links)
    with col2:
        st.metric("Pending", pending_count)
    with col3:
        st.metric("Completed", completed_count)
    with col4:
        st.metric("Failed", failed_count)

    if not st.session_state.processing:
        if st.button("🚀 Start Extraction", type="primary", width='stretch', disabled=pending_count == 0):
            st.session_state.processing = True
            st.rerun()

    if st.session_state.processing:
        if st.button("⏸️ Stop Processing", width='stretch'):
            st.session_state.processing = False
            st.warning("⚠️ Processing will stop after this batch.")
            st.rerun()

        st.info("⏳ Processing PDFs in batches...")
        progress_bar = st.progress(processed_count / total_links if total_links > 0 else 0)
        status_text = st.empty()

        client, config_id = get_client_and_config(agent_name, st.session_state.get('llama_api_key', ''))

        if not client or not config_id:
            st.session_state.processing = False
            st.rerun()
        else:
            pending_links_list = [p for p in st.session_state.pdf_links if p['status'] == 'pending']

            if not pending_links_list:
                st.success(f"✅ Processing complete! {completed_count} successful, {failed_count} failed")
                st.session_state.processing = False
                st.rerun()
            else:
                links_to_process = pending_links_list[:batch_size]
                status_text.text(
                    f"Processing batch of {len(links_to_process)}... "
                    f"({processed_count + 1} to {processed_count + len(links_to_process)} of {total_links})"
                )

                batch_error = False
                for link in links_to_process:
                    for p in st.session_state.pdf_links:
                        if p.get('index') == link['index'] and p['status'] == 'pending':
                            p['status'] = 'processing'
                            break

                    try:
                        result = process_single_pdf(
                            link['url'], client, config_id,
                            timeout_seconds, page_limit, link['metadata']
                        )
                        st.session_state.extraction_results.append(result)

                        for p in st.session_state.pdf_links:
                            if p.get('index') == link['index'] and p['status'] == 'processing':
                                p['status'] = result['status']
                                break

                        if result['status'] == 'error' and not continue_on_error:
                            st.error(f"❌ Stopped: {result.get('error', 'Unknown error')}")
                            st.session_state.processing = False
                            batch_error = True
                            break

                    except Exception as e:
                        error_result = {
                            'url': link['url'], 'status': 'error',
                            'error': f'Batch exception: {str(e)[:200]}',
                            'metadata': link['metadata']
                        }
                        st.session_state.extraction_results.append(error_result)

                        for p in st.session_state.pdf_links:
                            if p.get('index') == link['index'] and p['status'] == 'processing':
                                p['status'] = 'error'
                                break

                        if not continue_on_error:
                            st.error(f"❌ Stopped: {str(e)}")
                            st.session_state.processing = False
                            batch_error = True
                            break

                    if delay_between_requests > 0:
                        time.sleep(delay_between_requests)

                if st.session_state.processing and not batch_error:
                    st.rerun()
                else:
                    st.rerun()

    if not st.session_state.processing:
        if st.button("🗑️ Clear All Results", width='stretch'):
            st.session_state.pdf_links = []
            st.session_state.extraction_results = []
            st.session_state.processing = False
            st.rerun()

if st.session_state.extraction_results:
    st.divider()
    st.subheader("📊 Extraction Results")

    results_data = []
    for result in st.session_state.extraction_results:
        row = {
            'URL': result['url'],
            'Status': '✅ Success' if result['status'] == 'success' else '❌ Error'
        }
        if 'metadata' in result:
            row.update(result['metadata'])

        if result['status'] == 'success':
            data = result.get('data', {})
            if isinstance(data, dict):
                for key, value in data.items():
                    row[key] = str(value) if isinstance(value, (dict, list)) else value
            elif isinstance(data, list):
                row['Extracted_Data'] = str(data)
            else:
                row['Extracted_Data'] = str(data)
        else:
            row['Error'] = result.get('error', 'Unknown error')

        results_data.append(row)

    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, width='stretch', height=400)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name=f"extraction_results_{int(time.time())}.csv",
            mime="text/csv", width='stretch'
        )
    with col2:
        st.download_button(
            label="📥 Download Results JSON",
            data=results_df.to_json(orient='records', indent=2),
            file_name=f"extraction_results_{int(time.time())}.json",
            mime="application/json", width='stretch'
        )
    with col3:
        success_df = results_df[results_df['Status'] == '✅ Success']
        if not success_df.empty:
            st.download_button(
                label="📥 Download Success Only",
                data=success_df.to_csv(index=False),
                file_name=f"extraction_success_{int(time.time())}.csv",
                mime="text/csv", width='stretch'
            )

    failed_df = results_df[results_df['Status'] == '❌ Error']
    if not failed_df.empty:
        st.divider()
        st.subheader("⚠️ Failed Extractions")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Found {len(failed_df)} failed extractions")
            with st.expander("View Failed URLs"):
                st.dataframe(failed_df[['URL', 'Error']], width='stretch')
        with col2:
            st.download_button(
                label="📥 Download Failed URLs",
                data=failed_df[['URL']].rename(columns={'URL': 'url'}).to_csv(index=False),
                file_name=f"failed_urls_{int(time.time())}.csv",
                mime="text/csv", width='stretch'
            )

    with st.expander("🔍 View Raw Extraction Data"):
        st.json(st.session_state.extraction_results)

    st.divider()
    st.subheader("📈 Final Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    success_count = len([r for r in st.session_state.extraction_results if r['status'] == 'success'])
    error_count = len([r for r in st.session_state.extraction_results if r['status'] == 'error'])
    total_processed = len(st.session_state.extraction_results)
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    with col1:
        st.metric("Total Processed", total_processed)
    with col2:
        st.metric("Successful", success_count)
    with col3:
        st.metric("Failed", error_count)
    with col4:
        st.metric("Success Rate", f"{success_rate:.1f}%")

else:
    st.info("📋 Upload a CSV file with PDF URLs to begin extraction")

st.divider()
st.caption("Powered by LlamaCloud & Streamlit")
