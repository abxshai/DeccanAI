import streamlit as st
import pandas as pd
from llama_cloud_services import LlamaExtract
import requests
import tempfile
import os
import time
from typing import List, Dict
import threading
from queue import Queue

# Page configuration
st.set_page_config(
    page_title="Bulk PDF Extraction Platform",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = []

if 'pdf_links' not in st.session_state:
    st.session_state.pdf_links = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

# Title and description
st.title("📄 Bulk PDF Extraction Platform")
st.markdown("Extract structured data from thousands of PDF documents using LlamaExtract")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    agent_name = st.text_input(
        "Agent Name",
        value="Extraction agent1",
        help="Name of your LlamaExtract agent"
    )
    
    api_key = st.text_input(
        "LLAMA_CLOUD_API_KEY",
        type="password",
        help="Your Llama Cloud API key (or set as environment variable)"
    )
    
    if api_key:
        os.environ['LLAMA_CLOUD_API_KEY'] = api_key
    
    st.divider()
    
    st.subheader("🚀 Processing Settings")
    
    delay_between_requests = st.slider(
        "Delay Between Requests (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="Add delay between extractions to avoid rate limits"
    )
    
    timeout_seconds = st.number_input(
        "Download Timeout (seconds)",
        min_value=10,
        max_value=300,
        value=30,
        help="Timeout for downloading each PDF"
    )
    
    continue_on_error = st.checkbox(
        "Continue on Error",
        value=True,
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
    https://example.com/doc3.pdf
    ```
    
    Optional columns:
    - `id` - Document identifier
    - `name` - Document name
    - Any other metadata columns
    """)

def process_single_pdf(url: str, agent_name: str, timeout: int, metadata: dict = None):
    """Process a single PDF - creates fresh agent instance"""
    result = {
        'url': url,
        'status': 'pending',
        'metadata': metadata or {}
    }
    
    tmp_path = None
    
    try:
        # Download PDF
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file.close()
        tmp_path = tmp_file.name
        
        # Create a fresh extractor and agent for this request
        extractor = LlamaExtract()
        agent = extractor.get_agent(name=agent_name)
        
        if agent is None:
            result['status'] = 'error'
            result['error'] = f'Agent "{agent_name}" not found'
            return result
        
        # Extract data
        extraction_result = agent.extract(tmp_path)
        
        # Store results
        result['status'] = 'success'
        result['data'] = extraction_result.data if hasattr(extraction_result, 'data') else str(extraction_result)
        
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
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    return result

# Main content area
st.subheader("📎 Upload CSV with PDF Links")

uploaded_csv = st.file_uploader(
    "Choose a CSV file with PDF URLs",
    type=['csv'],
    help="CSV should contain a column named 'url' or 'link' with PDF URLs"
)

if uploaded_csv is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_csv)
        
        # Find URL column
        url_column = None
        for col in ['url', 'link', 'URL', 'Link', 'pdf_url', 'pdf_link']:
            if col in df.columns:
                url_column = col
                break
        
        if url_column is None:
            st.error("❌ CSV must contain a column named 'url' or 'link'")
        else:
            st.success(f"✅ Found {len(df)} PDF links in column '{url_column}'")
            
            # Show preview
            st.subheader("📋 CSV Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total PDFs", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Unique URLs", df[url_column].nunique())
            with col4:
                st.metric("Null URLs", df[url_column].isnull().sum())
            
            # Load links into session state
            if st.button("📥 Load Links for Processing", type="primary", use_container_width=True):
                st.session_state.pdf_links = []
                for idx, row in df.iterrows():
                    url = row[url_column]
                    if pd.notna(url):
                        metadata = {col: str(row[col]) for col in df.columns if col != url_column and pd.notna(row[col])}
                        st.session_state.pdf_links.append({
                            'url': str(url),
                            'metadata': metadata,
                            'status': 'pending'
                        })
                st.success(f"✅ Loaded {len(st.session_state.pdf_links)} valid URLs")
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

# Processing section
if st.session_state.pdf_links:
    st.divider()
    st.subheader("🚀 Process PDFs")
    
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    
    pending = sum(1 for p in st.session_state.pdf_links if p['status'] == 'pending')
    processing_status = sum(1 for p in st.session_state.pdf_links if p['status'] == 'processing')
    completed = len([r for r in st.session_state.extraction_results if r['status'] == 'success'])
    failed = len([r for r in st.session_state.extraction_results if r['status'] == 'error'])
    
    with col1:
        st.metric("Pending", pending)
    with col2:
        st.metric("Processing", processing_status)
    with col3:
        st.metric("Completed", completed)
    with col4:
        st.metric("Failed", failed)
    
    if not st.session_state.processing:
        if st.button("🚀 Start Extraction", type="primary", use_container_width=True, disabled=pending == 0):
            st.session_state.processing = True
            st.rerun()
    
    if st.session_state.processing:
        st.info("⏳ Processing PDFs sequentially to avoid event loop conflicts...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        pending_links = [p for p in st.session_state.pdf_links if p['status'] == 'pending']
        total = len(pending_links)
        
        if total == 0:
            st.warning("No pending links to process")
            st.session_state.processing = False
            st.rerun()
        
        completed_count = 0
        success_count = 0
        error_count = 0
        
        for idx, link in enumerate(pending_links):
            status_text.text(f"Processing {idx + 1}/{total}: {link['url'][:60]}...")
            
            for p in st.session_state.pdf_links:
                if p['url'] == link['url']:
                    p['status'] = 'processing'
                    break
            
            try:
                result = process_single_pdf(
                    link['url'],
                    agent_name,
                    timeout_seconds,
                    link['metadata']
                )
                
                st.session_state.extraction_results.append(result)
                
                for p in st.session_state.pdf_links:
                    if p['url'] == link['url']:
                        p['status'] = result['status']
                        break
                
                if result['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1
                    if not continue_on_error:
                        st.error(f"❌ Stopped due to error: {result.get('error', 'Unknown error')}")
                        break
                
            except Exception as e:
                error_result = {
                    'url': link['url'],
                    'status': 'error',
                    'error': f'Processing exception: {str(e)[:200]}',
                    'metadata': link['metadata']
                }
                st.session_state.extraction_results.append(error_result)
                error_count += 1
                
                if not continue_on_error:
                    st.error(f"❌ Stopped due to exception: {str(e)}")
                    break
            
            completed_count += 1
            progress_bar.progress(completed_count / total)
            
            with results_container:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed", completed_count)
                with col2:
                    st.metric("Success", success_count)
                with col3:
                    st.metric("Failed", error_count)
            
            if delay_between_requests > 0 and idx < total - 1:
                time.sleep(delay_between_requests)
        
        st.success(f"✅ Processing complete! {success_count} successful, {error_count} failed")
        st.session_state.processing = False
        st.rerun()
    
    if st.session_state.processing:
        if st.button("⏸️ Stop Processing", use_container_width=True):
            st.session_state.processing = False
            st.warning("⚠️ Processing stopped by user")
            st.rerun()
    
    if not st.session_state.processing:
        if st.button("🗑️ Clear All Results", use_container_width=True):
            st.session_state.pdf_links = []
            st.session_state.extraction_results = []
            st.session_state.processing = False
            st.rerun()

# Display extraction results
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
                    if isinstance(value, (dict, list)):
                        row[key] = str(value)
                    else:
                        row[key] = value

                # ✅ Explicitly include key fields
                row['University'] = data.get('university', '')
                row['Company_Name'] = data.get('company_name', data.get('company', ''))
                row['Links'] = data.get('links', '')

            elif isinstance(data, list):
                row['Extracted_Data'] = str(data)
            else:
                row['Extracted_Data'] = str(data)
        else:
            row['Error'] = result.get('error', 'Unknown error')
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    st.dataframe(results_df, use_container_width=True, height=400)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"extraction_results_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = results_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download Results JSON",
            data=json_data,
            file_name=f"extraction_results_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        success_df = results_df[results_df['Status'] == '✅ Success']
        if not success_df.empty:
            success_csv = success_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Success Only",
                data=success_csv,
                file_name=f"extraction_success_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    failed_df = results_df[results_df['Status'] == '❌ Error']
    if not failed_df.empty:
        st.divider()
        st.subheader("⚠️ Failed Extractions")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Found {len(failed_df)} failed extractions")
            with st.expander("View Failed URLs"):
                st.dataframe(failed_df[['URL', 'Error']], use_container_width=True)
        with col2:
            failed_csv = failed_df[['URL']].rename(columns={'URL': 'url'}).to_csv(index=False)
            st.download_button(
                label="📥 Download Failed URLs",
                data=failed_csv,
                file_name=f"failed_urls_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with st.expander("🔍 View Raw Extraction Data"):
        st.json(st.session_state.extraction_results)
    
    st.divider()
    st.subheader("📈 Summary Statistics")
    
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

# Footer
st.divider()
st.caption("Powered by LlamaExtract & Streamlit | Sequential processing to ensure stability")
