import pandas as pd
import requests
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import openai
from pydantic import BaseModel
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import traceback
import streamlit as st
from pathlib import Path
import yaml
import pickle
from dataclasses import dataclass
import numpy as np
import io

# test
@dataclass
class TaxonomyConfig:
    """Configuration for taxonomy structure"""
    categories: Dict[str, List[str]]
    descriptions: Dict[str, str]
    hierarchical: bool = False
    hierarchy_mapping: Dict[str, Dict[str, List[str]]] = None


class GenericTagger:
    def __init__(self, perplexity_api_key: str = None, openai_api_key: str = None):
        """
        Initialize the GenericTagger with optional API keys
        
        Args:
            perplexity_api_key: Your Perplexity API key (optional)
            openai_api_key: Your OpenAI API key
        """
        self.perplexity_api_key = perplexity_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.taxonomy = None
        self.checkpoint_dir = Path("tagging_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.backup_dir = Path("tagging_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.file_lock = threading.Lock()  # For thread-safe file operations
        
    def load_taxonomy_from_dict(self, taxonomy_dict: Dict[str, Any]) -> TaxonomyConfig:
        """Load taxonomy from a dictionary structure"""
        categories = taxonomy_dict.get('categories', {})
        descriptions = taxonomy_dict.get('descriptions', {})
        hierarchical = taxonomy_dict.get('hierarchical', False)
        hierarchy_mapping = taxonomy_dict.get('hierarchy_mapping', {})
        
        return TaxonomyConfig(
            categories=categories,
            descriptions=descriptions,
            hierarchical=hierarchical,
            hierarchy_mapping=hierarchy_mapping
        )
    
    def load_taxonomy_from_excel(self, file_path: str, sheet_name: str = None) -> TaxonomyConfig:
        """Load taxonomy from Excel file"""
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        # Try to detect taxonomy structure
        categories = {}
        descriptions = {}
        
        # Check for category columns
        if 'Category' in df.columns and 'Tag' in df.columns:
            for category in df['Category'].unique():
                if pd.notna(category):
                    category_tags = df[df['Category'] == category]['Tag'].dropna().tolist()
                    categories[category] = category_tags
                    
                    # Get descriptions if available
                    if 'Description' in df.columns:
                        for _, row in df[df['Category'] == category].iterrows():
                            if pd.notna(row.get('Description')):
                                descriptions[row['Tag']] = row['Description']
        else:
            # Simple list format
            categories['default'] = df.iloc[:, 0].dropna().tolist()
            if df.shape[1] > 1:
                # Second column might be descriptions
                for i, tag in enumerate(categories['default']):
                    if i < len(df) and pd.notna(df.iloc[i, 1]):
                        descriptions[tag] = df.iloc[i, 1]
        
        return TaxonomyConfig(categories=categories, descriptions=descriptions)
    
    def save_automatic_backup(self, data: pd.DataFrame, backup_type: str = "input"):
        """Save automatic backup of data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"{backup_type}_backup_{timestamp}.xlsx"
        data.to_excel(backup_path, index=False)
        return backup_path
    
    def save_checkpoint(self, results: List[Dict], checkpoint_name: str):
        """Save intermediate results to a checkpoint file in a thread-safe manner"""
        with self.file_lock:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(results, f)
            return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> List[Dict]:
        """Load results from a checkpoint file"""
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    
    def retry_with_exponential_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0, 
                                     entity_name: str = None, progress_callback=None):
        """Retry a function with exponential backoff"""
        last_exception = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                
                # Log retry attempt
                retry_msg = f"Retry {attempt + 1}/{max_retries} for {entity_name if entity_name else 'operation'} in {delay:.1f}s..."
                if progress_callback:
                    progress_callback(retry_msg)
                else:
                    print(retry_msg)
                
                time.sleep(delay)
        raise last_exception
    
    def search_entity_info(self, entity_name: str, entity_url: str = None, 
                        additional_context: str = "", max_retries: int = 3,
                        progress_callback=None, custom_prompt: str = None,
                        include_sources: bool = True) -> Tuple[str, bool]:
        """
        Search for entity information using Perplexity API
        
        Args:
            entity_name: Name of the entity to search for
            entity_url: Optional URL for the entity
            additional_context: Additional context for the search
            max_retries: Maximum number of retry attempts
            progress_callback: Optional callback for progress updates
            custom_prompt: Optional custom prompt to guide the search
            include_sources: Whether to include source citations in the output
        
        Returns:
            Tuple of (description, success_flag)
        """
        if not self.perplexity_api_key:
            return f"No search performed for {entity_name} - Perplexity API key not provided", False
        
        def _search():
            # Build the query based on whether we have a custom prompt
            if custom_prompt:
                # Include the custom prompt to guide Perplexity's search
                query = f"""You are searching for information about {entity_name} to help answer the following analysis prompt:

ANALYSIS PROMPT: {custom_prompt}

Please find and provide information about {entity_name} that would be most relevant for answering the above prompt. Focus on:
1. Information directly relevant to the analysis prompt
2. Key facts and data points needed for the classification/analysis
3. Any specific aspects mentioned in the prompt

{additional_context}"""
            else:
                # Original generic query
                query = f"""Analyze {entity_name} and provide:
1. What they do or offer
2. Key characteristics and attributes
3. Industry or domain they operate in
{additional_context}"""
            
            # Add source citation instruction
            if include_sources:
                query += "\n\nIMPORTANT: Please cite your sources by including [Source: URL or source name] after each key fact or piece of information."
            
            url = "https://api.perplexity.ai/chat/completions"
            
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Updated system prompt to emphasize source citations
            system_content = "You are an analyst. Provide detailed, factual descriptions based on available information. Focus on information that would be useful for the specific analysis requested."
            if include_sources:
                system_content += " Always cite your sources by including [Source: URL or source name] after each fact or claim."
            
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 700
            }
            
            # Add domain filter if URL provided
            if entity_url:
                clean_domain = entity_url.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
                payload["search_domain_filter"] = [clean_domain]
            
            response = requests.post(url, headers=headers, json=payload)
            
            # Check for rate limiting
            if response.status_code == 429:
                raise Exception("Perplexity API rate limit reached (429)")
            
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        try:
            description = self.retry_with_exponential_backoff(
                _search, 
                max_retries=max_retries,
                entity_name=entity_name,
                progress_callback=progress_callback
            )
            time.sleep(0.05)  # Reduced from 0.5 for faster processing
            return description, True
        except Exception as e:
            error_msg = f"Error retrieving information for {entity_name}: {str(e)}"
            return error_msg, False
    
    def select_tags_with_ai(self, description: str, entity_name: str, 
                           available_tags: List[str], tag_descriptions: Dict[str, str],
                           multi_select: bool = False, existing_data: Dict = None,
                           custom_prompt: str = None) -> Dict:
        """Use AI to select appropriate tags or classify based on prompt using Responses API"""
        if not self.openai_client:
            return {
                'status': 'error',
                'error': 'OpenAI client not initialized'
            }
        
        def _select_tags():
            # If custom prompt is provided (no taxonomy mode)
            if custom_prompt and not available_tags:
                # Add existing data context if provided
                context = ""
                if existing_data:
                    context = "\n\nAdditional context from data:\n"
                    context += "\n".join([f"{k}: {v}" for k, v in existing_data.items() if v])
                
                input_content = f"{custom_prompt}\n\nEntity: {entity_name}\nDescription: {description}{context}"
                
                # Use the responses API for custom prompts
                response = self.openai_client.responses.create(
                    model="gpt-4o-2024-08-06",
                    input=input_content
                )
                
                # For custom prompts, return the raw response
                return {
                    'status': 'success',
                    'result': response.output_text,
                    'confidence': 1.0,
                    'reasoning': 'Custom prompt classification'
                }
            
            # For taxonomy-based selection, use structured outputs with responses.parse
            # Build tag descriptions for the prompt
            tags_desc = "\n".join([
                f"- {tag}: {tag_descriptions.get(tag, 'No description available')}"
                for tag in available_tags
            ])
            
            # Add existing data context if provided
            context = ""
            if existing_data:
                context = "\n\nAdditional context from data:\n"
                context += "\n".join([f"{k}: {v}" for k, v in existing_data.items() if v])
            
            if multi_select:
                # Create structured output class for multi-select
                class MultiTagOutput(BaseModel):
                    primary_tag: str
                    secondary_tags: List[str]
                    confidence: float
                    reasoning: str
                
                system_content = f"""You are an expert at classifying entities based on the following taxonomy.
Select multiple tags if appropriate, with one primary and optional secondary tags.

Available tags:
{tags_desc}

Ensure your primary_tag and all secondary_tags are from the available tags list above."""
                
                user_content = f"Entity: {entity_name}\nDescription: {description}{context}"
                
                # Use the responses.parse API with structured outputs
                response = self.openai_client.responses.parse(
                    model="gpt-4o-2024-08-06",
                    input=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    text_format=MultiTagOutput
                )
                
                parsed = response.output_parsed
                
                # Clean up parsed tags to remove any trailing commas
                if hasattr(parsed, 'primary_tag'):
                    parsed.primary_tag = parsed.primary_tag.strip().rstrip(',')
                if hasattr(parsed, 'secondary_tags'):
                    parsed.secondary_tags = [tag.strip().rstrip(',') for tag in parsed.secondary_tags]
                
                return {
                    'status': 'success',
                    'primary_tag': parsed.primary_tag,
                    'secondary_tags': parsed.secondary_tags,
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning
                }
            else:
                # Create structured output class for single tag
                class SingleTagOutput(BaseModel):
                    selected_tag: str
                    confidence: float
                    reasoning: str
                
                system_content = f"""You are an expert at classifying entities based on the following taxonomy.
Select the single most appropriate tag.

Available tags:
{tags_desc}

Ensure your selected_tag is from the available tags list above."""
                
                user_content = f"Entity: {entity_name}\nDescription: {description}{context}"
                
                # Use the responses.parse API with structured outputs
                response = self.openai_client.responses.parse(
                    model="gpt-4o-2024-08-06",
                    input=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    text_format=SingleTagOutput
                )
                
                parsed = response.output_parsed
                
                # Clean up the parsed tag to remove any trailing commas
                if hasattr(parsed, 'selected_tag'):
                    parsed.selected_tag = parsed.selected_tag.strip().rstrip(',')
                
                return {
                    'status': 'success',
                    'tag': parsed.selected_tag,
                    'confidence': parsed.confidence,
                    'reasoning': parsed.reasoning
                }
        
        try:
            return self.retry_with_exponential_backoff(_select_tags)
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def process_single_entity(self, row_data: Dict, config: Dict, progress_callback=None) -> Dict:
        """Process a single entity based on configuration"""
        try:
            entity_name = row_data.get(config['name_column'], 'Unknown')
            
            # Handle multiple custom queries
            custom_queries = config.get('custom_queries', [])
            
            # If using multiple queries mode
            if custom_queries:
                result = row_data.copy()
                
                # Process each custom query
                for query_config in custom_queries:
                    query_name = query_config['name']
                    query_prompt = query_config['prompt']
                    use_search = query_config.get('use_search', config.get('use_search', False))
                    
                    # Step 1: Get description (from search or existing data)
                    search_success = True
                    if use_search and self.perplexity_api_key:
                        url_column = config.get('url_column')
                        entity_url = row_data.get(url_column) if url_column else None
                        
                        # Get retry settings from config
                        max_retries = config.get('search_max_retries', 3)
                        
                        # Pass custom prompt to search function with source citation enabled
                        description, search_success = self.search_entity_info(
                            entity_name, 
                            entity_url,
                            max_retries=max_retries,
                            progress_callback=progress_callback,
                            custom_prompt=query_prompt,
                            include_sources=True  # Enable source citations
                        )
                        
                        # Store search description for this query
                        result[f'{query_name}_Search_Description'] = description[:500] if search_success else description
                        
                        # If search failed, mark this query as failed
                        if not search_success:
                            result[f'{query_name}_Result'] = 'Search Failed'
                            result[f'{query_name}_Confidence'] = '0%'
                            result[f'{query_name}_Reasoning'] = 'Perplexity search failed - cannot analyze without description'
                            result[f'{query_name}_Status'] = 'Search Error'
                            continue
                    else:
                        # Use existing description columns if specified
                        desc_columns = config.get('description_columns', [])
                        description_parts = []
                        for col in desc_columns:
                            if col in row_data and row_data[col]:
                                description_parts.append(f"{col}: {row_data[col]}")
                        description = "\n".join(description_parts) if description_parts else f"No description available for {entity_name}"
                    
                    # Step 2: Prepare context from other columns
                    context_columns = config.get('context_columns', [])
                    context_data = {col: row_data.get(col) for col in context_columns if col in row_data}
                    
                    # Step 3: AI analysis for this query
                    tag_result = self.select_tags_with_ai(
                        description=description,
                        entity_name=entity_name,
                        available_tags=[],  # No taxonomy for custom queries
                        tag_descriptions={},
                        multi_select=False,
                        existing_data=context_data,
                        custom_prompt=query_prompt
                    )
                    
                    # Step 4: Store results for this query
                    if tag_result['status'] == 'error':
                        result[f'{query_name}_Result'] = 'Error'
                        result[f'{query_name}_Confidence'] = '0%'
                        result[f'{query_name}_Reasoning'] = tag_result['error']
                        result[f'{query_name}_Status'] = 'Error'
                    else:
                        result[f'{query_name}_Result'] = tag_result['result']
                        result[f'{query_name}_Confidence'] = f"{tag_result['confidence']:.0%}"
                        result[f'{query_name}_Reasoning'] = tag_result['reasoning']
                        result[f'{query_name}_Status'] = 'Success'
                
                return result
            
            # Original single query/taxonomy mode
            else:
                # Get custom prompt if available
                custom_prompt = config.get('custom_prompt', None)
                
                # Step 1: Get description (from search or existing data)
                search_success = True
                if config['use_search'] and self.perplexity_api_key:
                    url_column = config.get('url_column')
                    entity_url = row_data.get(url_column) if url_column else None
                    
                    # Get retry settings from config
                    max_retries = config.get('search_max_retries', 3)
                    
                    # Pass custom prompt to search function with source citation enabled
                    description, search_success = self.search_entity_info(
                        entity_name, 
                        entity_url,
                        max_retries=max_retries,
                        progress_callback=progress_callback,
                        custom_prompt=custom_prompt,
                        include_sources=True  # Enable source citations
                    )
                    
                    # If search failed (e.g., 429 error), don't proceed with tagging
                    if not search_success:
                        result = row_data.copy()
                        result.update({
                            'Search_Description': description,
                            'Tagged_Result': 'Search Failed',
                            'Confidence': '0%',
                            'Reasoning': 'Perplexity search failed - cannot tag without description',
                            'Status': 'Search Error'
                        })
                        return result
                else:
                    # Use existing description columns if specified
                    desc_columns = config.get('description_columns', [])
                    description_parts = []
                    for col in desc_columns:
                        if col in row_data and row_data[col]:
                            description_parts.append(f"{col}: {row_data[col]}")
                    description = "\n".join(description_parts) if description_parts else f"No description available for {entity_name}"
                
                # Step 2: Prepare context from other columns
                context_columns = config.get('context_columns', [])
                context_data = {col: row_data.get(col) for col in context_columns if col in row_data}
                
                # Step 3: Determine if using taxonomy or custom prompt
                use_taxonomy = config.get('use_taxonomy', True)
                custom_prompt = config.get('custom_prompt', None)
                
                if use_taxonomy and self.taxonomy:
                    # Get available tags based on category if specified
                    if config.get('category_column') and self.taxonomy.categories:
                        category = row_data.get(config['category_column'], 'default')
                        available_tags = self.taxonomy.categories.get(category, 
                                                                    self.taxonomy.categories.get('default', []))
                    else:
                        # Use all tags
                        all_tags = []
                        for tags in self.taxonomy.categories.values():
                            all_tags.extend(tags)
                        available_tags = list(set(all_tags))
                    
                    tag_descriptions = self.taxonomy.descriptions
                else:
                    # No taxonomy - use custom prompt
                    available_tags = []
                    tag_descriptions = {}
                
                # Step 4: AI tagging
                tag_result = self.select_tags_with_ai(
                    description=description,
                    entity_name=entity_name,
                    available_tags=available_tags,
                    tag_descriptions=tag_descriptions,
                    multi_select=config.get('multi_select', False),
                    existing_data=context_data,
                    custom_prompt=custom_prompt
                )
                
                # Step 5: Compile results
                result = row_data.copy()
                
                if config['use_search']:
                    result['Search_Description'] = description[:500]
                
                if tag_result['status'] == 'error':
                    result.update({
                        'Tagged_Result': 'Error',
                        'Confidence': '0%',
                        'Reasoning': tag_result['error'],
                        'Status': 'Error'
                    })
                else:
                    # Handle custom prompt results
                    if 'result' in tag_result:
                        result.update({
                            'Tagged_Result': tag_result['result'],
                            'Confidence': f"{tag_result['confidence']:.0%}",
                            'Reasoning': tag_result['reasoning'],
                            'Status': 'Success'
                        })
                    elif config.get('multi_select'):
                        # Clean up secondary tags - ensure they're a proper list
                        secondary_tags = tag_result.get('secondary_tags', [])
                        if isinstance(secondary_tags, list):
                            # Join with semicolon or another delimiter to avoid confusion
                            secondary_tags_str = '; '.join(secondary_tags) if secondary_tags else ''
                        else:
                            secondary_tags_str = str(secondary_tags)
                        
                        result.update({
                            'Primary_Tag': tag_result['primary_tag'],
                            'Secondary_Tags': secondary_tags_str,
                            'Confidence': f"{tag_result['confidence']:.0%}",
                            'Reasoning': tag_result['reasoning'],
                            'Status': 'Success'
                        })
                    else:
                        # Single tag selection - ensure clean output
                        tag_value = tag_result.get('tag', '')
                        # Clean any trailing commas or whitespace
                        if isinstance(tag_value, str):
                            tag_value = tag_value.strip().rstrip(',')
                        
                        result.update({
                            'Tagged_Result': tag_value,
                            'Confidence': f"{tag_result['confidence']:.0%}",
                            'Reasoning': tag_result['reasoning'],
                            'Status': 'Success'
                        })
                
                return result
            
        except Exception as e:
            result = row_data.copy()
            result.update({
                'Tagged_Result': 'Error',
                'Confidence': '0%',
                'Reasoning': str(e),
                'Status': 'Error'
            })
            return result


# Pre-configured prompts
PRESET_PROMPTS = {
    "Industry Classification": """You are an expert at classifying companies by industry. Analyze the provided information and classify the entity into one of these industries: Technology, Healthcare, Finance, Retail, Manufacturing, Services, Education, Real Estate, Energy, Transportation, Media/Entertainment, or Other. Provide your answer in the format: "Industry: [classification]" followed by a brief explanation.""",
    
    "Company Size": """Based on the provided information, classify this company's size as: Startup (1-50 employees), Small (51-200), Medium (201-1000), Large (1001-5000), or Enterprise (5000+). Provide your answer in the format: "Size: [classification]" with reasoning.""",
    
    "B2B vs B2C": """Analyze the entity and determine if it primarily serves Business customers (B2B), Consumer customers (B2C), or Both (B2B/B2C). Provide your answer in the format: "Type: [classification]" with explanation.""",
    
    "Technology Level": """Assess the technology sophistication of this entity. Classify as: High-Tech (cutting-edge technology focus), Tech-Enabled (uses technology significantly), Traditional Tech (basic technology use), or Non-Tech. Format: "Tech Level: [classification]" with reasoning.""",
    
    "Market Maturity": """Evaluate the entity's market position and maturity. Classify as: Early Stage (new/unproven), Growth Stage (expanding rapidly), Mature (established market position), or Declining. Format: "Maturity: [classification]" with explanation.""",
    
    "Custom Analysis": """[This is a placeholder - users should write their own prompt for custom analysis]"""
}


def create_tag_categories_helper(key_suffix=""):
    """Create the 'Select this if...' helper text and popup functionality"""
    # Initialize session state for popup
    popup_key = f"show_categories_popup_{key_suffix}"
    if popup_key not in st.session_state:
        st.session_state[popup_key] = False
    
    # Create clickable text
    if st.button("*Select this if...*", key=f"categories_help_{key_suffix}", 
                help="Click for explanation of tag categories",
                type="secondary"):
        st.session_state[popup_key] = True
    
    # Show popup if activated
    if st.session_state[popup_key]:
        with st.expander("Tag Categories Explanation", expanded=True):
            col1, col2 = st.columns([10, 1])
            with col2:
                if st.button("✕", key=f"close_popup_{key_suffix}", help="Close"):
                    st.session_state[popup_key] = False
                    st.rerun()
            
            with col1:
                st.markdown("""
                <div style="font-size: 16px; color: #333; margin-bottom: 15px;">
                Select this if your tags are grouped by category. Each entity will be restricted to the tags in its category. E.g.
                </div>
                """, unsafe_allow_html=True)
                
                # First table - Categorized tags with smaller text
                categorized_data = {
                    "Category": ["Technology", "Technology", "Healthcare", "Healthcare", "Finance", "Finance"],
                    "Tag": ["Software", "Hardware", "Pharmaceutical", "Medical Devices", "Banking", "Fintech"],
                    "Description": [
                        "Companies that develop software products",
                        "Companies that manufacture hardware", 
                        "Drug development and manufacturing",
                        "Medical equipment manufacturers",
                        "Traditional banking services",
                        "Financial technology companies"
                    ]
                }
                categorized_df = pd.DataFrame(categorized_data)
                
                # Apply styling to make table text smaller
                st.markdown("""
                <style>
                .small-table table {
                    font-size: 12px !important;
                }
                .small-table td, .small-table th {
                    font-size: 12px !important;
                    padding: 4px 8px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="small-table">', unsafe_allow_html=True)
                st.table(categorized_df)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div style="font-size: 16px; color: #333; margin: 15px 0;">
                As opposed to the simpler:
                </div>
                """, unsafe_allow_html=True)
                
                # Second table - Simple tags with smaller text
                simple_data = {
                    "Tag": ["Software", "Hardware", "Consulting"],
                    "Description": [
                        "Companies that develop software",
                        "Companies that manufacture hardware",
                        "Professional services companies"
                    ]
                }
                simple_df = pd.DataFrame(simple_data)
                
                st.markdown('<div class="small-table">', unsafe_allow_html=True)
                st.table(simple_df)
                st.markdown('</div>', unsafe_allow_html=True)


def create_streamlit_app():
    """Create the Streamlit user interface"""
    st.set_page_config(page_title="Generic Tagging Tool", layout="wide")
    
    st.title("🏷️ Tagging Tool")
    st.markdown("Tag any entities using AI with customizable taxonomies or prompts")
    
    # User Guide - expandable section (always visible at top)
    with st.expander("📚 Quick Start Guide (5 steps)", expanded=False):
        st.markdown("""
        ### 1. **Initialize** in the sidebar
        Enter API keys. Enter Perplexity key only if using Web Search.
        
        ### 2. **Choose a tagging method**
        - **Use Taxonomy** → If you have a complete list of tags, provide tags via **Upload Excel**, **Paste YAML/JSON**, or **Enter manually**.
          - *Tip: if you set a **Category column**, the tool will only use tags from that category.*
          - Toggle **Allow multiple tags** for primary + secondary tags.
        - **Use Custom Prompt** → pick a preset or write your own instruction.
        - **Use Multiple Custom Queries** → add several named prompts; each creates its own result columns.
        
        ### 3. **Upload input data**
        CSV or Excel, formatted with labeled columns starting in cell A1; pick a sheet if prompted.
        
        ### 4. **Map columns**
        - **Entity name column**: the entity to classify.
        - **URL column (optional)**: if using web search and you have URLs you want to search, e.g., company websites
        - **Description columns**: if not using web search -- the main text the model reads about each entity.
        - **Context columns (optional)**: extra fields (e.g., country, sector) the model can reference.
        - **Category column (optional)**: ties rows to taxonomy categories.
        
        ### 5. **Run**
        In **Processing Options** set threads, checkpoint **batch size**, and (if searching) **retry settings**. 
        Select **all rows**, filter by **category**, or pick a **row range** → **Start Tagging**.
        """)
    
    # Initialize session state
    if 'tagger' not in st.session_state:
        st.session_state.tagger = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'config' not in st.session_state:
        st.session_state.config = {}
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'custom_queries' not in st.session_state:
        st.session_state.custom_queries = []
    
    # Sidebar for API Keys only
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        with st.expander("🔑 API Keys", expanded=True):
            openai_key = st.text_input("OpenAI API Key", type="password", 
                                      help="Required for AI tagging")
            perplexity_key = st.text_input("Perplexity API Key", type="password",
                                         help="Optional - only needed if using web search")
            
            if st.button("Initialize Tagger"):
                if openai_key:
                    st.session_state.tagger = GenericTagger(
                        perplexity_api_key=perplexity_key if perplexity_key else None,
                        openai_api_key=openai_key
                    )
                    st.success("✅ Tagger initialized!")
                else:
                    st.error("OpenAI API key is required")
        
        # Checkpoint management (moved here to keep sidebar functionality)
        if st.session_state.tagger:
            st.header("💾 Checkpoints & Backups")
            
            # Backup management
            with st.expander("📁 Backups"):
                backup_files = list(st.session_state.tagger.backup_dir.glob("*.xlsx"))
                
                if backup_files:
                    st.write(f"Found {len(backup_files)} backup(s)")
                    
                    # Show recent backups
                    recent_backups = sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
                    for backup in recent_backups:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(backup.name)
                        with col2:
                            file_size = backup.stat().st_size / 1024  # KB
                            st.text(f"{file_size:.1f} KB")
                    
                    if st.button("Clear Old Backups"):
                        # Keep only last 5 backups
                        for backup in backup_files[5:]:
                            backup.unlink()
                        st.success("Old backups cleared")
                else:
                    st.info("No backups yet")
            
            # Checkpoint management
            with st.expander("💾 Checkpoints"):
                checkpoint_files = list(st.session_state.tagger.checkpoint_dir.glob("*.pkl"))
                
                if checkpoint_files:
                    selected_checkpoint = st.selectbox(
                        "Load checkpoint",
                        ["None"] + [f.name for f in checkpoint_files]
                    )
                    
                    if selected_checkpoint != "None":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Load Checkpoint"):
                                checkpoint_path = st.session_state.tagger.checkpoint_dir / selected_checkpoint
                                loaded_results = st.session_state.tagger.load_checkpoint(checkpoint_path)
                                st.session_state.results = loaded_results
                                st.success(f"Loaded {len(loaded_results)} results")
                                
                                # Show what was loaded
                                if loaded_results and 'df' in st.session_state and st.session_state.df is not None and 'config' in st.session_state:
                                    name_column = st.session_state.config.get('name_column')
                                    if name_column:
                                        processed_entities = {r.get(name_column) for r in loaded_results if r.get(name_column)}
                                        total_rows = len(st.session_state.df)
                                        processed_rows = len(processed_entities)
                                        remaining_rows = total_rows - processed_rows
                                        
                                        st.info(f"""
                                        📊 Checkpoint Status:
                                        - Processed: {processed_rows} rows
                                        - Remaining: {remaining_rows} rows
                                        - Total: {total_rows} rows
                                        """)
                                        
                                        # Store the remaining rows
                                        remaining_mask = ~st.session_state.df[name_column].isin(processed_entities)
                                        st.session_state.remaining_df = st.session_state.df[remaining_mask]
                                        st.session_state.checkpoint_loaded = True
                                        st.session_state.existing_results = loaded_results
                        
                        with col2:
                            if hasattr(st.session_state, 'checkpoint_loaded') and st.session_state.checkpoint_loaded:
                                if st.button("Resume Processing", type="primary"):
                                    # This will trigger the main processing section
                                    st.session_state.resume_processing = True
                                    st.rerun()  # Force a rerun to update the UI
                
                if st.button("Clear All Checkpoints"):
                    for f in checkpoint_files:
                        f.unlink()
                    st.success("All checkpoints cleared")
    
    # Main area with tabs - SWAPPED ORDER
    if st.session_state.tagger:
        # Create tabs - now with reordered tabs
        tab1, tab2, tab3 = st.tabs(["Data Input and Column Configuration", "Tagging Method and Configuration", "Start Tagging"])
        
        with tab1:
            # Data Input and Column Configuration (moved from tab2)
            st.header("📁 Data Input")
            
            # Add explanatory text in smaller font
            st.markdown("""
            <div style="font-size: 14px; color: #666; margin-bottom: 20px;">
            Input one Excel file containing your entities to be tagged (and any other additional columns of information), formatted with column titles starting in cell A1.
            <br><br>
            <strong>Data Input columns:</strong>
            <ul>
                <li><strong>Entity to be tagged</strong> (required)</li>
                <li><strong>'Description' column:</strong> Only applicable if not using Web Search; the main information the model reads about each entity to inform the tagging</li>
                <li><strong>'URL' column:</strong> Optional, only applicable if using Web Search and you have specific websites you want to search, e.g. company websites</li>
                <li><strong>'Context' column(s):</strong> Optional, extra fields (e.g. country, sector) the model can reference</li>
                <li><strong>Category column:</strong> Only applicable if using tag categories, i.e. if tags and entities are assigned to categories (e.g. Technology) such that an entity's tag options are restricted to the tags within that category (e.g. Software, Hardware)</li>
            </ul>
            The tool will output your original file with an added column containing tags (will add multiple columns containing tags if you use multiple custom queries).
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose Excel/CSV file", 
                                           type=['xlsx', 'xls', 'csv'])
            
            if uploaded_file:
                # Load file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    sheet_names = ['main']
                else:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    
                    selected_sheet = st.selectbox("Select sheet", sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                
                st.session_state.df = df
                st.success(f"Loaded {len(df)} rows")
                
                # Auto-backup input file
                if st.session_state.tagger:
                    backup_path = st.session_state.tagger.save_automatic_backup(df, "input")
                    st.info(f"📁 Input backup saved: {backup_path.name}")
                
                # Column selection
                st.header("📊 Column Configuration")
                columns = df.columns.tolist()
                
                name_column = st.selectbox("Entity name column", columns,
                                          help="Column containing the names to tag")
                
                use_search = st.checkbox("Use web search (Perplexity)", 
                                       value=False,
                                       disabled=not perplexity_key)
                
                url_column = None
                description_columns = []
                
                if use_search:
                    url_column = st.selectbox("URL column (optional)", 
                                            ['None'] + columns)
                    url_column = None if url_column == 'None' else url_column
                else:
                    description_columns = st.multiselect(
                        "Description columns",
                        columns,
                        help="Columns containing descriptions or relevant text"
                    )
                
                context_columns = st.multiselect(
                    "Context columns (optional)",
                    [c for c in columns if c != name_column],
                    help="Additional columns to provide context for tagging"
                )
                
                category_column = st.selectbox(
                    "Category column (optional)",
                    ['None'] + columns,
                    help="If your taxonomy has categories"
                )
                category_column = None if category_column == 'None' else category_column
                
                # Save configuration
                st.session_state.config = {
                    'name_column': name_column,
                    'use_search': use_search,
                    'url_column': url_column,
                    'description_columns': description_columns,
                    'context_columns': context_columns,
                    'category_column': category_column,
                    'multi_select': False  # Will be set based on taxonomy
                }
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.info(f"Showing first 10 rows of {len(df)} total rows")

        with tab2:
            # Tagging Method and Configuration (moved from tab1)
            # Choose tagging method
            st.header("🎯 Tagging Method")
            
            # Use dropdown instead of radio buttons
            tagging_method = st.selectbox(
                "Choose tagging method:",
                ["Use Taxonomy", "Use Custom Prompt", "Use Multiple Custom Queries"],
                help="Select how you want to classify your entities"
            )
            
            if tagging_method == "Use Taxonomy":
                # Only update config if file has been loaded
                if 'df' in st.session_state and st.session_state.df is not None:
                    st.session_state.config['use_taxonomy'] = True
                    st.session_state.config['custom_queries'] = []  # Clear multiple queries
                
                # Updated layout with checkbox moved left and button right next to it
                col1, col2, col3 = st.columns([2, 2, 6])
                with col1:
                    st.subheader("📋 Taxonomy Setup")
                with col2:
                    use_tag_categories_taxonomy = st.checkbox("Use tag categories", key="categories_taxonomy")
                with col3:
                    create_tag_categories_helper("taxonomy")
                
                taxonomy_method = st.selectbox(
                    "How would you like to provide the taxonomy?",
                    ["Upload Excel file", "Paste as text", "Enter manually"]
                )
                
                if taxonomy_method == "Upload Excel file":
                    taxonomy_file = st.file_uploader("Upload taxonomy file", 
                                                type=['xlsx', 'xls'],
                                                key="taxonomy_upload")
                    if taxonomy_file:
                        tax_sheet = None
                        if taxonomy_file.name.endswith(('.xlsx', '.xls')):
                            tax_excel = pd.ExcelFile(taxonomy_file)
                            if len(tax_excel.sheet_names) > 1:
                                tax_sheet = st.selectbox("Select taxonomy sheet", 
                                                        tax_excel.sheet_names)
                        
                        taxonomy = st.session_state.tagger.load_taxonomy_from_excel(
                            taxonomy_file, tax_sheet)
                        st.session_state.tagger.taxonomy = taxonomy
                        st.success("✅ Taxonomy loaded!")
                        
                        # Display taxonomy
                        with st.expander("View loaded taxonomy"):
                            for category, tags in taxonomy.categories.items():
                                st.write(f"**{category}**")
                                for tag in tags:
                                    desc = taxonomy.descriptions.get(tag, "")
                                    st.write(f"- {tag}: {desc}" if desc else f"- {tag}")
                    
                    # Excel taxonomy formatting section
                    st.subheader("📋 Excel Taxonomy Formatting")
                    
                    if use_tag_categories_taxonomy:
                        # When using tag categories
                        st.markdown("Starting in cell A1, three columns: **Category**, **Tag**, and **Description** of tag.")
                        
                        # Show categorized example table
                        categorized_data = {
                            "Category": ["Technology", "Technology", "Healthcare", "Healthcare", "Finance", "Finance"],
                            "Tag": ["Software", "Hardware", "Pharmaceutical", "Medical Devices", "Banking", "Fintech"],
                            "Description": [
                                "Companies that develop software products",
                                "Companies that manufacture hardware", 
                                "Drug development and manufacturing",
                                "Medical equipment manufacturers",
                                "Traditional banking services",
                                "Financial technology companies"
                            ]
                        }
                        categorized_df = pd.DataFrame(categorized_data)
                        
                        # Apply styling to make table text smaller
                        st.markdown("""
                        <style>
                        .small-table table {
                            font-size: 12px !important;
                        }
                        .small-table td, .small-table th {
                            font-size: 12px !important;
                            padding: 4px 8px !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="small-table">', unsafe_allow_html=True)
                        st.table(categorized_df)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    else:
                        # When NOT using tag categories
                        st.markdown("Starting in cell A1, two columns: **Tag** and **Description** of tag.")
                        
                        # Show simple example table
                        simple_data = {
                            "Tag": ["Software", "Hardware", "Consulting"],
                            "Description": [
                                "Companies that develop software",
                                "Companies that manufacture hardware",
                                "Professional services companies"
                            ]
                        }
                        simple_df = pd.DataFrame(simple_data)
                        
                        # Apply styling to make table text smaller
                        st.markdown("""
                        <style>
                        .small-table table {
                            font-size: 12px !important;
                        }
                        .small-table td, .small-table th {
                            font-size: 12px !important;
                            padding: 4px 8px !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="small-table">', unsafe_allow_html=True)
                        st.table(simple_df)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                elif taxonomy_method == "Paste as text":
                    st.markdown("Paste your taxonomy in YAML format:")
                    
                    # Conditional label and placeholder based on tag categories checkbox
                    if use_tag_categories_taxonomy:
                        # When using tag categories
                        taxonomy_label = "Taxonomy"
                        taxonomy_placeholder = """categories:
  Category1:
    - Tag1
    - Tag2
  Category2:
    - Tag3
    - Tag4
  Category3:
    - Tag5
    - Tag6
descriptions:
  Tag1: "Description of Tag1"
  Tag2: "Description of Tag2"
  Tag3: "Description of Tag3"
  Tag4: "Description of Tag4"
  Tag5: "Description of Tag5"
  Tag6: "Description of Tag6"
"""
                    else:
                        # When NOT using tag categories
                        taxonomy_label = "Taxonomy\n\nSince tags are not categorized, there is one category of tags called 'default'."
                        taxonomy_placeholder = """categories:
  default:
    - Tag1
    - Tag2
    - Tag3
descriptions:
  Tag1: "Description of Tag1"
  Tag2: "Description of Tag2"
  Tag3: "Description of Tag3"
"""
                    
                    taxonomy_text = st.text_area(taxonomy_label, height=300, 
                                            placeholder=taxonomy_placeholder)
                    
                    if st.button("Parse Taxonomy"):
                        try:
                            # Parse YAML only
                            taxonomy_dict = yaml.safe_load(taxonomy_text)
                            taxonomy = st.session_state.tagger.load_taxonomy_from_dict(taxonomy_dict)
                            st.session_state.tagger.taxonomy = taxonomy
                            st.success("✅ Taxonomy parsed successfully!")
                        except Exception as e:
                            st.error(f"Failed to parse YAML taxonomy: {str(e)}")
                
                else:  # Manual entry
                    st.markdown("Enter tags manually:")
                    
                    categories = {}
                    descriptions = {}
                    
                    if use_tag_categories_taxonomy:
                        # When using tag categories - show full interface
                        num_categories = st.number_input("Number of categories", 1, 10, 1)
                        
                        for i in range(num_categories):
                            with st.expander(f"Category {i+1}", expanded=i==0):
                                cat_name = st.text_input(f"Category name", "default", key=f"cat_{i}")
                                tags_text = st.text_area(
                                    "Tags (one per line)", 
                                    key=f"tags_{i}",
                                    help="Enter each tag on a new line"
                                )
                                st.markdown("*You will enter descriptions for each tag after entering the list of tags themselves and clicking \"Create Taxonomy\".*")
                                
                                if tags_text:
                                    tags = [t.strip() for t in tags_text.split('\n') if t.strip()]
                                    categories[cat_name] = tags
                                    
                                    # Optional descriptions
                                    if st.checkbox("Add descriptions", key=f"desc_check_{i}"):
                                        for tag in tags:
                                            desc = st.text_input(f"Description for '{tag}'", 
                                                            key=f"desc_{i}_{tag}")
                                            if desc:
                                                descriptions[tag] = desc
                    else:
                        # When NOT using tag categories - simplified interface
                        tags_text = st.text_area(
                            "Tags (one per line)", 
                            key="tags_simple",
                            help="Enter each tag on a new line"
                        )
                        st.markdown("*You will enter descriptions for each tag after entering the list of tags themselves and clicking \"Create Taxonomy\".*")
                        
                        if tags_text:
                            tags = [t.strip() for t in tags_text.split('\n') if t.strip()]
                            categories["default"] = tags
                            
                            # Optional descriptions
                            if st.checkbox("Add descriptions", key="desc_check_simple"):
                                for tag in tags:
                                    desc = st.text_input(f"Description for '{tag}'", 
                                                    key=f"desc_simple_{tag}")
                                    if desc:
                                        descriptions[tag] = desc
                    
                    if st.button("Create Taxonomy"):
                        taxonomy = TaxonomyConfig(
                            categories=categories,
                            descriptions=descriptions
                        )
                        st.session_state.tagger.taxonomy = taxonomy
                        st.success("✅ Taxonomy created!")
            
            elif tagging_method == "Use Custom Prompt":
                # Only update config if file has been loaded
                if 'df' in st.session_state and st.session_state.df is not None:
                    st.session_state.config['use_taxonomy'] = False
                    st.session_state.config['custom_queries'] = []  # Clear multiple queries
                
                st.subheader("✍️ Prompt Configuration")
                st.markdown("<small>Custom queries method is recommended only if required, i.e. if you do not have a complete list of tags for your task. If you have a list of tags, Taxonomy method is recommended for optimal results.</small>", unsafe_allow_html=True)
                
                # Preset prompts
                preset_choice = st.selectbox(
                    "Choose a preset prompt or create custom:",
                    list(PRESET_PROMPTS.keys())
                )
                
                if preset_choice == "Custom Analysis":
                    custom_prompt = st.text_area(
                        "Enter your custom prompt:",
                        height=200,
                        placeholder="""Example: You are an expert analyst. Based on the entity description, determine if this is a high-growth potential company. Consider factors like innovation, market opportunity, and scalability. Respond with "High Growth", "Moderate Growth", or "Low Growth" followed by your reasoning."""
                    )
                else:
                    custom_prompt = st.text_area(
                        "Prompt (you can edit):",
                        value=PRESET_PROMPTS[preset_choice],
                        height=200
                    )
                
                st.session_state.config['custom_prompt'] = custom_prompt
                
                # Show example
                with st.expander("💡 Prompt Tips"):
                    st.markdown("""
                    **Tips for writing effective prompts:**
                    - Be specific about what you want to classify or analyze
                    - Define clear categories or output format
                    - Include any specific criteria for classification
                    - Ask for reasoning to understand the AI's decision
                    
                    **Example format:**
                    "You are an expert at [domain]. Analyze the entity and classify it as [categories]. 
                    Consider [specific factors]. Provide your answer as '[Output format]' with explanation."
                    """)
            
            else:  # Use Multiple Custom Queries
                # Only update config if file has been loaded
                if 'df' in st.session_state and st.session_state.df is not None:
                    st.session_state.config['use_taxonomy'] = False
                    st.session_state.config['custom_prompt'] = None  # Clear single prompt
                
                st.subheader("✍️ Multiple Query Configuration")
                st.markdown("<small>Custom queries method is recommended only if required, i.e. if you do not have a complete list of tags for your task. If you have a list of tags, Taxonomy method is recommended for optimal results.</small>", unsafe_allow_html=True)
                
                st.info("Configure multiple custom queries to run on each entity. Each query will create separate result columns.")
                
                # Add new query
                with st.expander("➕ Add New Query", expanded=True):
                    query_name = st.text_input("Query Name", 
                                            placeholder="e.g., Industry_Analysis",
                                            help="This will be used as the prefix for result columns")
                    
                    # Preset or custom
                    use_preset = st.checkbox("Use preset prompt")
                    
                    if use_preset:
                        preset_choice = st.selectbox(
                            "Choose preset:",
                            list(PRESET_PROMPTS.keys()),
                            key="preset_multi"
                        )
                        if preset_choice == "Custom Analysis":
                            query_prompt = st.text_area(
                                "Enter prompt:",
                                height=150,
                                key="prompt_multi"
                            )
                        else:
                            query_prompt = st.text_area(
                                "Prompt (you can edit):",
                                value=PRESET_PROMPTS[preset_choice],
                                height=150,
                                key="prompt_multi"
                            )
                    else:
                        query_prompt = st.text_area(
                            "Enter custom prompt:",
                            height=150,
                            placeholder="Enter your analysis prompt here...",
                            key="prompt_multi"
                        )
                    
                    query_use_search = st.checkbox("Use web search for this query", 
                                                value=st.session_state.config.get('use_search', False),
                                                key="search_multi")
                    
                    if st.button("Add Query", type="primary"):
                        if query_name and query_prompt:
                            # Add to session state
                            if 'custom_queries' not in st.session_state:
                                st.session_state.custom_queries = []
                            
                            st.session_state.custom_queries.append({
                                'name': query_name,
                                'prompt': query_prompt,
                                'use_search': query_use_search
                            })
                            st.success(f"✅ Query '{query_name}' added!")
                            st.rerun()
                        else:
                            st.error("Please provide both a name and prompt for the query")
                
                # Display existing queries
                if st.session_state.custom_queries:
                    st.subheader("📋 Configured Queries")
                    
                    for i, query in enumerate(st.session_state.custom_queries):
                        with st.expander(f"Query {i+1}: {query['name']}", expanded=False):
                            st.text(f"Name: {query['name']}")
                            st.text(f"Use Search: {'Yes' if query['use_search'] else 'No'}")
                            st.text("Prompt:")
                            st.code(query['prompt'], language=None)
                            
                            if st.button(f"Remove", key=f"remove_{i}"):
                                st.session_state.custom_queries.pop(i)
                                st.rerun()
                    
                    # Update config with queries
                    st.session_state.config['custom_queries'] = st.session_state.custom_queries
                else:
                    st.warning("No queries configured yet. Add at least one query to proceed.")
        
        with tab3:
            # Processing Options (moved from tab1)
            st.header("🚀 Processing Options")
            
            # Get the current tagging method from the selectbox in tab2
            # We need to retrieve it from the widget state
            current_method = None
            for widget_id, widget_value in st.session_state.items():
                if 'selectbox' in str(widget_id) and isinstance(widget_value, str):
                    if widget_value in ["Use Taxonomy", "Use Custom Prompt", "Use Multiple Custom Queries"]:
                        current_method = widget_value
                        break
            
            # If we can't find it from widgets, try to get it from existing logic
            if not current_method:
                # Default fallback - try to determine from config
                if st.session_state.config.get('custom_queries'):
                    current_method = "Use Multiple Custom Queries"
                elif st.session_state.config.get('custom_prompt'):
                    current_method = "Use Custom Prompt"
                else:
                    current_method = "Use Taxonomy"
            
            # Determine if we're ready to process
            ready_to_process = False
            
            if current_method == "Use Taxonomy" and st.session_state.tagger and st.session_state.tagger.taxonomy:
                ready_to_process = True
            elif current_method == "Use Custom Prompt" and st.session_state.config.get('custom_prompt'):
                ready_to_process = True
            elif current_method == "Use Multiple Custom Queries" and st.session_state.custom_queries:
                ready_to_process = True
            
            # Show processing options only if ready and data is loaded
            if hasattr(st.session_state, 'df') and st.session_state.df is not None and ready_to_process:
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if current_method == "Use Taxonomy":
                        multi_select = st.checkbox("Allow multiple tags per entity")
                        if 'config' in st.session_state:
                            st.session_state.config['multi_select'] = multi_select
                
                with col2:
                    max_workers = st.number_input("Parallel threads", 1, 100, 5)
                
                with col3:
                    batch_size = st.number_input("Checkpoint batch size", 10, 1000, 100)
                
                # Search retry settings
                if st.session_state.config.get('use_search') or any(q.get('use_search') for q in st.session_state.config.get('custom_queries', [])):
                    with st.expander("🔄 Search Retry Settings", expanded=False):
                        search_retries = st.slider(
                            "Max retries for failed searches",
                            min_value=0,
                            max_value=10,
                            value=3,
                            help="Number of times to retry if Perplexity search fails (e.g., rate limiting)"
                        )
                        
                        retry_delay = st.number_input(
                            "Base retry delay (seconds)",
                            min_value=0.5,
                            max_value=10.0,
                            value=1.0,
                            step=0.5,
                            help="Initial delay before retry. Doubles with each attempt (exponential backoff)"
                        )
                        
                        st.info(f"Retry delays will be: {', '.join([f'{retry_delay * (2**i):.1f}s' for i in range(min(search_retries, 4))])}{' ...' if search_retries > 4 else ''}")
                        
                        st.session_state.config['search_max_retries'] = search_retries
                        
                        # Source citation option
                        st.markdown("### 📚 Source Citations")
                        st.info("Perplexity will be instructed to include source citations in its search results.")
                
                # Row selection
                st.subheader("Select rows to process")

                # Check if we're resuming from checkpoint
                if hasattr(st.session_state, 'resume_processing') and st.session_state.resume_processing:
                    st.warning("📌 RESUMING FROM CHECKPOINT")
                    rows_to_process = st.session_state.remaining_df
                    st.info(f"Will process {len(rows_to_process)} remaining rows")
                    
                    # Clear the resume flag after using it
                    del st.session_state.resume_processing
                else:
                    # Normal row selection
                    process_all = st.checkbox("Process all rows", value=True)
                    
                    if not process_all:
                        if 'config' in st.session_state and st.session_state.config.get('category_column'):
                            # Filter by category
                            categories = st.session_state.df[st.session_state.config['category_column']].unique()
                            selected_categories = st.multiselect("Select categories", categories)
                            
                            if selected_categories:
                                mask = st.session_state.df[st.session_state.config['category_column']].isin(selected_categories)
                                rows_to_process = st.session_state.df[mask]
                            else:
                                rows_to_process = st.session_state.df
                        else:
                            # Manual row range
                            total_rows = len(st.session_state.df)
                            row_range = st.slider("Select row range", 0, total_rows, (0, min(100, total_rows)))
                            rows_to_process = st.session_state.df.iloc[row_range[0]:row_range[1]]
                    else:
                        rows_to_process = st.session_state.df

                st.info(f"Will process {len(rows_to_process)} rows")            
                    
                # Start processing
                if st.button("🏁 Start Tagging", type="primary", disabled=st.session_state.processing):
                    st.session_state.processing = True
                    
                    # Initialize results - preserve existing if resuming from checkpoint
                    if hasattr(st.session_state, 'existing_results'):
                        # Starting with existing results from checkpoint
                        initial_results = st.session_state.existing_results.copy()
                        st.info(f"Starting with {len(initial_results)} existing results from checkpoint")
                    else:
                        # Fresh start
                        initial_results = []
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_placeholder = st.empty()
                    
                    # Process entities
                    results = []
                    errors = []
                    
                    # Use ThreadPoolExecutor for parallel processing
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_row = {
                            executor.submit(
                                st.session_state.tagger.process_single_entity,
                                row.to_dict(),
                                st.session_state.config,
                                lambda msg: None  # Simple progress callback
                            ): (idx, row) 
                            for idx, row in rows_to_process.iterrows()
                        }
                        
                        # Process completed tasks
                        for future in as_completed(future_to_row):
                            idx, row = future_to_row[future]
                            
                            try:
                                entity_name = row[st.session_state.config['name_column']]
                                
                                # Update status
                                completed = len(results) + len(errors) + 1
                                status_text.text(f"Processing {completed}/{len(rows_to_process)}: {entity_name}")
                                
                                # Get result
                                result = future.result()
                                results.append(result)
                                
                                # Update progress
                                progress = completed / len(rows_to_process)
                                progress_bar.progress(progress)
                                
                                # Save checkpoint (include initial results)
                                all_results = initial_results + results
                                if len(results) % batch_size == 0:
                                    checkpoint_path = st.session_state.tagger.save_checkpoint(
                                        all_results, f"batch_{len(all_results)}"
                                    )
                                    status_text.text(f"Checkpoint saved: {checkpoint_path.name}")
                                
                                # Show live results
                                if len(results) > 0:
                                    # Show last 5 results
                                    recent_results = results[-5:]
                                    results_df = pd.DataFrame(recent_results)
                                    
                                    # For multiple queries, show a subset of columns
                                    if st.session_state.config.get('custom_queries'):
                                        # Show entity name and first query results
                                        display_cols = [st.session_state.config['name_column']]
                                        for query in st.session_state.config['custom_queries'][:2]:  # Show first 2 queries
                                            query_name = query['name']
                                            display_cols.extend([
                                                f'{query_name}_Result',
                                                f'{query_name}_Status'
                                            ])
                                        display_cols = [col for col in display_cols if col in results_df.columns]
                                        results_placeholder.dataframe(results_df[display_cols])
                                    else:
                                        results_placeholder.dataframe(results_df)
                                
                            except Exception as e:
                                errors.append({
                                    'row': idx,
                                    'entity': row.get(st.session_state.config['name_column'], 'Unknown'),
                                    'error': str(e)
                                })
                                # Update progress even for errors
                                completed = len(results) + len(errors)
                                progress = completed / len(rows_to_process)
                                progress_bar.progress(progress)
                    
                    # Combine initial results with new results
                    st.session_state.results = initial_results + results
                    st.session_state.processing = False
                    
                    # Clear the existing_results flag
                    if hasattr(st.session_state, 'existing_results'):
                        del st.session_state.existing_results
                    
                    # Final summary
                    if st.session_state.config.get('custom_queries'):
                        # Summary for multiple queries
                        st.success(f"""
                        ✅ Processing complete!
                        - Total entities processed: {len(st.session_state.results)}
                        - Queries run per entity: {len(st.session_state.config['custom_queries'])}
                        - Total analyses performed: {len(st.session_state.results) * len(st.session_state.config['custom_queries'])}
                        """)
                        
                        # Per-query statistics
                        with st.expander("📊 Per-Query Statistics"):
                            for query in st.session_state.config['custom_queries']:
                                query_name = query['name']
                                status_col = f'{query_name}_Status'
                                
                                if status_col in pd.DataFrame(st.session_state.results).columns:
                                    status_counts = pd.DataFrame(st.session_state.results)[status_col].value_counts()
                                    st.write(f"**{query_name}:**")
                                    for status, count in status_counts.items():
                                        st.write(f"- {status}: {count}")
                    else:
                        # Original summary
                        success_count = sum(1 for r in st.session_state.results if r.get('Status') == 'Success')
                        error_count = sum(1 for r in st.session_state.results if r.get('Status') == 'Error')
                        search_error_count = sum(1 for r in st.session_state.results if r.get('Status') == 'Search Error')
                        
                        st.success(f"""
                        ✅ Processing complete!
                        - Total in results: {len(st.session_state.results)}
                        - Newly processed: {len(results)}
                        - Successful: {success_count}
                        - Search Errors: {search_error_count}
                        - Other Errors: {error_count}
                        """)
                    
                    # Save results
                    if st.session_state.results:
                        results_df = pd.DataFrame(st.session_state.results)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download as Excel
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                results_df.to_excel(writer, index=False, sheet_name='Tagged Results')
                            
                            st.download_button(
                                label="📥 Download Excel",
                                data=output.getvalue(),
                                file_name=f"tagged_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        with col2:
                            # Download as CSV
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"tagged_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        # Display results
                        st.header("📊 Results")
                        
                        # Filter options
                        show_filter = st.checkbox("Show filter options")
                        if show_filter:
                            filter_col1, filter_col2 = st.columns(2)
                            
                            if st.session_state.config.get('custom_queries'):
                                # Filters for multiple queries
                                with filter_col1:
                                    # Select which query to filter by
                                    query_names = [q['name'] for q in st.session_state.config['custom_queries']]
                                    selected_query = st.selectbox("Filter by query", ['All'] + query_names)
                                
                                with filter_col2:
                                    if selected_query != 'All':
                                        status_col = f'{selected_query}_Status'
                                        if status_col in results_df.columns:
                                            status_filter = st.multiselect(
                                                f"Filter by {selected_query} status",
                                                results_df[status_col].unique(),
                                                default=results_df[status_col].unique()
                                            )
                                            filtered_df = results_df[results_df[status_col].isin(status_filter)]
                                        else:
                                            filtered_df = results_df
                                    else:
                                        filtered_df = results_df
                            else:
                                # Original filters
                                with filter_col1:
                                    status_filter = st.multiselect(
                                        "Filter by status",
                                        results_df['Status'].unique() if 'Status' in results_df.columns else [],
                                        default=results_df['Status'].unique() if 'Status' in results_df.columns else []
                                    )
                                with filter_col2:
                                    if 'Tagged_Result' in results_df.columns:
                                        tag_filter = st.multiselect(
                                            "Filter by tag",
                                            results_df['Tagged_Result'].unique(),
                                            default=results_df['Tagged_Result'].unique()
                                        )
                                    else:
                                        tag_filter = None
                                
                                # Apply filters
                                if 'Status' in results_df.columns:
                                    filtered_df = results_df[results_df['Status'].isin(status_filter)]
                                else:
                                    filtered_df = results_df
                                    
                                if tag_filter and 'Tagged_Result' in results_df.columns:
                                    filtered_df = filtered_df[filtered_df['Tagged_Result'].isin(tag_filter)]
                            
                            st.dataframe(filtered_df, use_container_width=True)
                        else:
                            st.dataframe(results_df, use_container_width=True)
                        
                        # Error report
                        if errors:
                            st.header("⚠️ Processing Errors")
                            error_df = pd.DataFrame(errors)
                            st.dataframe(error_df)
                        
                        # Search errors for multiple queries
                        if st.session_state.config.get('custom_queries'):
                            search_error_queries = []
                            for query in st.session_state.config['custom_queries']:
                                query_name = query['name']
                                status_col = f'{query_name}_Status'
                                if status_col in results_df.columns:
                                    search_errors = results_df[results_df[status_col] == 'Search Error']
                                    if len(search_errors) > 0:
                                        search_error_queries.append((query_name, len(search_errors)))
                            
                            if search_error_queries:
                                st.warning("⚠️ Search Errors by Query:")
                                for query_name, count in search_error_queries:
                                    st.write(f"- {query_name}: {count} entities failed due to search errors")
            else:
                # Show helpful message when not ready to process
                if not hasattr(st.session_state, 'df') or st.session_state.df is None:
                    st.info("📁 Please upload data in the 'Data Input and Column Configuration' tab first.")
                elif not ready_to_process:
                    st.info("🎯 Please complete the configuration in the 'Tagging Method and Configuration' tab first.")
                else:
                    st.info("⚙️ Configuration incomplete. Please check the previous tabs.")
    
    else:
        st.warning("Please initialize the tagger with your API keys in the sidebar")


if __name__ == "__main__":
    create_streamlit_app()