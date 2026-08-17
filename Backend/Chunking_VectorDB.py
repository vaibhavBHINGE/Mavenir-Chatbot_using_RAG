import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

# 1. Load Environment Variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing from .env file.")

index_name = "mavenir-3gpp-index"

# 2. XML Parser to extract sequential content
def iter_block_items(parent):
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Invalid parent object")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

# 3. Convert Tables to Markdown
def table_to_markdown(table):
    md = "\n"
    for i, row in enumerate(table.rows):
        row_data = [cell.text.replace("\n", " ").strip() for cell in row.cells]
        md += "| " + " | ".join(row_data) + " |\n"
        if i == 0:
            md += "|" + "|".join(["---"] * len(row.cells)) + "|\n"
    return md + "\n"

# 4. Level 1 Split: Group by Heading
def parse_3gpp_hierarchical(file_path):
    doc = Document(file_path)
    clauses = []
    
    current_clause_num = "General"
    current_clause_title = "General Context"
    current_content = []
    
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            
            style_name = block.style.name.lower()
            
            if 'heading' in style_name:
                if current_content:
                    clauses.append({
                        "clause_num": current_clause_num,
                        "title": current_clause_title,
                        "text": "\n".join(current_content)
                    })
                    current_content = []
                
                parts = text.split(" ", 1)
                if len(parts) > 1 and parts[0].replace(".", "").isdigit():
                    current_clause_num = parts[0]
                    current_clause_title = parts[1]
                else:
                    current_clause_num = "N/A"
                    current_clause_title = text
                    
                current_content.append(text)
                
        elif isinstance(block, Table):
            current_content.append(table_to_markdown(block))
            
    if current_content:
        clauses.append({
            "clause_num": current_clause_num,
            "title": current_clause_title,
            "text": "\n".join(current_content)
        })
        
    return clauses

# 5. Pure Python Recursive Chunking Fallback
def split_text_recursive(text, max_chars=1800, overlap=200):
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a clean split point
        split_pos = text.rfind("\n\n", start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = text.rfind("\n", start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = text.rfind(". ", start, end)
        if split_pos == -1 or split_pos <= start:
            split_pos = end
        else:
            split_pos += 1
            
        chunks.append(text[start:split_pos].strip())
        start = max(start + 1, split_pos - overlap)
        
    return [c for c in chunks if c]

def build_vector_payloads(clauses):
    records = []
    chunk_counter = 0
    
    for clause in clauses:
        clause_text = clause["text"]
        
        if len(clause_text) > 2000:
            sub_chunks = split_text_recursive(clause_text, max_chars=1800, overlap=200)
            total_parts = len(sub_chunks)
            for idx, chunk_str in enumerate(sub_chunks):
                records.append({
                    "id": f"chunk-{chunk_counter}",
                    "text": chunk_str,
                    "metadata": {
                        "source": "TS 23.501",
                        "clause_number": clause["clause_num"],
                        "clause_title": clause["title"],
                        "part": f"{idx+1}/{total_parts}",
                        "text": chunk_str  # Stored directly in metadata for zero-lookup retrieval
                    }
                })
                chunk_counter += 1
        else:
            records.append({
                "id": f"chunk-{chunk_counter}",
                "text": clause_text,
                "metadata": {
                    "source": "TS 23.501",
                    "clause_number": clause["clause_num"],
                    "clause_title": clause["title"],
                    "part": "1/1",
                    "text": clause_text
                }
            })
            chunk_counter += 1
            
    return records

# --- Pipeline Execution ---
if __name__ == "__main__":
    doc_path = "./knowledge_base/23501-j80.docx"
    print(f"Extracting hierarchical data from {doc_path}...")
    raw_clauses = parse_3gpp_hierarchical(doc_path)
    
    print("Generating chunk records...")
    records = build_vector_payloads(raw_clauses)
    print(f"Generated {len(records)} structured chunks.")
    
    # 6. Local Hugging Face Embeddings via sentence-transformers (384 dimensions)
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # 7. Pinecone Initialization
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)
    
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Serverless index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
            
    index = pc.Index(index_name)
    
    # 8. Batch Upserting
    batch_size = 64
    print("Embedding and upserting chunks to Pinecone...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        
        vectors_to_upsert = []
        for item, emb in zip(batch, embeddings):
            vectors_to_upsert.append({
                "id": item["id"],
                "values": emb,
                "metadata": item["metadata"]
            })
            
        index.upsert(vectors=vectors_to_upsert)
        print(f"Uploaded {min(i + batch_size, len(records))}/{len(records)} chunks...")
        
    print("Data ingestion complete. Knowledge base is live on Pinecone.")