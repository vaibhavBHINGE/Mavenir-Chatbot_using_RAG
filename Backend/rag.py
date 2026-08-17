import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

# --- Configuration ---
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Go up one level to the project root
PROJECT_ROOT = CURRENT_DIR.parent

# 3. Define absolute paths based on the project root
# This perfectly matches your structure: project_root/Knowedge_base/23501-j80.docx
DOC_PATH = PROJECT_ROOT / "Knowedge_base" / "23501-j80.docx"

# Put Chroma DB in the project root so both backend and frontend can easily access it
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
# all-mpnet-base-v2 is excellent for dense technical text and semantic search
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" 

def ingest_document():
    """
    Parses the 1510-page docx, preserves table structures, and builds the ChromaDB.
    You only need to run this once.
    """
    print(f"Loading document from {DOC_PATH}...")
    
    # Using mode="elements" forces unstructured to identify semantic blocks
    # (e.g., keeping an entire table together instead of splitting it)
    loader = UnstructuredWordDocumentLoader(DOC_PATH, mode="elements")
    raw_elements = loader.load()
    
    print(f"Extracted {len(raw_elements)} semantic elements (tables, paragraphs, titles).")
    
    processed_chunks = []
    current_section_title = "Unknown Section"
    
    for element in raw_elements:
        # Keep track of the current heading so we can attach it as metadata to tables/text
        if element.metadata.get("category") == "Title":
            current_section_title = element.page_content
            continue
            
        # We only want to embed actual content (Text and Tables)
        if element.metadata.get("category") in ["NarrativeText", "Table"]:
            
            # If it's a table, unstructured natively preserves its layout. 
            # We wrap it in a clear marker so the LLM knows how to read it.
            content = element.page_content
            if element.metadata.get("category") == "Table":
                content = f"--- TABLE DATA ---\n{content}\n--- END TABLE ---"
                
            doc = Document(
                page_content=content,
                metadata={
                    "source": "3GPP_Spec",
                    "section": current_section_title,
                    "type": element.metadata.get("category")
                }
            )
            processed_chunks.append(doc)

    print(f"Refined into {len(processed_chunks)} high-quality chunks.")
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating Chroma vector store at {CHROMA_PATH} (this will take a while)...")
    Chroma.from_documents(
        documents=processed_chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("Vector database successfully built!")


def get_retriever():
    """
    Connects to the existing Chroma database and returns a heavily constrained retriever.
    Called dynamically by our Agent in Backend/tools.py.
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Chroma DB not found at {CHROMA_PATH}. Run ingest_document() first."
        )
        
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # HALLUCINATION GUARDRAIL: Use MMR (Maximal Marginal Relevance).
    # This retrieves 10 chunks to check context (fetch_k), but strictly returns 
    # the 4 most relevant AND diverse chunks to ensure the LLM gets a complete picture 
    # without redundant noise.
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4, 
            "fetch_k": 10,
            "lambda_mult": 0.5 # Balances exact match vs diversity
        }
    )
    
    return retriever

if __name__ == "__main__":
    # If this script is run directly via terminal, build the database.
    # Make sure your docx is in the data/ folder first!
    ingest_document()