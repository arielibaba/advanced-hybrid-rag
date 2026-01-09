#!/usr/bin/env python3
"""
Focused end-to-end test for the CogniDoc pipeline.
Tests ONLY our test document by temporarily isolating it.
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("=" * 60)
    print("CogniDoc Focused End-to-End Test")
    print("=" * 60)

    base = Path(__file__).parent / "data"
    sources_dir = base / "sources"
    pdfs_dir = base / "pdfs"

    # Check test document exists
    test_doc = sources_dir / "test_document.txt"
    if not test_doc.exists():
        print(f"ERROR: Test document not found at {test_doc}")
        return False

    print(f"\nTest document: {test_doc}")

    # Create a temporary directory to backup existing PDFs
    backup_dir = None
    existing_pdfs = list(pdfs_dir.iterdir()) if pdfs_dir.exists() else []

    if existing_pdfs:
        backup_dir = tempfile.mkdtemp(prefix="cognidoc_backup_")
        print(f"\nBacking up {len(existing_pdfs)} existing items to {backup_dir}")
        for item in existing_pdfs:
            if item.name != ".DS_Store":
                dest = Path(backup_dir) / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
                # Remove from pdfs_dir
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    try:
        # Import and run
        from cognidoc import CogniDoc

        print("\n1. Initializing CogniDoc...")
        doc = CogniDoc(
            llm_provider="gemini",
            embedding_provider="ollama",
            use_graph=False,
            use_reranking=False,
        )
        print(f"   Config: {doc.get_info()}")

        print("\n2. Running ingestion pipeline on test document...")
        print("   (skipping YOLO, descriptions, and graph for speed)")

        result = doc.ingest(
            source=str(test_doc),
            skip_yolo=True,
            skip_descriptions=True,
            skip_graph=True,
        )

        print(f"\n   Ingestion Results:")
        print(f"   - Documents processed: {result.documents_processed}")
        print(f"   - Chunks created: {result.chunks_created}")
        print(f"   - Errors: {result.errors}")

        if result.errors:
            print(f"\n   ERRORS during ingestion:")
            for err in result.errors:
                print(f"   - {err}")
            return False

        # Verify files were created
        print("\n3. Verifying created files...")

        # Check PDF was created
        pdf_files = list(pdfs_dir.glob("test_document*.pdf"))
        print(f"   - PDFs created: {len(pdf_files)}")
        for p in pdf_files:
            print(f"     - {p.name}")

        # Check images
        images_dir = base / "images"
        image_files = list(images_dir.glob("test_document*"))
        print(f"   - Images created: {len(image_files)}")

        # Check chunks
        chunks_dir = base / "chunks"
        chunk_files = list(chunks_dir.glob("*test_document*"))
        print(f"   - Chunk files: {len(chunk_files)}")

        # Test query
        print("\n4. Testing query functionality...")
        query = "What are the key features of CogniDoc?"
        print(f"   Query: {query}")

        result = doc.query(query)

        print(f"\n   Query Results:")
        if result.answer:
            answer_preview = result.answer[:300] + "..." if len(result.answer) > 300 else result.answer
            print(f"   - Answer: {answer_preview}")
        else:
            print("   - Answer: (empty)")
        print(f"   - Sources: {len(result.sources)}")
        print(f"   - Query type: {result.query_type}")

        print("\n" + "=" * 60)
        print("END-TO-END TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore backed up PDFs
        if backup_dir and os.path.exists(backup_dir):
            print(f"\nRestoring backed up items from {backup_dir}")
            for item in Path(backup_dir).iterdir():
                dest = pdfs_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            shutil.rmtree(backup_dir)
            print("   Backup restored successfully")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
