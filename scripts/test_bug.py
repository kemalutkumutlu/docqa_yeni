import asyncio
import sys
from pathlib import Path
from src.config import load_settings
from src.core.ingestion import OCRConfig
from src.core.pipeline import RAGPipeline
from src.core.vlm_extract import VLMConfig

async def main():
    settings = load_settings()
    pipe = RAGPipeline(
        embedding_model=settings.embedding_model,
        chroma_dir=settings.chroma_dir,
        gemini_api_key=settings.gemini_api_key or "",
        gemini_model=settings.gemini_model,
        ocr_config=OCRConfig(
            enabled=getattr(settings, "ocr_enabled", True),
            lang="tur+eng",
            tesseract_cmd=settings.tesseract_cmd,
            tessdata_prefix=settings.tessdata_prefix,
            tesseract_config=getattr(settings, "tesseract_config", None),
        ),
        vlm_config=VLMConfig(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
            mode=settings.vlm_mode,
        ),
    )
    doc_path = Path("test_data/Case_Study_20260205.pdf")
    if not doc_path.exists():
        print("Test file not found")
        return
    print("Ingesting...")
    pipe.add_document(doc_path, "Case_Study_20260205.pdf")
    print("Retrieving...")
    ret = pipe.get_retrieval("Teslimatlar nelerdir")
    print("Evidences format for section_list:")
    for ev in ret.evidences:
        if ev.kind == "parent":
            print(f"PARENT TEXT:\n{ev.text}")
            
    print("Generating...")
    # we need an async call for generation usually or if it's sync, let's look at pipeline
    try:
        from src.core.generation import Generator
        gen = Generator(settings)
        res = await gen.generate_answer(
            query="Teslimatlar nelerdir",
            retrieval_result=ret,
            document_context="",
            chat_history=[]
        )
        print("DONE", res.answer)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
