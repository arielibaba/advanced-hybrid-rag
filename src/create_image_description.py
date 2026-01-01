#!/usr/bin/env python3
# create_image_descriptions.py

import asyncio
from pathlib import Path
import logging

from .helpers import extract_markdown_tables, remove_code_blocks, is_relevant_image
from .constants import IMAGE_PIXEL_THRESHOLD, IMAGE_PIXEL_VARIANCE_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


async def _describe_single_image(
    client,
    model: str,
    system_prompt: str,
    description_prompt: str,
    model_options: dict,
    image_path: Path
) -> str:
    """
    Offload the blocking ollama_client.chat(...) call to a thread so it won't
    block the asyncio event loop. Returns the cleaned description.
    """
    def sync_chat() -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": description_prompt,
                "images": [str(image_path)]
            }
        ]
        resp = client.chat(
            model=model,
            messages=messages,
            options=model_options
        )
        content = resp["message"]["content"]
        return content
    # If you want to remove code blocks, uncomment the next line
        # return remove_code_blocks(content)

    # run the blocking call in a thread
    description = await asyncio.to_thread(sync_chat)
    return description


async def create_image_descriptions_async(
    image_dir: str,
    ollama_client,
    model: str,
    system_prompt: str,
    description_prompt: str,
    model_options: dict,
    output_dir: str,
    max_concurrency: int = 5
):
    """
    Describe all *_Picture_*.jpg in `image_dir` with a bounded semaphore.
    """
    sem = asyncio.Semaphore(max_concurrency)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _worker(img_path: Path):
        async with sem:
            try:
                logging.info(f"Describing {img_path.name}")
                desc = await _describe_single_image(
                    client=ollama_client,
                    model=model,
                    system_prompt=system_prompt,
                    description_prompt=description_prompt,
                    model_options=model_options,
                    image_path=img_path
                )

                # Write description
                desc_file = output_dir / f"{img_path.stem}_description.txt"
                desc_file.write_text(desc, encoding="utf-8")

                # Extract & write tables
                for idx, (table_md, _) in enumerate(extract_markdown_tables(desc), start=1):
                    tbl_file = output_dir / f"{img_path.stem}_table_{idx}.md"
                    tbl_file.write_text(table_md, encoding="utf-8")

                logging.info(f"Completed {img_path.name}")
            except Exception as e:
                logging.error(f"Failed {img_path.name}: {e}")

    images = list(image_dir.glob("*_Picture_*.jpg"))
    # Filter images based on pixel thresholds
    images = [
        img for img in images
        if is_relevant_image(img, IMAGE_PIXEL_THRESHOLD, IMAGE_PIXEL_VARIANCE_THRESHOLD)
    ]
    
    if not images:
        logging.warning(f"No relevant images or no images matching '*_Picture_*.jpg' in {image_dir}")
        return

    logging.info(f"Found {len(images)} images; starting concurrent description…")
    await asyncio.gather(*[_worker(img) for img in images])
    logging.info(f"All done. Outputs in {output_dir}")