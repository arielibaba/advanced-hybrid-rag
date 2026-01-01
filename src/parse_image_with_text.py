from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from PIL import Image
from pathlib import Path




def parse_image_with_text_func(image_path, model, processor, config, prompt, output_dir):

    print(f"\nParse image: {image_path.name}")
    
    ## Prepare input
    image = str(image_path.resolve())

    # Load image resource
    pil_image = Image.open(image)

    # Apply chat template
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    ## Generate output
    output = ""
    for token in stream_generate(
        model, processor, formatted_prompt, [image], max_tokens=4096, verbose=False
    ):
        output += token.text
        if "</doctag>" in token.text:
            break

    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([output], [pil_image])
    doc = DoclingDocument(name="SampleDocument")
    doc.load_from_doctags(doctags_doc)

    ## Export as Markdown
    doc_path = Path(output_dir) / f"{image_path.stem}.md"
    with open(doc_path, "w") as file:
        file.write(doc.export_to_markdown())

    print(f"Image content saved to: {doc_path.name}")


def parse_image_with_text(image_dir, model_dir, prompt, output_dir):
    # Define the download directory for the model and config
    # Or set the cache directory to where the model is already downloaded.
    download_dir = Path(model_dir)                       # Path("models") / "ds4sd_SmolDocling-256M-preview-mlx-bf16"
    download_dir.mkdir(parents=True, exist_ok=True)

    ## Load the model using the local cache directory
    model_id = "ds4sd/SmolDocling-256M-preview-mlx-bf16"
    model, processor = load(model_id, cache_dir=str(download_dir))
    config = load_config(model_id, cache_dir=str(download_dir))

    # Process image files with text only in the directory
    # Extract the texts

    print(f"\nProcessing the images with text only in directory: {image_dir} ...\n")

    for image_path in Path(image_dir).glob('*_Text.jpg'):
        parse_image_with_text_func(image_path, model, processor, config, prompt, output_dir)
    
    print(f"\nAll the images were processed successfully!\nThe texts extracted were saved to: {output_dir}\n")
    

