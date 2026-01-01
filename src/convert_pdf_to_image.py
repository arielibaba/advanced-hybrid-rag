from pathlib import Path
from pdf2image import convert_from_path



def convert_pdf_to_image(pdf_dir, image_dir):
    """
    Converts each page of every PDF file in the specified directory into separate image files.
    """
    # Ensure pdf_dir and image_dir are Path objects
    pdf_dir = Path(pdf_dir)
    image_dir = Path(image_dir)

    # Set parameter to control image resolution
    dpi = 600

    # Check if the pdf_dir exists
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"The specified pdf_dir '{pdf_dir}' does not exist or is not a directory.")
        return

    # Iterate over all PDF files in the pdf_dir
    print(f"\nProcessing the documents in folder: {pdf_dir} ...")
    for pdf_path in pdf_dir.glob('*.pdf'):
        print(f"\nConverting document: {pdf_path.name}")
        # Convert the PDF to images
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            print(f"Failed to convert '{pdf_path.name}': {e}")
            continue

        # Save each image
        for i, image in enumerate(images, start=1):
            # Construct the output image file path
            output_path = image_dir / f"{pdf_path.stem}_page_{i}.png"
            try:
                image.save(output_path, 'PNG')
                print(f"Page {i} saved as: {output_path.name}")
            except Exception as e:
                print(f"Failed to save page {i} of '{pdf_path.name}': {e}")

