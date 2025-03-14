import os
import re

import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Apply nest_asyncio (required for some environments)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Directory paths
pdfs_directory = "./pdfs"
output_directory = "./output"

# Ensure the output directories exist
os.makedirs(output_directory, exist_ok=True)

# Initialize LlamaParse with OpenAI integration and custom parsing instructions
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="text",  # Can also be "text" or "json"
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    vendor_multimodal_api_key=os.getenv("OPENAI_API_KEY"),
    verbose=True,
    num_workers=4,
    page_separator="\n---\n",  # Ensures pages are separated correctly
    parsing_instruction=
    """The document is a clinical medical guidance PDF. The content structure includes structured sections such as titles, subtitles, bulleted lists, numbered lists, and paragraphs. Most pages are always double column, have this structure into consideration and parse it accurately. Prioritize extracting medical terminology, instructions, and procedural steps. Accurately extract and name tables (e.g., "Table 1: Recommended Dosages," "Table 2: Common Side Effects") without merging rows or columns, and retain the structure and organization as presented. Recognize section headers and sub-headers (e.g., "Introduction," "Guidelines," "Dosage Instructions") and treat titles as standalone entities. For lists, ensure all bulleted and numbered lists are captured as distinct elements without merging with adjacent text. Maintain the original sequence of the text, preserving the intended flow without reordering content or making assumptions. Special characters like medical abbreviations, dosage notations, and symbols (e.g., "mg," "ml," "IV") should be preserved as presented. Avoid generating any content not explicitly mentioned in the document. Ensure text accuracy, especially for clinical procedures, drug names, or other critical medical guidance. If a section appears incomplete or unclear, extract it as-is without hypothesizing missing content. Capture footnotes and references if directly related to the content, and ensure they are tagged appropriately. For images and diagrams, provide detailed descriptions, including labels, colors, shapes, and relative positions of elements. For example, “Image 1: A flowchart showing the decision process for prescribing medication, with branches labeled ‘Mild,’ ‘Moderate,’ and ‘Severe,’ each leading to specific drug recommendations.” If images include text, extract the text and relate it clearly to the image description."""
)


def natural_sort_key(filename):
    # Extract numeric value from the filename for sorting
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', filename)
    ]


def parse_guidelines():
    # Loop through each guideline folder in the pdfs directory
    for guideline_folder in os.listdir(pdfs_directory):
        guideline_path = os.path.join(pdfs_directory, guideline_folder)
        if os.path.isdir(guideline_path):
            output_guideline_path = os.path.join(output_directory,
                                                 guideline_folder)
            os.makedirs(output_guideline_path, exist_ok=True)

            # Store individual parsed texts and combine them in numerical order
            combined_texts = []

            # Parse each PDF file in the guideline folder using the natural sorting key
            for pdf_file in sorted(os.listdir(guideline_path),
                                   key=natural_sort_key):
                if pdf_file.endswith(".pdf"):
                    pdf_path = os.path.join(guideline_path, pdf_file)
                    output_txt_path = os.path.join(
                        output_guideline_path,
                        f"{os.path.splitext(pdf_file)[0]}.txt")
                    with open(pdf_path, "rb") as file:
                        try:
                            documents = parser.load_data(
                                file, extra_info={"file_name": pdf_file})
                            if not documents:
                                print(
                                    f"Warning: No text extracted from {pdf_file}. The document might be empty or unsupported."
                                )
                            else:
                                # Correctly separate pages with "\n---\n"
                                page_text = "\n---\n".join(
                                    [doc.text for doc in documents])
                                combined_texts.append(page_text)
                                with open(output_txt_path, "w") as output_file:
                                    output_file.write(page_text)
                                print(
                                    f"Text extracted and saved to {output_txt_path}"
                                )
                        except Exception as e:
                            print(f"Error while parsing {pdf_file}: {e}")

            # Create the master file for the entire guideline with correct page separation
            master_txt_path = os.path.join(output_guideline_path,
                                           f"{guideline_folder}_combined.txt")
            with open(master_txt_path, "w") as master_file:
                master_file.write("\n---\n".join(combined_texts))
            print(f"Master file saved to {master_txt_path}")


# Run the function
parse_guidelines()
