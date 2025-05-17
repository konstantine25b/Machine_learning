#!/bin/bash
set -euo pipefail

# Define your HTML files in the order you want them in the PDF
HTML_FILES=(
    "FullyConnectedNets.html"
    "BatchNormalization.html"
    "Dropout.html"
    "ConvolutionalNetworks.html"
    "PyTorch.html"
)

PDF_FILENAME="a2_inline_submission.pdf"

# Check if files exist
for FILE in "${HTML_FILES[@]}"
do
    if [ ! -f ${FILE} ]; then
        echo -e "Required file ${FILE} not found, Exiting."
        exit 1
    fi
done

# Install required Python packages
echo "Installing required Python packages..."
python3.11 -m pip install python-docx htmldocx docx2pdf --quiet

echo "### Creating PDF from HTML files ###"
python3.11 simple_html_to_pdf.py --html_files "${HTML_FILES[@]}" --pdf_filename "${PDF_FILENAME}"

echo "### Done! PDF created as ${PDF_FILENAME} ###" 