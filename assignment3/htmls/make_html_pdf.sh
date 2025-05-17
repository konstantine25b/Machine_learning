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

# Check if wkhtmltopdf is installed
if ! command -v wkhtmltopdf &> /dev/null; then
    echo "Installing wkhtmltopdf using Homebrew..."
    brew install wkhtmltopdf
fi

# Install required Python packages
echo "Installing required Python packages..."
python3.11 -m pip install pdfkit PyPDF2 --quiet

echo "### Creating PDF from HTML files ###"
python3.11 html_to_pdf.py --html_files "${HTML_FILES[@]}" --pdf_filename "${PDF_FILENAME}"

echo "### Done! PDF created as ${PDF_FILENAME} ###" 