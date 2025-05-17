#!/usr/bin/env python3.11
import os
import argparse
import tempfile
from docx import Document
from htmldocx import HtmlToDocx
from docx2pdf import convert

def convert_html_to_pdf(html_files, output_pdf):
    """
    Convert multiple HTML files to a single PDF using docx as intermediary format
    """
    # Create a single Word document
    doc = Document()
    html_parser = HtmlToDocx()
    
    # Add each HTML file to the document
    for i, html_file in enumerate(html_files):
        print(f"Processing {html_file}...")
        
        try:
            with open(html_file, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # Add a page break if not the first file
            if i > 0:
                doc.add_page_break()
            
            # Add a title for each section
            doc.add_heading(os.path.basename(html_file), level=1)
            
            # Convert HTML to docx
            html_parser.add_html_to_document(html_content, doc)
        except Exception as e:
            print(f"Error processing {html_file}: {str(e)}")
    
    # Save the Word document to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
        temp_docx = temp_file.name
    
    doc.save(temp_docx)
    print(f"Saved intermediary Word document: {temp_docx}")
    
    # Convert Word document to PDF
    print(f"Converting to PDF: {output_pdf}")
    convert(temp_docx, output_pdf)
    
    # Clean up temporary files
    os.remove(temp_docx)
    
    print(f"PDF generated: {output_pdf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HTML files to a single PDF via Word')
    parser.add_argument('--html_files', nargs='+', default=[], help='HTML files to convert')
    parser.add_argument('--all', action='store_true', help='Convert all HTML files in directory')
    parser.add_argument('--pdf_filename', default='html_submission.pdf', help='Output PDF filename')
    
    args = parser.parse_args()
    
    html_files = args.html_files
    
    # If --all flag is provided, get all HTML files in current directory
    if args.all:
        html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    
    # Sort files to maintain order if using --all
    html_files.sort()
    
    if not html_files:
        print("No HTML files specified. Use --html_files or --all option.")
        exit(1)
    
    convert_html_to_pdf(html_files, args.pdf_filename) 