#!/usr/bin/env python3.11
import os
import argparse
import pdfkit
from PyPDF2 import PdfMerger

def convert_html_to_pdf(html_files, output_pdf):
    """
    Convert multiple HTML files to a single PDF
    """
    # Create a temporary directory for intermediate PDFs
    if not os.path.exists('temp_pdfs'):
        os.makedirs('temp_pdfs')
    
    temp_pdfs = []
    
    # Convert each HTML file to a temporary PDF
    for i, html_file in enumerate(html_files):
        output_path = f'temp_pdfs/temp_{i}.pdf'
        print(f"Converting {html_file} to PDF...")
        
        try:
            pdfkit.from_file(html_file, output_path)
            temp_pdfs.append(output_path)
        except Exception as e:
            print(f"Error converting {html_file}: {str(e)}")
    
    # Merge all temporary PDFs into one
    merger = PdfMerger()
    
    for pdf in temp_pdfs:
        merger.append(pdf)
    
    merger.write(output_pdf)
    merger.close()
    
    # Clean up temporary files
    for pdf in temp_pdfs:
        try:
            os.remove(pdf)
        except:
            pass
    
    try:
        os.rmdir('temp_pdfs')
    except:
        pass
    
    print(f"PDF generated: {output_pdf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HTML files to a single PDF')
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