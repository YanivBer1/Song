#!/usr/bin/env python3
"""
Document to ASL Gloss Converter

This script combines document parsing and ASL glossing to convert
uploaded documents (PDF, TXT, DOC, DOCX, EPUB) directly to ASL gloss format.
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

# Import our existing modules
from document_parsing import DocumentParser
from asl_gloss import ASLGlossConverter


class DocumentToASLConverter:
    """
    Combines document parsing and ASL glossing functionality.
    Extracts text from various document formats and converts to ASL gloss.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the document to ASL converter.
        
        Args:
            api_key: Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var.
        """
        self.document_parser = DocumentParser()
        self.asl_converter = ASLGlossConverter(api_key=api_key)
    
    def convert_document(self, document_path: str, output_file: Optional[str] = None) -> str:
        """
        Convert a document file to ASL gloss.
        
        Args:
            document_path: Path to the document file
            output_file: Path to output file (optional)
            
        Returns:
            The ASL gloss text
        """
        try:
            print(f"Processing document: {document_path}")
            
            # Step 1: Extract text from document
            print("Step 1: Extracting text from document...")
            extracted_text = self.document_parser.extract_text(document_path)
            
            if not extracted_text:
                raise Exception("Failed to extract text from document")
            
            print(f"✓ Extracted {len(extracted_text)} characters")
            
            # Step 2: Convert text to ASL gloss
            print("Step 2: Converting to ASL gloss...")
            asl_gloss = self.asl_converter.convert_text(extracted_text)
            
            print("✓ ASL gloss conversion completed")
            
            # Step 3: Save to output file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(asl_gloss)
                print(f"✓ ASL gloss saved to: {output_file}")
            
            return asl_gloss
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def batch_convert_documents(self, document_paths: list, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Convert multiple documents to ASL gloss.
        
        Args:
            document_paths: List of document file paths
            output_dir: Directory to save output files (optional)
            
        Returns:
            Dictionary mapping input files to their ASL gloss
        """
        results = {}
        
        for document_path in document_paths:
            try:
                print(f"\n{'='*50}")
                print(f"Converting: {document_path}")
                print(f"{'='*50}")
                
                if output_dir:
                    # Create output filename
                    input_path = Path(document_path)
                    output_filename = f"{input_path.stem}_asl_gloss.txt"
                    output_file = Path(output_dir) / output_filename
                else:
                    output_file = None
                
                asl_gloss = self.convert_document(document_path, str(output_file) if output_file else None)
                results[document_path] = asl_gloss
                
                print(f"✓ Completed: {document_path}")
                
            except Exception as e:
                print(f"✗ Error processing {document_path}: {str(e)}")
                results[document_path] = f"ERROR: {str(e)}"
        
        return results
    
    def get_supported_formats(self) -> list:
        """
        Get list of supported document formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.pdf', '.txt', '.docx', '.doc', '.epub']


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert documents to ASL gloss using Claude's API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single document
  python document_to_asl.py document.pdf
  
  # Convert document with output file
  python document_to_asl.py document.pdf -o output.txt
  
  # Batch convert multiple documents
  python document_to_asl.py -b doc1.pdf doc2.docx doc3.txt -d output_dir/
  
  # Interactive mode
  python document_to_asl.py -i
  
  # Show supported formats
  python document_to_asl.py --formats
        """
    )
    
    parser.add_argument(
        'document',
        nargs='?',
        help='Document file to convert to ASL gloss'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file for ASL gloss'
    )
    
    parser.add_argument(
        '-b', '--batch',
        nargs='+',
        help='Batch convert multiple documents'
    )
    
    parser.add_argument(
        '-d', '--output-dir',
        help='Output directory for batch conversion'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--formats',
        action='store_true',
        help='Show supported document formats'
    )
    
    parser.add_argument(
        '--api-key',
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = DocumentToASLConverter(api_key=args.api_key)
        
        if args.formats:
            print("Supported Document Formats:")
            print("=" * 30)
            formats = converter.get_supported_formats()
            for fmt in formats:
                print(f"  • {fmt}")
            print("\nExamples: .pdf, .txt, .docx, .doc, .epub")
            return 0
        
        if args.interactive:
            print("Document to ASL Gloss Converter - Interactive Mode")
            print("Enter document file paths to convert (or 'quit' to exit):")
            print("-" * 60)
            
            while True:
                try:
                    doc_path = input("\nDocument path: ").strip()
                    if doc_path.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not doc_path:
                        continue
                    
                    if not os.path.exists(doc_path):
                        print(f"Error: File not found: {doc_path}")
                        continue
                    
                    # Ask for output file
                    output_file = input("Output file (optional, press Enter to skip): ").strip()
                    if not output_file:
                        output_file = None
                    
                    print("Converting...")
                    asl_gloss = converter.convert_document(doc_path, output_file)
                    
                    if not output_file:
                        print("\nASL Gloss:")
                        print("-" * 20)
                        print(asl_gloss)
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        elif args.batch:
            if not args.batch:
                print("Error: No documents specified for batch conversion")
                return 1
            
            print(f"Batch converting {len(args.batch)} documents...")
            results = converter.batch_convert_documents(args.batch, args.output_dir)
            
            print("\n" + "="*60)
            print("BATCH CONVERSION RESULTS")
            print("="*60)
            for doc_path, result in results.items():
                print(f"\nDocument: {doc_path}")
                print("-" * 40)
                if result.startswith("ERROR:"):
                    print(f"❌ {result}")
                else:
                    print("✅ Conversion successful")
                    if not args.output_dir:
                        print("ASL Gloss:")
                        print(result[:500] + "..." if len(result) > 500 else result)
        
        elif args.document:
            asl_gloss = converter.convert_document(args.document, args.output)
            if not args.output:
                print("\nASL Gloss:")
                print("-" * 20)
                print(asl_gloss)
        
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 