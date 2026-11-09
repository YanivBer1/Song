import os
import sys
from pathlib import Path
from typing import Optional, Union
import logging

# Import document parsing libraries
try:
    import PyPDF2
    from docx import Document
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentParser:
    """
    A class to parse and extract text from various document formats.
    Supports PDF, TXT, DOC, DOCX, and EPUB files.
    """
    
    def __init__(self):
        self.supported_formats = {
            'application/pdf': self._parse_pdf,
            'text/plain': self._parse_txt,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._parse_docx,
            'application/msword': self._parse_doc,
            'application/epub+zip': self._parse_epub
        }
    
    def get_file_type(self, file_path: Union[str, Path]) -> str:
        """
        Detect the MIME type of a file using file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        return self._get_mime_from_extension(file_path)
    
    def _get_mime_from_extension(self, file_path: Union[str, Path]) -> str:
        """
        Determine MIME type from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        extension = Path(file_path).suffix.lower()
        extension_map = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.epub': 'application/epub+zip'
        }
        
        mime_type = extension_map.get(extension, 'unknown')
        
        # If no extension or unknown extension, try to detect by content
        if mime_type == 'unknown':
            mime_type = self._detect_mime_by_content(file_path)
            
        return mime_type
    
    def _detect_mime_by_content(self, file_path: Union[str, Path]) -> str:
        """
        Detect MIME type by reading file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type string
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes to detect file type
                header = f.read(1024)
                
                # PDF detection
                if header.startswith(b'%PDF'):
                    return 'application/pdf'
                
                # ZIP-based formats (DOCX, EPUB)
                if header.startswith(b'PK\x03\x04'):
                    # Check if it's EPUB by looking for mimetype file
                    try:
                        import zipfile
                        with zipfile.ZipFile(file_path, 'r') as zf:
                            if 'mimetype' in zf.namelist():
                                with zf.open('mimetype') as mf:
                                    mimetype = mf.read().decode('utf-8').strip()
                                    if mimetype == 'application/epub+zip':
                                        return 'application/epub+zip'
                        # If not EPUB, assume DOCX
                        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    except:
                        pass
                
                # Plain text detection (try to decode as UTF-8)
                try:
                    header.decode('utf-8')
                    return 'text/plain'
                except UnicodeDecodeError:
                    pass
                
        except Exception as e:
            logger.warning(f"Error detecting MIME type by content: {e}")
        
        return 'unknown'
    
    def extract_text(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            mime_type = self.get_file_type(file_path)
            logger.info(f"Detected file type: {mime_type}")
            
            if mime_type in self.supported_formats:
                return self.supported_formats[mime_type](file_path)
            else:
                logger.error(f"Unsupported file type: {mime_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None
    
    def _parse_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise
            
        return text.strip()
    
    def _parse_txt(self, file_path: Path) -> str:
        """
        Extract text from plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Extracted text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise
    
    def _parse_docx(self, file_path: Path) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            raise
    
    def _parse_doc(self, file_path: Path) -> str:
        """
        Extract text from DOC file (legacy Word format).
        Note: This requires additional dependencies like antiword or catdoc.
        
        Args:
            file_path: Path to DOC file
            
        Returns:
            Extracted text
        """
        try:
            # Try using antiword if available
            import subprocess
            result = subprocess.run(['antiword', str(file_path)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Fallback: try catdoc
            result = subprocess.run(['catdoc', str(file_path)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
                
            raise Exception("Neither antiword nor catdoc found. Please install one of them.")
            
        except FileNotFoundError:
            raise Exception("antiword or catdoc not found. Please install one of them for DOC file support.")
        except Exception as e:
            logger.error(f"Error parsing DOC {file_path}: {e}")
            raise
    
    def _parse_epub(self, file_path: Path) -> str:
        """
        Extract text from EPUB file.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Extracted text
        """
        try:
            book = epub.read_epub(file_path)
            text = ""
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(content, 'html.parser')
                    text += soup.get_text() + "\n"
                    
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error parsing EPUB {file_path}: {e}")
            raise


def main():
    """
    Main function to demonstrate usage of the DocumentParser.
    """
    if len(sys.argv) != 2:
        print("Usage: python document_parsing.py <file_path>")
        print("Supported formats: PDF, TXT, DOC, DOCX, EPUB")
        sys.exit(1)
    
    file_path = sys.argv[1]
    parser = DocumentParser()
    
    print(f"Extracting text from: {file_path}")
    print("-" * 50)
    
    extracted_text = parser.extract_text(file_path)
    
    if extracted_text:
        print("Extracted text:")
        print(extracted_text)
        print(f"\nTotal characters: {len(extracted_text)}")
    else:
        print("Failed to extract text from the file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
