#!/usr/bin/env python3
"""
ASL Gloss Converter using Claude API

This script converts English text input and convert it to ASL gloss format.
ASL gloss is a written representation of sign language
that preserves the spatial and grammatical structure of ASL.
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not found. Please install it with:")
    print("pip install anthropic")
    sys.exit(1)


class ASLGlossConverter:
    """
    Converts English text to ASL gloss using Claude's API.
    
    ASL gloss preserves the spatial and grammatical structure of American Sign Language,
    including features like:
    - Topic-comment structure
    - Spatial referencing
    - Non-manual markers (facial expressions, head movements)
    - Classifier predicates
    - Time indicators
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ASL gloss converter.
        
        Args:
            api_key: Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # System prompt that defines ASL gloss conversion rules
        self.system_prompt = """You are an expert in American Sign Language (ASL) and ASL gloss. Your task is to convert English text to ASL gloss format with bracketed phrases.

ASL GLOSS RULES:
1. Use ALL CAPS for all signs
2. Group related words/concepts into bracketed phrases [PHRASE]
3. Use underscores (_) to connect words within a phrase that are signed together
4. Use classifiers (CL:1, CL:3, CL:C, etc.) for spatial relationships
5. Use pronouns: I (first person), YOU (second person), HE/SHE/THEY (third person)
6. Use time indicators: PAST, FUTURE, NOW, ALWAYS, NEVER
7. Use topic-comment structure: TOPIC COMMENT
8. Use rhetorical questions: RHQ
9. Use conditional markers: IF-THEN
10. Use negation: NOT, NONE, CAN'T, DON'T-WANT
11. Use aspect markers: FINISH, CONTINUE, REPEAT
12. Use directional verbs: GIVE-TO, TELL-TO, ASK-TO
13. Use location markers: HERE, THERE, WHERE
14. Use manner adverbs: FAST, SLOW, CAREFUL, HARD

PHRASE GROUPING GUIDELINES:
- Group compound expressions: [GOOD_MORNING], [THANK_YOU], [HOW_ARE_YOU]
- Keep names as single phrases: [JOHN], [NATALIIA], [CHRISTOPHER_ROBIN]
- Group related concepts: [MY_NAME], [YOUR_HOUSE], [LAST_WEEK]
- Keep simple words separate: [I] [LOVE] [YOU]

IMPORTANT: Output ONLY the bracketed ASL phrases. Each phrase should be in ALL CAPS with underscores connecting related words.

EXAMPLES:
- "Good morning, Brian" → [GOOD_MORNING] [BRIAN]
- "My name is Nataliia" → [I] [NAME] [NATALIIA]
- "I love you" → [I] [LOVE] [YOU]
- "What is your name?" → [YOU] [NAME] [WHAT]
- "I don't understand" → [I] [UNDERSTAND] [NOT]
- "Where is the bathroom?" → [BATHROOM] [WHERE]
- "I want to go home" → [I] [WANT] [GO] [HOME]
- "The cat is sleeping" → [CAT] [SLEEP]
- "I finished my homework" → [I] [HOMEWORK] [FINISH]
- "Do you want coffee?" → [YOU] [WANT] [COFFEE]
- "I can't hear you" → [I] [HEAR] [YOU] [CAN'T]
- "The weather is nice today" → [TODAY] [WEATHER] [NICE]
- "Thank you very much" → [THANK_YOU] [VERY_MUCH]
- "How are you doing?" → [HOW_ARE_YOU] [DOING]
- "See you later" → [SEE_YOU_LATER]
- "I work at Google" → [I] [WORK] [GOOGLE]

Convert the given English text to proper ASL gloss format with bracketed phrases, maintaining the meaning and intent while following ASL grammar and structure."""

    def convert_text(self, english_text: str) -> str:
        """
        Convert English text to ASL gloss using Anthropic v1.x messages API.
        """
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": f"Convert this English text to ASL gloss:\n\n{english_text}"}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            raise Exception(f"Error converting text to ASL gloss: {str(e)}")

    def convert_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Convert text from a file to ASL gloss.
        
        Args:
            input_file: Path to input text file
            output_file: Path to output file (optional)
            
        Returns:
            The ASL gloss text
        """
        try:
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                english_text = f.read().strip()
            
            if not english_text:
                raise ValueError("Input file is empty")
            
            # Convert to ASL gloss
            asl_gloss = self.convert_text(english_text)
            
            # Write to output file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(asl_gloss)
                print(f"ASL gloss saved to: {output_file}")
            
            return asl_gloss
            
        except FileNotFoundError:
            raise Exception(f"Input file not found: {input_file}")
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def batch_convert(self, input_files: list, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Convert multiple files to ASL gloss.
        
        Args:
            input_files: List of input file paths
            output_dir: Directory to save output files (optional)
            
        Returns:
            Dictionary mapping input files to their ASL gloss
        """
        results = {}
        
        for input_file in input_files:
            try:
                print(f"Converting: {input_file}")
                
                if output_dir:
                    # Create output filename
                    input_path = Path(input_file)
                    output_filename = f"{input_path.stem}_asl_gloss{input_path.suffix}"
                    output_file = Path(output_dir) / output_filename
                else:
                    output_file = None
                
                asl_gloss = self.convert_file(input_file, str(output_file) if output_file else None)
                results[input_file] = asl_gloss
                
                print(f"✓ Completed: {input_file}")
                
            except Exception as e:
                print(f"✗ Error processing {input_file}: {str(e)}")
                results[input_file] = f"ERROR: {str(e)}"
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert English text to ASL gloss using Claude's API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert text directly
  python asl_gloss.py "Hello, how are you?"
  
  # Convert from file
  python asl_gloss.py -f input.txt
  
  # Convert from file with output
  python asl_gloss.py -f input.txt -o output.txt
  
  # Batch convert multiple files
  python asl_gloss.py -b file1.txt file2.txt -d output_dir/
  
  # Interactive mode
  python asl_gloss.py -i
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        help='English text to convert to ASL gloss'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='Input file containing English text'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file for ASL gloss'
    )
    
    parser.add_argument(
        '-b', '--batch',
        nargs='+',
        help='Batch convert multiple files'
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
        '--api-key',
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = ASLGlossConverter(api_key=args.api_key)
        
        if args.interactive:
            print("ASL Gloss Converter - Interactive Mode")
            print("Enter English text to convert to ASL gloss (or 'quit' to exit):")
            print("-" * 50)
            
            while True:
                try:
                    text = input("\nEnglish text: ").strip()
                    if text.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not text:
                        continue
                    
                    print("Converting...")
                    asl_gloss = converter.convert_text(text)
                    print(f"ASL Gloss: {asl_gloss}")
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        elif args.batch:
            if not args.batch:
                print("Error: No files specified for batch conversion")
                return 1
            
            print(f"Batch converting {len(args.batch)} files...")
            results = converter.batch_convert(args.batch, args.output_dir)
            
            print("\nResults:")
            for input_file, result in results.items():
                print(f"\n{input_file}:")
                print(result)
        
        elif args.file:
            asl_gloss = converter.convert_file(args.file, args.output)
            if not args.output:
                print("ASL Gloss:")
                print(asl_gloss)
        
        elif args.text:
            asl_gloss = converter.convert_text(args.text)
            print("ASL Gloss:")
            print(asl_gloss)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(asl_gloss)
                print(f"\nSaved to: {args.output}")
        
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
