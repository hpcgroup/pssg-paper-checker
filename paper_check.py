import os
import re
import json
import argparse
from PyPDF2 import PdfReader
import language_tool_python
from pdf2image import convert_from_path
from openai import OpenAI
try:
    import google.generativeai as genai
except ImportError:
    genai = None

DEBUG = False

SUPPORTED_MODELS = {
    "grok-2-latest": {"provider": "xai", "vision_model": "grok-2-vision-1212"},
    "gemini-2.5-flash": {"provider": "gemini", "vision_model": "gemini-2.5-flash"},
    "gemini-2.5-pro": {"provider": "gemini", "vision_model": "gemini-2.5-pro"},
    "gpt-4o-mini": {"provider": "openai", "vision_model": "gpt-4o-mini"},
}

class ChecklistEvaluator:
    def __init__(self, pdf_path, latex_path_or_dir, api_key=None, model_name=None, skip_llm=False):
        self.pdf_path = pdf_path
        self.file_text_map = {}
        self.model_name = model_name
        self.model_config = SUPPORTED_MODELS.get(model_name) if model_name else None
        self.provider = self.model_config['provider'] if self.model_config else None
        self.skip_llm = skip_llm or (self.model_config is None)
        self.api_key = api_key
        self._clients = {}
        if not self.skip_llm and self.model_config is None:
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS)}")
        if os.path.isdir(latex_path_or_dir):
            # Prioritize loading 'paper.tex' and process \input commands to include corresponding files
            self.latex_root_dir = os.path.abspath(latex_path_or_dir)
            self.latex_text = self.load_latex_dir(self.latex_root_dir)
        else:
            self.latex_root_dir = os.path.abspath(os.path.dirname(latex_path_or_dir))
            self.latex_text = self.load_latex_file(latex_path_or_dir)
        # Extract sections from the merged LaTeX text
        self.sections = self.extract_sections(self.latex_text)
        self.pdf_text = self.load_pdf()
        if self.provider:
            self._store_api_key()
        self.report = {}

    @staticmethod
    def _remove_comment_only_lines(text):
        """Strip lines that are purely LaTeX comments (starting with %)"""
        return ''.join(
            line for line in text.splitlines(keepends=True)
            if not line.lstrip().startswith('%')
        )

    def _register_file_text(self, file_path, content):
        """Record comment-stripped LaTeX content for per-file static checks."""
        abs_path = os.path.abspath(file_path)
        try:
            rel_path = os.path.relpath(abs_path, self.latex_root_dir)
        except ValueError:
            # Fallback for different drives or missing root
            rel_path = os.path.basename(abs_path)
        self.file_text_map[rel_path] = content
        return rel_path

    def _iterate_latex_files(self):
        """Yield (relative_path, content) pairs for each processed LaTeX source."""
        if self.file_text_map:
            for path in sorted(self.file_text_map.keys()):
                yield path, self.file_text_map[path]
        else:
            yield "document.tex", self.latex_text

    def _store_api_key(self):
        """Persist provided API key to appropriate environment variable for downstream SDKs."""
        if not self.api_key:
            return
        if self.provider == "xai":
            os.environ["XAI_API_KEY"] = self.api_key
        elif self.provider == "gemini":
            os.environ["GEMINI_API_KEY"] = self.api_key
        elif self.provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key

    def _get_api_key(self):
        """Return the API key for the configured provider."""
        if self.provider == "xai":
            return self.api_key or os.getenv("XAI_API_KEY")
        if self.provider == "gemini":
            return self.api_key or os.getenv("GEMINI_API_KEY")
        if self.provider == "openai":
            return self.api_key or os.getenv("OPENAI_API_KEY")
        return None

    def _get_vision_model(self):
        """Return the vision-capable model to use for figure analysis."""
        if not self.model_config:
            return None
        return self.model_config.get("vision_model", self.model_name)

    def load_pdf(self):
        """Extract text from PDF"""
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def load_latex_file(self, file_path):
        """Load a single LaTeX file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        content_no_comments = self._remove_comment_only_lines(original_content)
        self._register_file_text(file_path, content_no_comments)
        return content_no_comments

    def load_latex_dir(self, directory):
        """
        Load LaTeX documents from a directory:
        1. Prioritize 'paper.tex'.
        2. Recursively process all \\input{...} commands, loading corresponding .tex files (relative to directory).
        3. Replace \\input commands with file content and return the merged LaTeX text.
        4. Prevent infinite recursion caused by circular references.
        5. Remove all comment lines starting with % during processing.
        """
        paper_path = os.path.join(directory, "paper.tex")
        if not os.path.exists(paper_path):
            raise FileNotFoundError("paper.tex file not found in directory.")
        
        print(f"Starting to process LaTeX file: {paper_path}")
        
        # Track processed files to prevent circular references
        processed_files = set()
        
        def process_input_commands(file_path, current_directory):
            # If file has been processed, prevent circular references
            abs_file_path = os.path.abspath(file_path)
            if abs_file_path in processed_files:
                print(f"Warning: Circular reference detected for {file_path}, skipping")
                return f"% Circular reference: {file_path} has been ignored"
            
            # Mark file as processed
            processed_files.add(abs_file_path)
            
            print(f"Processing file: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                # Remove all comment lines starting with %
                content = self._remove_comment_only_lines(original_content)
                self._register_file_text(file_path, content)
                # print(f"  Removed {len(content_lines) - len(non_comment_lines)} comment lines from file {file_path}")
                
            except Exception as e:
                print(f"Error: Failed to read file {file_path}: {str(e)}")
                return f"% File reading error: {file_path}"
            
            # Recursively process all \input commands
            def replace_input(match):
                input_file = match.group(1).strip()
                print(f"  Found input command: \\input{{{input_file}}}")
                
                # Add extension (if not present)
                if not os.path.splitext(input_file)[1]:
                    input_file += ".tex"
                
                # Build complete path for the referenced file
                # First try path relative to current file
                input_path = os.path.join(current_directory, input_file)
                
                if not os.path.exists(input_path):
                    # If relative path doesn't exist, try path relative to original directory
                    input_path = os.path.join(directory, input_file)
                
                if os.path.exists(input_path):
                    print(f"  Loading file: {input_path}")
                    # Recursively process referenced file
                    # Use directory of referenced file as current directory
                    input_dir = os.path.dirname(input_path)
                    return process_input_commands(input_path, input_dir)
                else:
                    print(f"Warning: Referenced file not found {input_path}")
                    return f"% Referenced file not found: {input_file}"
            
            # Replace all \input commands
            processed_content = re.sub(r'\\input\{([^}]+)\}', replace_input, content)
            print(f"Completed processing file: {file_path}")
            return processed_content
        
        # Start processing from paper.tex
        result = process_input_commands(paper_path, directory)
        print(f"LaTeX processing complete, merged text length: {len(result)} characters")
        with open("merged_paper.tex", "w", encoding="utf-8") as f:
            f.write(result)
        return result

    def extract_sections(self, latex_text):
        """
        Extract sections from LaTeX text structure, such as abstract and chapters.
        Returns a dictionary with section titles as keys and corresponding content as values.
        """
        sections = {}
        # Extract abstract
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', latex_text, re.DOTALL)
        if abstract_match:
            sections["abstract"] = abstract_match.group(1).strip()
        
        # Extract main sections
        section_matches = re.findall(r'\\section\{(.+?)\}(.*?)(?=\\section|\\end\{document\})', latex_text, re.DOTALL)
        for title, content in section_matches:
            sections[title.strip()] = content.strip()
            
        # Extract subsections (optional, enable as needed)
        # subsection_matches = re.findall(r'\\subsection\{(.+?)\}(.*?)(?=\\subsection|\\section|\\end\{document\})', latex_text, re.DOTALL)
        # for title, content in subsection_matches:
        #     sections["sub:" + title.strip()] = content.strip()
            
        # Extract subsubsections (optional, enable as needed)
        # subsubsection_matches = re.findall(r'\\subsubsection\{(.+?)\}(.*?)(?=\\subsubsection|\\subsection|\\section|\\end\{document\})', latex_text, re.DOTALL)
        # for title, content in subsubsection_matches:
        #     sections["subsub:" + title.strip()] = content.strip()
        
        return sections

    # ---------------- Static Checks ----------------
    def check_title_presence(self):
        """Check if paper title exists"""
        match = re.search(r'\\title\{(.+?)\}', self.latex_text, re.DOTALL)
        title = match.group(1).strip() if match else None
        result = {
            'check': 'Paper title exists',
            'result': bool(title),
            'detail': title if title else "Title not found."
        }
        self.report['title_presence'] = result
        return title

    def check_section_titles_descriptive(self):
        """Check if section titles are descriptive (avoid single-word titles)"""
        # Capture titles of all section levels
        title_matches = re.findall(r'\\(section|subsection|subsubsection)\{(.+?)\}', self.latex_text)
        # Extract actual title text from matches
        titles = [match[1] for match in title_matches]
        
        # Define a list of common single-word titles that don't need checking
        excluded_titles = ["Introduction", "Background", "Conclusion", "Motivation", "Abstract", "Related Work", "Discussion", "Results" "Appendix"]
        
        issues = []
        problematic_titles = []
        for title in titles:
            # Skip commands or markers starting with backslash
            if title.strip().startswith('\\'):
                continue
                
            # Skip titles in the exclusion list
            if title.strip() in excluded_titles:
                continue
                
            words = title.split()
            if len(words) == 1:
                issues.append(f"Section title '{title}' has only one word.")
                problematic_titles.append(title)
                
        result = {
            'check': 'Section title descriptiveness',
            'result': (len(issues) == 0),
            'detail': issues if issues else "All section titles are sufficiently descriptive or are allowed single-word titles.",
            'failed_content': problematic_titles if problematic_titles else None
        }
        self.report['section_titles_descriptive'] = result
        return titles

    def check_section_titles_capitalization(self):
        """Check if all section titles have consistent capitalization (either all capitalized or all lowercase)"""
        # Modify regex to capture different section level titles
        title_matches = re.findall(r'\\(section|subsection|subsubsection)\{(.+?)\}', self.latex_text)
        
        if not title_matches:
            result = {
                'check': 'Section title capitalization',
                'result': False,
                'detail': "No section titles found.",
                'failed_content': None
            }
            self.report['section_titles_capitalization'] = result
            return False

        # Extract actual title text from matches (second element of tuple) and filter out commands
        titles = [match[1] for match in title_matches if not match[1].strip().startswith('\\')]
        
        if not titles:
            result = {
                'check': 'Section title capitalization',
                'result': False,
                'detail': "No valid section titles available for checking.",
                'failed_content': None
            }
            self.report['section_titles_capitalization'] = result
            return False
        
        # Define a list of words that should typically be lowercase in titles
        lowercase_words = {
            # Articles
            'a', 'an', 'the',
            # Coordinating conjunctions
            'and', 'but', 'or', 'nor', 'for', 'so', 'yet',
            # Common prepositions (prepositions with 3 letters or fewer are usually lowercase)
            'in', 'to', 'of', 'at', 'by', 'up', 'for', 'on',
            # Other common prepositions (may vary according to specific style guides)
            'with', 'into', 'onto', 'from', 'about', 'under',
            'above', 'after', 'over', 'till', 'down', 'past',
            'off', 'via',
            'but', 'down', 'less', 'out', 'than', 'through',
            'save', 'per', 'plus'
        }
        
        def analyze_title_part(s, is_first_part=True):
            """Analyze title part, return a list of capitalization errors"""
            words = s.split()
            errors = []
            
            for i, word in enumerate(words):
                if not word:
                    continue
                
                # Ignore punctuation
                word_clean = ''.join(c for c in word if c.isalnum())
                if not word_clean:
                    continue
                
                # Skip words that are pure numbers
                if word_clean.isdigit():
                    continue
                
                # First word should always be capitalized
                if i == 0:
                    if not word_clean[0].isupper():
                        correct_word = word_clean[0].upper() + word_clean[1:]
                        errors.append(f"'{word}' should be capitalized as '{correct_word}'")
                # Last word should also be capitalized, but only if it's the first part or the entire title
                elif i == len(words) - 1 and is_first_part:
                    if not word_clean[0].isupper():
                        correct_word = word_clean[0].upper() + word_clean[1:]
                        errors.append(f"'{word}' should be capitalized as '{correct_word}'")
                # For other words, if not in lowercase words list, should be capitalized
                elif word_clean.lower() not in lowercase_words and not word_clean[0].isupper():
                    correct_word = word_clean[0].upper() + word_clean[1:]
                    errors.append(f"'{word}' should be capitalized as '{correct_word}'")
                # Check words that should be lowercase but are capitalized
                elif word_clean.lower() in lowercase_words and word_clean[0].isupper() and i != 0:
                    correct_word = word_clean.lower()
                    errors.append(f"'{word}' should be lowercase as '{correct_word}'")
            
            return errors
        
        def is_title_case(s):
            """Check if title follows title case format, handling colon-separated cases"""
            # If there's a colon, split the title into multiple parts and check each
            if ':' in s:
                parts = s.split(':')
                for i, part in enumerate(parts):
                    # Each part should follow title case rules
                    # Spaces before the first part and after the last part might be trimmed
                    part = part.strip()
                    if part and not is_title_part_case(part, True):  # First word of each part should be capitalized
                        return False
                return True
            else:
                # No colon case, check directly
                return is_title_part_case(s)
        
        def is_title_part_case(s, is_first_part=True):
            """Check if a part of the title follows title case format"""
            words = s.split()
            if not words:
                return True
            
            incorrect_case_words = []
            
            # Check each word, excluding common lowercase words and numbers
            for i, word in enumerate(words):
                if not word:
                    continue
                
                # Ignore punctuation
                word_clean = ''.join(c for c in word if c.isalnum())
                if not word_clean:
                    continue
                
                # Skip words that are pure numbers
                if word_clean.isdigit():
                    continue
                
                # First word should always be capitalized
                if i == 0:
                    if not word_clean[0].isupper():
                        incorrect_case_words.append((word, True))  # True means should be capitalized
                # Last word should also be capitalized, but only if it's the first part or the entire title
                elif i == len(words) - 1 and is_first_part:
                    if not word_clean[0].isupper():
                        incorrect_case_words.append((word, True))  # True means should be capitalized
                # For other words, if not in lowercase words list, should be capitalized
                elif word_clean.lower() not in lowercase_words and not word_clean[0].isupper():
                    incorrect_case_words.append((word, True))  # True means should be capitalized
                # Check words that should be lowercase but are capitalized
                elif word_clean.lower() in lowercase_words and word_clean[0].isupper() and i != 0:
                    incorrect_case_words.append((word, False))  # False means should be lowercase
            
            return len(incorrect_case_words) == 0

        # Analyze each title, collect error information
        title_errors = {}
        for title in titles:
            errors = []
            
            # If there's a colon, split the title into multiple parts and check each
            if ':' in title:
                parts = title.split(':')
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        part_errors = analyze_title_part(part, i == 0)
                        if part_errors:
                            errors.extend(part_errors)
            else:
                # No colon case, check directly
                errors = analyze_title_part(title)
            
            if errors:
                title_errors[title] = errors
        
        # Check if consistent capitalization style is used (title case or all lowercase)
        title_case_count = sum(1 for title in titles if is_title_case(title))
        lowercase_count = sum(1 for title in titles if title.lower() == title)
        
        is_consistent = (title_case_count == len(titles)) or (lowercase_count == len(titles))
        
        if is_consistent and not title_errors:
            detail = "All section titles have consistent and correct capitalization."
            res = True
            failed_content = None
        else:
            # Collect inconsistent or problematic titles
            inconsistent_titles = []
            if title_case_count > lowercase_count:
                # Most are title case, find those that don't conform
                inconsistent_titles = [title for title in titles if not is_title_case(title)]
                preferred_style = "Title Case"
            else:
                # Most are lowercase, find those that don't conform
                inconsistent_titles = [title for title in titles if title.lower() != title]
                preferred_style = "lowercase"
            
            if title_errors:
                error_details = []
                for title, errors in title_errors.items():
                    error_details.append(f"Capitalization issues in title '{title}':")
                    for error in errors:
                        error_details.append(f"  - {error}")
                
                detail = f"Section titles have capitalization issues. Recommended style: {preferred_style}.\nDetailed errors: " + "\n".join(error_details)
            else:
                detail = f"Section titles have inconsistent capitalization. Recommended style: {preferred_style}."
            
            res = False
            failed_content = {
                "inconsistent_titles": inconsistent_titles if inconsistent_titles else [],
                "title_errors": title_errors
            }
            
        result = {
            'check': 'Section title capitalization',
            'result': res,
            'detail': detail,
            'failed_content': failed_content
        }
        self.report['section_titles_capitalization'] = result
        return res

    def check_section_buffer(self):
        """Check if section titles are directly followed by subsection titles (should include introductory text)"""
        # Get all section and subsection commands and their positions in the text
        section_pattern = r'\\(section|subsection|subsubsection)\{(.+?)\}'
        section_matches = []
        
        for match in re.finditer(section_pattern, self.latex_text):
            section_type = match.group(1)  # section, subsection or subsubsection
            section_title = match.group(2)  # section title
            position = match.start()  # position in the original text
            section_matches.append((section_type, section_title, position))
        
        # Sort by position in the original text
        section_matches.sort(key=lambda x: x[2])
        
        issues = []
        problematic_sections = []
        
        # Check if there are sections directly followed by subsections without content buffer
        for i in range(len(section_matches) - 1):
            current_level = section_matches[i][0]
            next_level = section_matches[i+1][0]
            current_title = section_matches[i][1]
            next_title = section_matches[i+1][1]
            
            # Only check if section is directly followed by subsection or subsection is directly followed by subsubsection
            if (current_level == 'section' and next_level == 'subsection') or \
               (current_level == 'subsection' and next_level == 'subsubsection'):
                
                # Calculate the text between the two tags
                current_pos = section_matches[i][2]
                next_pos = section_matches[i+1][2]
                
                # Find the end position of the current section title (i.e., the position after the closing brace)
                section_start_match = re.search(r'\{' + re.escape(current_title) + r'\}', self.latex_text[current_pos:])
                if section_start_match:
                    content_start = current_pos + section_start_match.end()
                    
                    # Extract the text between the two tags
                    between_text = self.latex_text[content_start:next_pos].strip()
                    
                    # Ignore text that only contains labels, such as \label{...}
                    between_text_no_labels = re.sub(r'\\label\{[^}]*\}', '', between_text).strip()
                    
                    # If there is only whitespace or only labels between the tags, consider there is no introductory text
                    if len(between_text_no_labels) < 10:
                        if current_level == 'section':
                            issue = f"Section '{current_title}' is directly followed by subsection '{next_title}' without introductory text."
                        else:
                            issue = f"Subsection '{current_title}' is directly followed by subsubsection '{next_title}' without introductory text."
                        
                        issues.append(issue)
                        problematic_sections.append(f"{current_title} -> {next_title}")
                
        result = {
            'check': 'Section Buffer Check',
            'result': (len(issues) == 0),
            'detail': issues if issues else "Sections and subsections have appropriate buffer content.",
            'failed_content': problematic_sections if problematic_sections else None
        }
        self.report['section_buffer'] = result
        return (len(issues) == 0)

    def _static_check_figures_in_content(self, rel_path, content, full_text):
        """Run figure static checks for a single LaTeX source file."""
        non_commented_text = self._remove_comment_only_lines(content)
        if not non_commented_text.strip():
            return [], []

        issues = []
        failed_messages = []
        figure_labels = set()

        def record(message):
            issues.append(message)
            failed_messages.append(f"{rel_path}: {message}")
            if DEBUG:
                print(f"[FigureCheck][{rel_path}] {message}")

        def extract_content_with_braces(text, command):
            pattern = f"{command}" + r"\{((?:[^{}]|(?:\{[^{}]*\}))*)\}"
            return re.search(pattern, text, re.DOTALL)

        def preview_text(text, length):
            snippet = text.strip()
            return snippet[:length] + ("..." if len(snippet) > length else "")

        figures_with_options = re.findall(r'\\begin\{figure\}\[([^\]]+)\](.*?)\\end\{figure\}', non_commented_text, re.DOTALL)
        figures_without_options = re.findall(r'\\begin\{figure\}(?!\[)(.*?)\\end\{figure\}', non_commented_text, re.DOTALL)

        if DEBUG:
            print(f"Found {len(figures_with_options)} figures with options and {len(figures_without_options)} figures without options in {rel_path}")

        # Process figures with options
        for placement, content_block in figures_with_options:
            caption_match = extract_content_with_braces(content_block, r"\\caption")
            caption_text = caption_match.group(1) if caption_match else ""
            caption_preview_short = preview_text(caption_text or "Figure without caption", 30)

            if 'h' not in placement:
                record(f"Figure missing [h] placement option: '{caption_preview_short}'")

            if not caption_match:
                record(f"Figure missing caption (options: [{placement}])")

            label_match = extract_content_with_braces(content_block, r"\\label")
            if label_match:
                label_text = label_match.group(1)
                figure_labels.add(label_text)
                if caption_match and label_match.start() < caption_match.start():
                    record(f"Figure label '{label_text}' should be placed after the caption")
                if not caption_match:
                    record(f"Figure label '{label_text}' exists but has no associated caption")
            elif caption_match:
                record(f"Figure has caption but missing label: '{caption_preview_short}'")

        # Process figures without options
        for content_block in figures_without_options:
            caption_match = extract_content_with_braces(content_block, r"\\caption")
            caption_text = caption_match.group(1) if caption_match else ""
            caption_preview_short = preview_text(caption_text or "Figure without caption", 30)

            record(f"Figure missing [h] placement option: '{caption_preview_short}'")

            if not caption_match:
                record("Figure missing caption (no placement options)")

            label_match = extract_content_with_braces(content_block, r"\\label")
            if label_match:
                label_text = label_match.group(1)
                figure_labels.add(label_text)
                if caption_match and label_match.start() < caption_match.start():
                    record(f"Figure label '{label_text}' should be placed after the caption")
                if not caption_match:
                    record(f"Figure label '{label_text}' exists but has no associated caption")
            elif caption_match:
                record(f"Figure has caption but missing label: '{caption_preview_short}'")

        # Check label references using the full merged LaTeX text
        full_text_clean = self._remove_comment_only_lines(full_text)
        for label in figure_labels:
            if not self._is_label_referenced(label, full_text_clean):
                record(f"Figure label '{label}' is not referenced in the text.")

        return issues, failed_messages

    @staticmethod
    def _is_label_referenced(label, text):
        """Check if a figure label is referenced anywhere in the LaTeX text."""
        patterns = [
            rf'\\ref\{{{re.escape(label)}\}}',
            rf'\\autoref\{{{re.escape(label)}\}}',
            rf'\\cref\{{{re.escape(label)}\}}',
            rf'\\Cref\{{{re.escape(label)}\}}'
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def static_check_figures(self):
        """Figure static check: verify [h] placement, captions, labels and references per source file."""
        issues = []
        failed_messages = []
        full_text = self.latex_text

        for rel_path, content in self._iterate_latex_files():
            file_issues, file_failed_messages = self._static_check_figures_in_content(rel_path, content, full_text)
            issues.extend(file_issues)
            failed_messages.extend(file_failed_messages)

        result = {
            'check': 'Figure Static Check',
            'result': (len(issues) == 0),
            'detail': issues if issues else "All figures passed static checks.",
            'failed_content': failed_messages if failed_messages else None
        }
        self.report['figures_static'] = result
        return issues

    def static_check_numbers_spelling(self):
        """Check if numbers less than 10 appearing alone are spelled out as words, per source file."""
        issues = []
        failed_messages = []
        pattern = r'(?<!\S)([1-9])(?!\S)'

        for rel_path, content in self._iterate_latex_files():
            matches = list(re.finditer(pattern, content))
            if not matches:
                continue

            digits_in_file = sorted({match.group(1) for match in matches})
            issues.append(f"{rel_path}: Found standalone numeric digits: {', '.join(digits_in_file)}. Consider spelling them out.")

            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 30)
                context = content[start:end]
                failed_messages.append(f"{rel_path}: Number '{match.group(0)}' appears in: '...{context}...'")

        result = {
            'check': 'Number Spelling Check',
            'result': (len(issues) == 0),
            'detail': issues if issues else "All numbers are properly spelled out or not standalone.",
            'failed_content': failed_messages if failed_messages else None
        }
        self.report['numbers_spelling'] = result
        return issues

    def static_check_informal_words(self):
        """Check for informal word usage"""
        informal_words = {
            "see": "observe",
            "show": "demonstrate",
            "like": "such as"
        }
        issues = []
        failed_messages = []

        for rel_path, content in self._iterate_latex_files():
            for word, suggestion in informal_words.items():
                matches = list(re.finditer(r'\b' + re.escape(word) + r'\b', content, re.IGNORECASE))
                if not matches:
                    continue

                issues.append(f"{rel_path}: Found informal word '{word}'. Consider replacing with '{suggestion}'.")

                for match in matches:
                    start = max(0, match.start() - 30)
                    end = min(len(content), match.end() + 30)
                    context = content[start:end]
                    failed_messages.append(f"{rel_path}: Found informal word '{word}' in context '...{context}...'")

        result = {
            'check': 'Informal Word Usage',
            'result': (len(issues) == 0),
            'detail': issues if issues else "No informal words detected.",
            'failed_content': failed_messages if failed_messages else None
        }
        self.report['informal_words'] = result
        return issues

    def static_spell_check(self):
        """Use language_tool_python to spell check the LaTeX file (with simple filtering to remove LaTeX commands)"""
        text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', self.latex_text)
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(text)
        
        issues = []
        spelling_errors = []
        
        for m in matches:
            issues.append(m.message)
            context = text[max(0, m.offset-30):min(len(text), m.offset+m.errorLength+30)]
            error_text = text[m.offset:m.offset+m.errorLength]
            spelling_errors.append({
                'error': error_text,
                'message': m.message,
                'context': f"'...{context}...'",
                'suggestions': m.replacements[:3] if m.replacements else []
            })
            
        result = {
            'check': 'Spell Check',
            'result': (len(issues) == 0),
            'detail': issues if issues else "No spelling errors found.",
            'failed_content': spelling_errors if spelling_errors else None
        }
        self.report['spell_check'] = result
        return issues

    def static_check_italics_usage(self):
        """Simple check for italics usage to determine if overused"""
        italic_count = len(re.findall(r'\\textit\{', self.latex_text))
        
        # Find all italic text
        italic_texts = re.findall(r'\\textit\{(.*?)\}', self.latex_text)
        
        if italic_count > 5:
            result = {
                'check': 'Italics Usage',
                'result': False,
                'detail': f"Italics used: {italic_count} times.",
                'failed_content': italic_texts
            }
            self.report['italics_usage'] = result
            return False
        else:
            result = {
                'check': 'Italics Usage',
                'result': True,
                'detail': "Italics usage is within acceptable range.",
                'failed_content': None
            }
            self.report['italics_usage'] = result
            return True

    # ---------------- LLM Checks ----------------
    def call_llm(self, prompt, max_tokens=2000):
        """Dispatch a text-only prompt to the configured LLM provider."""
        if self.skip_llm:
            return "LLM checks skipped."
        if not self.provider:
            raise ValueError("LLM model is not configured.")

        try:
            if self.provider == "xai":
                api_key = self._get_api_key()
                if not api_key:
                    raise ValueError("XAI_API_KEY is not set. Provide it via environment or --api_key.")
                if "xai_text" not in self._clients:
                    self._clients["xai_text"] = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                response = self._clients["xai_text"].completions.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    prompt=prompt,
                    temperature=0
                )
                return response.choices[0].text.strip()

            if self.provider == "gemini":
                if genai is None:
                    raise ImportError("google-generativeai is not installed. Install it to use Gemini models.")
                api_key = self._get_api_key()
                if not api_key:
                    raise ValueError("Gemini API key is not set. Provide it via environment or --api_key.")
                if "gemini_text" not in self._clients:
                    genai.configure(api_key=api_key)
                    self._clients["gemini_text"] = genai.GenerativeModel(self.model_name)
                response = self._clients["gemini_text"].generate_content(prompt)
                if hasattr(response, "text") and response.text:
                    return response.text
                candidates = getattr(response, "candidates", [])
                collected = []
                for candidate in candidates:
                    for part in getattr(candidate, "content", []):
                        text_part = getattr(part, "text", None)
                        if text_part:
                            collected.append(text_part)
                return "\n".join(collected) if collected else ""

            if self.provider == "openai":
                api_key = self._get_api_key()
                if not api_key:
                    raise ValueError("OpenAI API key is not set. Provide it via environment or --api_key.")
                if "openai_text" not in self._clients:
                    self._clients["openai_text"] = OpenAI(api_key=api_key)
                response = self._clients["openai_text"].chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0
                )
                return response.choices[0].message.content

            raise ValueError(f"Unsupported provider '{self.provider}'.")
        except Exception as e:
            return f"LLM call failed: {e}"

    def llm_evaluate_titles(self, title, section_titles):
        prompt = f"""Please evaluate the quality of the paper title section based on the following information:
Paper title: {title}
Section titles list: {section_titles}

Please check the following items:
1. Is the paper title concise, attractive, and highlighting the important content of the article?
2. Are section titles descriptive (avoiding single-word titles)?
3. Is the capitalization of all section titles consistent?

Please provide a "pass" or "fail" assessment for each item, along with a brief explanation. Ignore all LaTeX commands."""
        if DEBUG:
            print(prompt)
        return self.call_llm(prompt)

    def llm_evaluate_introduction(self):
        # Using extract_sections to get the introduction part
        intro_text = self.sections.get("Introduction", "")
        prompt = f"""Please evaluate whether the paper's introduction meets the following criteria:
1. Provides importance and broad introduction to the topic;
2. Clearly states the specific problem to be solved;
3. Explains the challenges of solving the problem;
4. Outlines the high-level solution;
5. Details what the reader will read;
6. Lists the main contributions.

Please provide a "pass" or "fail" assessment for each item, along with a brief explanation. The introduction content is as follows:
{intro_text}"""
        if DEBUG:
            print(prompt)
        return self.call_llm(prompt)

    def llm_evaluate_writing_for_section(self, section_title, section_content):
        """
        Evaluates the writing quality of a single section,
        takes the section title and content, and returns the LLM check result.
        """
        prompt = f"""Please evaluate the writing quality of the section "{section_title}" based on the following criteria:
1. Do spell and grammar check.
2. Does the body of the paper use present tense?
3. Is the tense usage consistent, avoiding frequent switching within the same paragraph?
4. Is the use of "which" and "that" correctly distinguished?
5. Are long sentences and passive voice avoided?
6. Is informal wording avoided (e.g., using "observe" instead of "see", using "demonstrate" instead of "show", using "such as" instead of "like")?
7. Are numbers less than 10 written out as words?

Please provide a "pass" or "fail" assessment for each item, along with a brief explanation.

The content of section "{section_title}" is as follows:
{section_content}"""
        if DEBUG:
            print(prompt)
        # return self.call_llm(prompt)
        return self.call_llm(prompt)

    def call_vision_llm(self, prompt, base64_image, max_tokens=2000):
        """Dispatch a multimodal prompt (text + image) to the configured provider."""
        if self.skip_llm:
            return "LLM checks skipped."
        if not self.provider:
            raise ValueError("LLM model is not configured.")

        vision_model = self._get_vision_model()
        try:
            if self.provider == "xai":
                api_key = self._get_api_key()
                if not api_key:
                    raise ValueError("XAI_API_KEY is not set. Provide it via environment or --api_key.")
                if "xai_vision" not in self._clients:
                    self._clients["xai_vision"] = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                response = self._clients["xai_vision"].chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}" }}
                            ]
                        }
                    ],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            if self.provider == "gemini":
                if genai is None:
                    raise ImportError("google-generativeai is not installed. Install it to use Gemini models.")
                api_key = self._get_api_key()
                if not api_key:
                    raise ValueError("Gemini API key is not set. Provide it via environment or --api_key.")
                if "gemini_vision" not in self._clients:
                    genai.configure(api_key=api_key)
                    self._clients["gemini_vision"] = genai.GenerativeModel(vision_model)
                import base64 as _base64
                image_bytes = _base64.b64decode(base64_image)
                response = self._clients["gemini_vision"].generate_content(
                    [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                    ]
                )
                if hasattr(response, "text") and response.text:
                    return response.text
                candidates = getattr(response, "candidates", [])
                collected = []
                for candidate in candidates:
                    for part in getattr(candidate, "content", []):
                        text_part = getattr(part, "text", None)
                        if text_part:
                            collected.append(text_part)
                return "\n".join(collected) if collected else ""

            if self.provider == "openai":
                api_key = self._get_api_key()
                if not api_key:
                    raise ValueError("OpenAI API key is not set. Provide it via environment or --api_key.")
                if "openai_multimodal" not in self._clients:
                    self._clients["openai_multimodal"] = OpenAI(api_key=api_key)
                response = self._clients["openai_multimodal"].responses.create(
                    model=vision_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "input_image", "image": {"b64": base64_image}}
                            ]
                        }
                    ],
                    max_output_tokens=max_tokens
                )
                return getattr(response, "output_text", None) or ""

            raise ValueError(f"Unsupported provider '{self.provider}'.")
        except Exception as e:
            return f"LLM vision call failed: {e}"

    def llm_evaluate_figures_from_pdf(self):
        """
        Uses pdf2image to convert each page of the PDF to an image,
        then calls the configured vision-capable model to check the figures.
        """
        import base64
        import tempfile

        if self.skip_llm:
            return "LLM checks skipped."

        images = convert_from_path(self.pdf_path)
        results = {}
    
        # Create temp directory to save images
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, image in enumerate(images):
                # Save the image temporarily
                image_path = f"{temp_dir}/page_{i+1}.png"
                image.save(image_path)
                
                # Encode image to base64
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Prepare prompt for figure evaluation
                prompt = f"""Please evaluate the quality of figures on page {i+1} of the PDF, identify any issues and provide suggestions for improvement. If there are no figures on the page, please indicate this.
1. Do the figures use a consistent style?
2. Check X-axis, Y-axis labels, legends and ticks: font size, capitalization, units and terms used should be consistent across all
3. Figure title - descriptive (not just “Strong Scaling”, etc.)
4. Key/legend - order of legend should be same as order of bars/lines
5. Use consistent coloring scheme, use patterns in addition to colors
6. Figure captions should be detailed, mention machine (if multiple machines are used), number of processes/cores/GPUs (if that is not clear from the figure)
7. Figures/tables must be referenced in the text and the text should tell the reader how to read it and what to take away from it
8. Figures should appear right after (using [h]) where they are mentioned (in most cases except layout issues)"""
                
                try:
                    # Store the result from the configured multimodal model
                    results[f"page_{i+1}"] = self.call_vision_llm(prompt, base64_image)

                except Exception as e:
                    results[f"page_{i+1}"] = f"Image analysis failed: {e}"
                # break

        return results

    def run_all_checks(self):
        # First perform all static checks
        title = self.check_title_presence()
        section_titles = self.check_section_titles_descriptive()
        self.check_section_titles_capitalization()
        self.check_section_buffer()
        self.static_check_figures()
        self.static_check_numbers_spelling()
        self.static_check_informal_words()
        self.static_check_italics_usage()
        
        
        # Currently unable to handle latex format, will report a large number of spelling errors
        # self.static_spell_check()

        if not self.skip_llm:
            # Then call LLM for subjective evaluation (by module)
            # Example: LLM checks for titles, introduction, and figures
            self.report['LLM_Titles'] = self.llm_evaluate_titles(title, section_titles)
            self.report['LLM_Introduction'] = self.llm_evaluate_introduction()
            self.report['LLM_Figures_Vision'] = self.llm_evaluate_figures_from_pdf()

            # # # Check writing quality for each section
            llm_writing_results = {}
            for section_title, section_content in self.sections.items():
                result = self.llm_evaluate_writing_for_section(section_title, section_content)
                llm_writing_results[section_title] = result
                
            self.report['LLM_Writing'] = llm_writing_results
        else:
            skip_message = "LLM checks skipped."
            self.report['LLM_Titles'] = skip_message
            self.report['LLM_Introduction'] = skip_message
            self.report['LLM_Figures_Vision'] = skip_message
            self.report['LLM_Writing'] = {section_title: skip_message for section_title in self.sections}

        return self.report

def main():
    parser = argparse.ArgumentParser(description="Check paper drafts against quality checklist using LLM and static methods.")
    parser.add_argument("--pdf", required=True, help="Path to the paper draft PDF file")
    parser.add_argument("--latex", required=True, help="Path to LaTeX source file or directory")
    parser.add_argument(
        "--model",
        required=False,
        help=f"LLM model to use. Supported models: {', '.join(SUPPORTED_MODELS.keys())}"
    )
    parser.add_argument(
        "--api_key",
        required=False,
        help="API key for the selected LLM provider (optional if set via environment variable)."
    )
    args = parser.parse_args()

    model_name = args.model
    skip_llm = False

    if not model_name:
        choice = input("No --model specified. Do you want to skip LLM checks? (y/N): ").strip().lower()
        if choice in ("y", "yes"):
            skip_llm = True
        else:
            available_models = ", ".join(SUPPORTED_MODELS.keys())
            while not model_name:
                candidate = input(f"Please specify the model name ({available_models}): ").strip()
                if candidate in SUPPORTED_MODELS:
                    model_name = candidate
                else:
                    print(f"Invalid model name '{candidate}'. Supported models: {available_models}")
    else:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS.keys())}")

    evaluator = ChecklistEvaluator(
        args.pdf,
        args.latex,
        api_key=args.api_key,
        model_name=model_name,
        skip_llm=skip_llm
    )
    report = evaluator.run_all_checks()

    # Separate static checks (with 'result') from LLM checks
    static_checks = []
    llm_checks = []
    other_entries = []
    for check_name, check_data in report.items():
        if isinstance(check_data, dict) and 'result' in check_data:
            static_checks.append((check_name, check_data))
        elif check_name.startswith('LLM_'):
            llm_checks.append((check_name, check_data))
        else:
            other_entries.append((check_name, check_data))
    
    # Save the report as a markdown document
    markdown_output = "# Paper Quality Check Report\n\n"

    # Static summary
    markdown_output += "## Static Check Summary\n\n"
    if static_checks:
        total_checks = len(static_checks)
        passed_checks = sum(1 for _, data in static_checks if data.get('result') is True)
        pass_rate = (passed_checks / total_checks) * 100
        markdown_output += f"- **Pass Rate**: {pass_rate:.1f}% ({passed_checks}/{total_checks} checks passed)\n\n"
    else:
        markdown_output += "No static checks were executed.\n\n"

    # Static detailed results
    markdown_output += "## Detailed Results\n\n"
    if not static_checks:
        markdown_output += "No static check results available.\n\n"
    else:
        for check_name, check_data in static_checks:
            status = "✅ PASS" if check_data['result'] is True else "❌ FAIL"
            check_title = check_data.get('check', check_name)
            markdown_output += f"### {status}: {check_title}\n\n"
            markdown_output += f"**Details**: {check_data.get('detail', 'No detailed information')}\n\n"

            failed_content = check_data.get('failed_content')
            if failed_content:
                markdown_output += "**Error content**:\n\n"
                if isinstance(failed_content, list):
                    for i, item in enumerate(failed_content, 1):
                        markdown_output += f"{i}. {item}\n"
                elif isinstance(failed_content, dict):
                    for key, value in failed_content.items():
                        markdown_output += f"- **{key}**:\n"
                        if isinstance(value, list):
                            for i, item in enumerate(value, 1):
                                markdown_output += f"  {i}. {item}\n"
                        else:
                            markdown_output += f"  {value}\n"
                else:
                    markdown_output += f"{failed_content}\n"
                markdown_output += "\n"

    # LLM section
    markdown_output += "## LLM Check\n\n"
    if not llm_checks:
        markdown_output += "No LLM checks were executed.\n\n"
    else:
        for check_name, check_data in llm_checks:
            markdown_output += f"### {check_name}\n\n"
            if isinstance(check_data, dict):
                for section, section_data in check_data.items():
                    if isinstance(section_data, dict):
                        markdown_output += f"#### {section}\n\n"
                        for key, value in section_data.items():
                            markdown_output += f"**{key}**: {value}\n\n"
                    else:
                        markdown_output += f"**{section}**: {section_data}\n\n"
            else:
                markdown_output += f"{check_data}\n\n"

    # Any other entries
    for check_name, check_data in other_entries:
        markdown_output += f"## {check_name}\n\n{check_data}\n\n"
    
    # Write the markdown report to a file
    report_filename = os.path.splitext(os.path.basename(args.pdf))[0] + "_quality_report.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(markdown_output)

    # Output the check report in JSON format, highlighting failed checks and their error content
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    print("\n=== Detailed Report of Failed Checks ===")
    for check_name, check_data in report.items():
        if isinstance(check_data, dict) and 'result' in check_data and check_data['result'] is False:
            print(f"\n[❌] {check_data.get('check', check_name)}:")
            print(f"   Details: {check_data.get('detail', 'No detailed information')}")
            
            if 'failed_content' in check_data and check_data['failed_content']:
                print("   Error content:")
                failed_content = check_data['failed_content']
                
                if isinstance(failed_content, list):
                    for i, item in enumerate(failed_content, 1):
                        print(f"   {i}. {item}")
                elif isinstance(failed_content, dict):
                    for key, value in failed_content.items():
                        print(f"   - {key}:")
                        if isinstance(value, list):
                            for i, item in enumerate(value, 1):
                                print(f"     {i}. {item}")
                        else:
                            print(f"     {value}")
                else:
                    print(f"   {failed_content}")

    print(f"\n\033[32mMarkdown report saved to: {report_filename}\033[0m")
if __name__ == "__main__":
    main()
