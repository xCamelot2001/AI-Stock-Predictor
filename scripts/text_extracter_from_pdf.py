import os
import re
import pandas as pd
from datetime import datetime
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from pathlib import Path
import logging

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class FinancialPDFExtractor:
    def __init__(self, pdf_directory, output_file="financial_news_data.csv"):
        """
        Initialize the PDF extractor for financial news
        
        Args:
            pdf_directory (str): Path to directory containing PDF files
            output_file (str): Output CSV filename
        """
        self.pdf_directory = Path(pdf_directory)
        self.output_file = output_file
        self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.stop_words = set(stopwords.words('english'))
        self.max_sequence_length = 128  # FinBERT max sequence length
        
        # Financial keywords to identify relevant content
        self.financial_keywords = [
            'stock', 'share', 'market', 'trading', 'investment', 'portfolio',
            'earnings', 'revenue', 'profit', 'loss', 'dividend', 'nasdaq',
            'dow', 'sp500', 'ftse', 'volatility', 'price', 'valuation',
            'merger', 'acquisition', 'ipo', 'financial', 'economic',
            'gdp', 'inflation', 'interest rate', 'federal reserve', 'fed',
            'quarter', 'quarterly', 'annual', 'fiscal', 'bull', 'bear'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path, method='pdfplumber'):
        """
        Extract text from PDF using specified method
        
        Args:
            pdf_path (Path): Path to PDF file
            method (str): Extraction method ('pdfplumber', 'pymupdf', 'pypdf2')
        
        Returns:
            str: Extracted text
        """
        text = ""
        
        try:
            if method == 'pdfplumber':
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            
            elif method == 'pymupdf':
                doc = fitz.open(pdf_path)
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text += page.get_text() + "\n"
                doc.close()
                
            elif method == 'pypdf2':
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                        
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            
        return text
    
    def extract_date_from_filename_or_content(self, filename, content):
        """
        Extract date from filename or PDF content
        
        Args:
            filename (str): PDF filename
            content (str): PDF text content
        
        Returns:
            datetime or None: Extracted date
        """
        # Common date patterns
        date_patterns = [
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',  # YYYY-MM-DD or YYYY/MM/DD
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',  # MM-DD-YYYY or MM/DD/YYYY
            r'\b(\d{1,2}\s+\w+\s+\d{4})\b',       # DD Month YYYY
            r'\b(\w+\s+\d{1,2},?\s+\d{4})\b',     # Month DD, YYYY
        ]
        
        # Try filename first
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    return pd.to_datetime(match.group(1))
                except:
                    continue
        
        # Try content (first 1000 characters)
        content_sample = content[:1000]
        for pattern in date_patterns:
            matches = re.findall(pattern, content_sample)
            for match in matches:
                try:
                    return pd.to_datetime(match)
                except:
                    continue
        
        # Return None if no date found
        return None
    
    def preprocess_text(self, text):
        """
        Preprocess text for FinBERT according to your project requirements
        
        Args:
            text (str): Raw text
        
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and punctuation (keeping spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stopwords (optional - sometimes keeping them helps with context)
        # Uncomment if you want to remove stopwords:
        # words = word_tokenize(text)
        # text = ' '.join([word for word in words if word not in self.stop_words])
        
        return text.strip()
    
    def is_financial_content(self, text):
        """
        Check if text contains financial content
        
        Args:
            text (str): Text to check
        
        Returns:
            bool: True if financial content detected
        """
        text_lower = text.lower()
        financial_score = sum(1 for keyword in self.financial_keywords 
                            if keyword in text_lower)
        
        # Consider it financial if at least 2 financial keywords found
        return financial_score >= 2
    
    def extract_headlines_and_sentences(self, text):
        """
        Extract headlines and key sentences for FinBERT processing
        
        Args:
            text (str): Full article text
        
        Returns:
            list: List of text segments suitable for FinBERT
        """
        segments = []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Find title/headline (usually first sentence or line)
        if sentences:
            title = sentences[0]
            if len(title.split()) <= 20:  # Likely a title if short
                segments.append(title)
        
        # Extract sentences with financial keywords
        for sentence in sentences:
            if (self.is_financial_content(sentence) and 
                len(sentence.split()) >= 5 and  # Minimum length
                len(sentence.split()) <= 50):   # Maximum length for meaningful sentences
                segments.append(sentence)
        
        return segments
    
    def tokenize_for_finbert(self, text):
        """
        Tokenize text according to FinBERT requirements
        
        Args:
            text (str): Text to tokenize
        
        Returns:
            dict: Tokenized inputs for FinBERT
        """
        # Tokenize with truncation and padding
        encoded = self.finbert_tokenizer(
            text,
            max_length=self.max_sequence_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'original_text': text
        }
    
    def process_single_pdf(self, pdf_path):
        """
        Process a single PDF file
        
        Args:
            pdf_path (Path): Path to PDF file
        
        Returns:
            list: List of processed articles/segments
        """
        self.logger.info(f"Processing: {pdf_path.name}")
        
        # Try different extraction methods if one fails
        text = ""
        for method in ['pdfplumber', 'pymupdf', 'pypdf2']:
            text = self.extract_text_from_pdf(pdf_path, method)
            if text.strip():
                break
        
        if not text.strip():
            self.logger.warning(f"Could not extract text from {pdf_path.name}")
            return []
        
        # Check if content is financial
        if not self.is_financial_content(text):
            self.logger.info(f"Skipping {pdf_path.name} - not financial content")
            return []
        
        # Extract date
        date = self.extract_date_from_filename_or_content(pdf_path.name, text)
        if not date:
            self.logger.warning(f"Could not extract date from {pdf_path.name}")
            date = datetime.now()  # Use current date as fallback
        
        # Extract headlines and key sentences
        segments = self.extract_headlines_and_sentences(text)
        
        processed_articles = []
        
        for i, segment in enumerate(segments):
            # Preprocess text
            cleaned_text = self.preprocess_text(segment)
            
            if len(cleaned_text.split()) < 3:  # Skip very short segments
                continue
            
            # Tokenize for FinBERT
            tokenized = self.tokenize_for_finbert(cleaned_text)
            
            article_data = {
                'filename': pdf_path.name,
                'date': date.strftime('%Y-%m-%d'),
                'segment_id': i,
                'original_text': segment,
                'cleaned_text': cleaned_text,
                'token_count': len(cleaned_text.split()),
                'is_title': i == 0,  # First segment is likely title
                'finbert_ready': True
            }
            
            processed_articles.append(article_data)
        
        return processed_articles
    
    def process_all_pdfs(self):
        """
        Process all PDFs in the directory
        
        Returns:
            pd.DataFrame: DataFrame with processed articles
        """
        all_articles = []
        
        # Find all PDF files
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.pdf_directory}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        for pdf_path in pdf_files:
            articles = self.process_single_pdf(pdf_path)
            all_articles.extend(articles)
        
        # Create DataFrame
        df = pd.DataFrame(all_articles)
        
        if not df.empty:
            # Sort by date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remove duplicates based on cleaned text
            df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
            
            self.logger.info(f"Processed {len(df)} unique articles from {len(pdf_files)} PDFs")
        
        return df
    
    def save_for_finbert(self, df, output_format='csv'):
        """
        Save processed data in format suitable for FinBERT
        
        Args:
            df (pd.DataFrame): Processed articles DataFrame
            output_format (str): Output format ('csv', 'json')
        """
        if df.empty:
            self.logger.warning("No data to save")
            return
        
        if output_format == 'csv':
            # Save main dataset
            df.to_csv(self.output_file, index=False)
            
            # Save FinBERT-ready format (simplified)
            finbert_df = df[['date', 'cleaned_text']].copy()
            finbert_df.columns = ['date', 'text']
            finbert_output = self.output_file.replace('.csv', '_finbert_ready.csv')
            finbert_df.to_csv(finbert_output, index=False)
            
            self.logger.info(f"Saved {len(df)} articles to {self.output_file}")
            self.logger.info(f"FinBERT-ready data saved to {finbert_output}")
            
        elif output_format == 'json':
            json_output = self.output_file.replace('.csv', '.json')
            df.to_json(json_output, orient='records', date_format='iso')
            self.logger.info(f"Saved data to {json_output}")
    
    def create_sample_usage_code(self):
        """
        Generate sample code for using the extracted data with FinBERT
        
        Returns:
            str: Sample usage code
        """
        sample_code = '''
# Sample code for using extracted data with FinBERT

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Load your processed data
df = pd.read_csv('financial_news_data_finbert_ready.csv')

# Initialize FinBERT
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

def get_sentiment_scores(texts):
    """Get sentiment scores for a batch of texts"""
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, 
                      max_length=128, return_tensors='pt')
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities
    probabilities = softmax(outputs.logits, dim=-1)
    
    # Convert to sentiment scores
    # FinBERT outputs: [positive, negative, neutral]
    sentiment_scores = []
    for prob in probabilities:
        pos, neg, neu = prob.tolist()
        sentiment_scores.append({
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'compound': pos - neg  # Simple compound score
        })
    
    return sentiment_scores

# Process your data
batch_size = 16
all_sentiments = []

for i in range(0, len(df), batch_size):
    batch_texts = df['text'].iloc[i:i+batch_size].tolist()
    batch_sentiments = get_sentiment_scores(batch_texts)
    all_sentiments.extend(batch_sentiments)

# Add sentiment scores to dataframe
sentiment_df = pd.DataFrame(all_sentiments)
result_df = pd.concat([df, sentiment_df], axis=1)

# Save results
result_df.to_csv('financial_news_with_sentiment.csv', index=False)

print(f"Processed {len(result_df)} articles with sentiment scores")
print(result_df.head())
'''
        return sample_code

# Usage example
if __name__ == "__main__":
    # Example usage
    pdf_directory = "/path/to/your/pdf/directory"  # Change this path
    
    # Initialize extractor
    extractor = FinancialPDFExtractor(pdf_directory)
    
    # Process all PDFs
    df = extractor.process_all_pdfs()
    
    # Save results
    extractor.save_for_finbert(df)
    
    # Print sample usage code
    print("\\n" + "="*50)
    print("SAMPLE FINBERT USAGE CODE:")
    print("="*50)
    print(extractor.create_sample_usage_code())