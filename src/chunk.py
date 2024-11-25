from PyPDF2 import PdfReader
from typing import List

class Chunker:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, pdf_path: str) -> List[str]:
        # extract_text_from_pdf
        text = ''
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

if __name__ == '__main__':
    chunker = Chunker(chunk_size=512)
    chunks = chunker.chunk('data/temp.pdf')
    print(len(chunks))

# python -m src.chunker