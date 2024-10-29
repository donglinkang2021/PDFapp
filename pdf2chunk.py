import csv
from PyPDF2 import PdfReader
from tqdm import tqdm
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def split_text(text: str, chunk_size: int) -> List[str]:
    # 按照 chunk_size 切分文本
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def save_to_csv(chunks:List[str], csv_path: str) -> None:
    pbar = tqdm(total=len(chunks), desc="Writing to CSV",
                unit="chunk", dynamic_ncols=True, leave=False)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'chunk'])  # 写入表头
        for id, chunk in enumerate(chunks, start=1):
            writer.writerow([id, chunk])
            pbar.update(1)
    pbar.close()

def pdf2chunk(pdf_path, chunk_size, csv_path):
    # 提取文本
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    # 切分文本
    print(f"Splitting text into chunks of size {chunk_size}...")
    chunks = split_text(text, chunk_size)

    # 保存到 CSV
    print(f"Saving chunks to {csv_path}...")
    save_to_csv(chunks, csv_path)
    print("Done!")

if __name__ == '__main__':
    pdf_path = 'PDFapp\data\Radford 等 - Language Models are Unsupervised Multitask Learners.pdf'  # 替换为你的PDF文件路径
    chunk_size = 512  # 设置每个chunk的大小
    csv_path = 'PDFapp\data\chunk.csv'  # 输出的CSV文件路径

    pdf2chunk(pdf_path, chunk_size, csv_path)

# python PDFapp\pdf2chunk.py