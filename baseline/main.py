from read_pdf import ocr_pdf
from baseline import review_paper

def main():
    paper_text = ocr_pdf("paper.pdf")

    print("Extracted text preview:")
    print(paper_text[:1000])

    print("\nGenerating review...\n")

    review = review_paper(paper_text)

    print(review)

if __name__ == "__main__":
    main()