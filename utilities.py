# Content: Class definition of interactive chatbot system 
# Author: Shuai Guo
# Email: shuai.guo@ch.abb.com
# Date: August, 2023

from langchain.document_loaders import PyPDFLoader
import re

def extract_TOC(issue_name):
    """Extracting the table of content page of the ABB review journal.

    Args:
    --------------
    issue_name: name of the journal issue.

    Outputs:
    --------
    TOC: extratced table-of-content page of the PDF
    """

    # Table of content page
    loader = PyPDFLoader("./papers/"+issue_name)
    raw_documents = loader.load()
    TOC_page = 3
    TOC = raw_documents[TOC_page-1].page_content
    TOC = TOC.replace('�', '')

    return TOC



def extract_metadata(TOC):
    """Extracting the article metadata given the table of content.

    Args:
    --------------
    TOC: table of content page (PDF).

    Outputs:
    --------
    articles: dict, with article title & subtitle, starting page number, and article length.
    TOC_page_number: page number of the table-of-content page in the original PDF.
    """

    # Split the text into lines
    lines = TOC.split('\n')
    
    # Extract TOC page number
    TOC_page_number = int(lines[0].split('|')[-1][4:])

    # Prepare to iterate through the lines
    i = 0
    articles = []
    while i < len(lines):
        line = lines[i].strip()

        # Check if the line starts with a number followed by the article title
        # Ensure the following characters are not digits
        match = re.search(r'(\d+)\s+([^\d]+)$', line)
        if match:
            # Extract the starting page number and title
            start_page, title = match.groups()
            subtitle = ""
            i += 1

            # Extract the subtitle (lines until delimiter '—' or next "page number + title" combination)
            while i < len(lines) and not re.match(r'\d+\s+[^\d]+$', lines[i]):
                if '—' in lines[i]:  # Check if delimiter is present in the line
                    # Split the line at the delimiter and keep only the left portion
                    left_portion = lines[i].split('—')[0].strip()
                    subtitle += ' ' + left_portion
                    i += 1
                    break
                subtitle += ' ' + lines[i].strip()
                i += 1

            subtitle = subtitle.strip()
            
            # remove invalid articles with too long subtitles
            if len(subtitle)<150:
                articles.append({"start_page": int(start_page), "title": title, "subtitle": subtitle})
        else:
            i += 1

    return articles, TOC_page_number



def extract_articles(issue_name):
    """Extracting the article metadata given an issue of ABB review journal.

    Args:
    --------------
    issue_name: name of the journal issue.

    Outputs:
    --------
    articles: dict, with article title & subtitle, starting page number, and article length.
              Articles are sorted based on their starting page number.
    """

    # Table of content page
    TOC = extract_TOC(issue_name)

    # Extract final articles with titles, subtitles, and categories
    articles, TOC_page_number = extract_metadata(TOC)

    # Rearrange artical order
    articles.sort(key=lambda x: int(x['start_page']))

    # Infer ending page
    end_pages = []
    for i in range(len(articles) - 1):
        end_pages.append(articles[i+1]['start_page'] - 1)
        
    # Add placeholder for the last article since we don't have its end page
    end_pages.append("Unknown")

    # Attach end page to articles
    for article, end_page in zip(articles, end_pages):
        if end_page != "Unknown":
            article['length'] = end_page - article['start_page'] + 1
        elif article['title']=='Editorial':
            article['length'] = 1
        else:
            article['length'] = "Unknown"
        
    # Retain only valid articles
    subscribe_index = next((i for i, article in enumerate(articles) if article['title'] == 'Subscribe'), None)
    if subscribe_index is not None:
        articles = articles[:subscribe_index]
        
    # Page number in relative (TOC is always the 3rd page in an issue)
    for article in articles:
        article['start_page'] = article['start_page'] - TOC_page_number + 3


    return articles



def remove_reference(documents):
    """Remove the reference section of the extracted article.
    Here, we made the assumption that the reference appears on the last page.

    Args:
    --------------
    documents: the extracted article.

    Outputs:
    --------
    documents: the extracted article with reference page removed.
    """
    
    # Last page
    text = documents[-1].page_content

    # Split the text by the "References" section
    before_references, references_and_after = text.split("References", 1)

    # Find the end of the "References" section by locating the first "—" after "References"
    after_references = references_and_after.split("\n—\n", 1)[1]

    # Combine the before and after parts
    cleaned_text = before_references + after_references

    # Update the raw documents
    documents[-1].page_content = cleaned_text

    return documents