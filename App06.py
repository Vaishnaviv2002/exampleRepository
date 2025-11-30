#!/usr/bin/env python3
"""
App06.py - scrape README.md from the GitHub page and save sections.

Usage:
    python3.12 App06.py

Creates files:
 - README_downloaded.md
 - README_sections.csv
 - README_sections.json
"""
import sys
import requests
from bs4 import BeautifulSoup
import csv
import json
import os

URL = "https://github.com/MarcusGrum/AIBAS/blob/main/README.md"
RAW_URL = "https://raw.githubusercontent.com/MarcusGrum/AIBAS/main/README.md"

OUT_MD = "README_downloaded.md"
OUT_CSV = "README_sections.csv"
OUT_JSON = "README_sections.json"

def fetch_raw(read_raw_url):
    try:
        r = requests.get(read_raw_url, timeout=15)
        if r.status_code == 200 and r.text.strip():
            print("Fetched raw README from:", read_raw_url)
            return r.text
        print("Raw fetch returned status", r.status_code)
    except Exception as e:
        print("Raw fetch failed:", e)
    return None

def fetch_rendered(url):
    """Fetch the GitHub HTML page and extract the rendered markdown section with BeautifulSoup."""
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print("HTML fetch returned status", r.status_code)
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # GitHub renders README under a div with class 'markdown-body' or <article class="markdown-body entry-content container-lg">
        # We'll search for common possibilities.
        candidate = None
        for cls in ("markdown-body", "entry-content", "repository-content"):
            el = soup.find("div", class_=cls)
            if el and el.get_text(strip=True):
                candidate = el
                break
        if candidate is None:
            # try the article element
            el = soup.find("article")
            if el and el.get_text(strip=True):
                candidate = el
        if candidate is None:
            print("Could not find rendered README in HTML.")
            return None
        # convert rendered HTML back to markdown-like plain text:
        text = candidate.get_text("\n")
        print("Extracted rendered README text from HTML page.")
        return text
    except Exception as e:
        print("Rendered fetch failed:", e)
        return None

def split_into_sections(text):
    """
    Split README text into sections by markdown headings.
    We treat lines starting with one or more '#' as a section header.
    Each returned section is (header, content_text).
    """
    lines = text.splitlines()
    sections = []
    cur_header = "Preamble"
    cur_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # new header
            # store previous
            if cur_lines:
                sections.append((cur_header, "\n".join(cur_lines).strip()))
            # header title: remove leading '#' and surrounding whitespace
            header_title = stripped.lstrip("#").strip()
            cur_header = header_title if header_title else "Untitled"
            cur_lines = []
        else:
            cur_lines.append(line)
    # add last
    if cur_lines or cur_header:
        sections.append((cur_header, "\n".join(cur_lines).strip()))
    return sections

def save_outputs(raw_text, sections):
    # save raw md
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(raw_text)
    # save csv
    with open(OUT_CSV, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["section_title", "section_text"])
        for title, content in sections:
            writer.writerow([title, content])
    # save json
    obj = [{"section_title": t, "section_text": c} for t, c in sections]
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("Saved files:", OUT_MD, OUT_CSV, OUT_JSON)

def main():
    print("Starting App06 README scraper.")
    # 1) try raw
    text = fetch_raw(RAW_URL)
    if not text:
        print("Raw not available. Trying rendered HTML extraction...")
        text = fetch_rendered(URL)
    if not text:
        print("Failed to fetch README from both raw and rendered sources. Exiting.")
        return 1
    # 2) normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 3) split into sections
    sections = split_into_sections(text)
    print("Parsed", len(sections), "sections. Top sections:")
    for i, (title, content) in enumerate(sections[:6]):
        print(f"  {i+1}. {title} (len {len(content)} chars)")
    # 4) save outputs
    save_outputs(text, sections)
    return 0

if __name__ == "__main__":
    exit(main())
