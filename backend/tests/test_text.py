from app.ingest.prepare import display_author, era_for_year, short_title
from app.ingest.text import chunk_text, is_heading, split_chapters, strip_gutenberg

RAW = """The Project Gutenberg eBook of Test
Some license text.
*** START OF THE PROJECT GUTENBERG EBOOK TEST ***
Body text.
*** END OF THE PROJECT GUTENBERG EBOOK TEST ***
License footer."""


def make_book(n_chapters: int, heading=lambda i: f"CHAPTER {i}") -> str:
    para = "This is a sentence about the sea. " * 30
    parts = ["Contents", *[heading(i) for i in range(1, n_chapters + 1)]]
    for i in range(1, n_chapters + 1):
        parts += [heading(i), para, para]
    return "\n\n".join(parts)


def test_strip_gutenberg_removes_header_and_footer():
    assert strip_gutenberg(RAW) == "Body text."


def test_headings_en_and_de():
    for line in ["CHAPTER XII.", "Chapter 3", "BOOK ONE", "Erstes Kapitel", "3. Kapitel", "XIV",
                 "I. A SCANDAL IN BOHEMIA", "STAVE ONE"]:
        assert is_heading(line), line
    for line in ["It was the best of times.", "Mr. Darcy", "I am here."]:
        assert not is_heading(line), line


def test_split_chapters_skips_table_of_contents():
    chapters = split_chapters(make_book(5))
    titles = [c.title for c in chapters if c.title != "Front matter"]
    assert titles == [f"CHAPTER {i}" for i in range(1, 6)]
    assert all("sea" in c.text for c in chapters if c.title != "Front matter")


def test_split_chapters_german():
    chapters = split_chapters(make_book(4, lambda i: f"{i}. Kapitel"))
    assert [c.title for c in chapters if c.title != "Front matter"] == [f"{i}. Kapitel" for i in range(1, 5)]


def test_split_chapters_fallback_sections():
    text = "\n\n".join(["Plain paragraph without any structure. " * 20] * 60)
    chapters = split_chapters(text)
    assert len(chapters) >= 2
    assert chapters[0].title == "Section 1"


def test_chunk_text_sizes_and_overlap():
    text = "\n\n".join(f"Sentence {i} is here. Another one follows {i}." for i in range(200))
    chunks = chunk_text(text, target_chars=300, max_chars=500)
    assert len(chunks) > 5
    assert all(len(c) <= 1000 for c in chunks)
    # overlap: the last sentence of a chunk starts the next one
    assert chunks[1].split("\n")[0] in chunks[0]


def test_chunk_text_joins_hard_wrapped_lines():
    chunks = chunk_text("A line that\nwraps here and _italic_ text, long enough.")
    assert chunks == ["A line that wraps here and italic text, long enough."]


def test_metadata_helpers():
    assert display_author("Carroll, Lewis") == "Lewis Carroll"
    assert display_author("Wells, H. G. (Herbert George)") == "H. G. Wells"
    assert display_author("Sacher-Masoch, Leopold, Ritter von") == "Leopold von Sacher-Masoch"
    assert display_author("Marcus Aurelius, Emperor of Rome") == "Marcus Aurelius"
    assert display_author("Plato") == "Plato"
    assert short_title("Frankenstein; or, the modern prometheus") == "Frankenstein"
    assert short_title("Peter Pan : $b [Peter and Wendy]") == "Peter Pan"
    assert era_for_year(-700) == "Antiquity & Middle Ages"
    assert era_for_year(1925) == "1900-1930"


def test_windows_line_endings_keep_paragraphs():
    raw = "*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\r\nline one\r\r\nline two\r\r\n\r\r\nnext para\r\r\n"
    text = strip_gutenberg(raw)
    assert text == "line one\nline two\n\nnext para"
    assert chunk_text(text + " that is long enough to be kept as a passage.") == [
        "line one line two\nnext para that is long enough to be kept as a passage."]
