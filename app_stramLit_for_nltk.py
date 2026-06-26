import streamlit as st
import nltk

# Download all required NLTK data on startup (needed for Streamlit Cloud)
for _pkg in [
    "punkt", "punkt_tab", "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
    "maxent_ne_chunker_tab", "words", "stopwords", "vader_lexicon",
    "wordnet", "omw-1.4", "brown", "gutenberg", "movie_reviews",
    "names", "universal_tagset", "tagsets", "conll2000",
    "treebank", "inaugural", "webtext",
]:
    nltk.download(_pkg, quiet=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import io
import hashlib
import contextlib
import traceback

st.set_page_config(page_title="NLTK Explorer", layout="wide", page_icon="📚")

# ── Runnable code helper ─────────────────────────────────────────────────────
def _execute(code: str):
    """Execute code, capture stdout and matplotlib figures, display results."""
    buf = io.StringIO()
    exec_globals = {
        "__builtins__": __builtins__,
        "nltk": nltk,
        "pd": pd,
        "plt": plt,
        "io": io,
        "random": __import__("random"),
        "os": __import__("os"),
    }
    plt.close("all")
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<string>", "exec"), exec_globals)
        output = buf.getvalue()
        figs = [plt.figure(n) for n in plt.get_fignums()]
        if output.strip():
            st.success("**Output:**")
            st.code(output.strip(), language=None)
        for fig in figs:
            st.pyplot(fig)
            plt.close(fig)
        if not output.strip() and not figs:
            st.info("✅ Ran successfully (no printed output)")
    except Exception:
        st.error(f"**Error:**\n```\n{traceback.format_exc()}\n```")


def runnable_code(code: str, key: str = None):
    """
    Beautiful st.code() display by default.
    ▶ Run  — executes as-is.
    ✏️ Edit — switches to editable text_area.
    ↩ Reset — returns to beautiful display.
    """
    _key = key or hashlib.md5(code.encode()).hexdigest()[:12]
    edit_state_key = f"_edit_{_key}"
    code_val_key   = f"_val_{_key}"

    if edit_state_key not in st.session_state:
        st.session_state[edit_state_key] = False
    if code_val_key not in st.session_state:
        st.session_state[code_val_key] = code.strip()

    in_edit_mode = st.session_state[edit_state_key]

    if in_edit_mode:
        # ── Edit mode: text_area ─────────────────────────────────────────
        lines  = st.session_state[code_val_key].count("\n") + 1
        height = max(120, min(lines * 21 + 24, 520))
        edited = st.text_area(
            "✏️ Edit then click Run or Done:",
            value=st.session_state[code_val_key],
            height=height,
            key=f"_ta_{_key}",
            label_visibility="visible",
        )
        col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.4, 4])
        run_btn      = col1.button("▶ Run",        key=f"_run_{_key}")
        done_btn     = col2.button("← Done",       key=f"_rst_{_key}")
        original_btn = col3.button("🔄 Original",  key=f"_org_{_key}")

        if done_btn:
            # Save edits and go back to beautiful display
            st.session_state[code_val_key]   = edited
            st.session_state[edit_state_key] = False
            st.rerun()

        if original_btn:
            # Discard edits, restore original, go back to display
            st.session_state[code_val_key]   = code.strip()
            st.session_state[edit_state_key] = False
            st.rerun()

        if run_btn:
            st.session_state[code_val_key] = edited
            _execute(edited)

    else:
        # ── Display mode: beautiful st.code() ───────────────────────────
        st.code(st.session_state[code_val_key], language="python")
        col1, col2, col3 = st.columns([1.1, 1.1, 6])
        run_btn  = col1.button("▶ Run",   key=f"_run_{_key}")
        edit_btn = col2.button("✏️ Edit", key=f"_edt_{_key}")

        if edit_btn:
            st.session_state[edit_state_key] = True
            st.rerun()

        if run_btn:
            _execute(st.session_state[code_val_key])

# ── Sidebar navigation ──────────────────────────────────────────────────────
PAGES = [
    "1. Tokenization",
    "2. Stemming & Lemmatization",
    "3. POS Tagging",
    "4. Named Entity Recognition & Chunking",
    "5. Sentiment Analysis (VADER)",
    "6. Frequency Distribution",
    "7. Collocations",
    "8. N-grams & Text Utilities",
    "9. WordNet",
    "10. Language Models",
    "11. Text Metrics & Distance",
    "12. Translation Metrics",
    "13. Grammar & Parsing",
    "14. Classification (Naive Bayes)",
    "15. Concordance & Text Analysis",
    "16. Probability Distributions",
    "17. ❌ Not Available in Streamlit",
]

st.sidebar.title("📚 NLTK Explorer")
st.sidebar.markdown("Navigate the full NLTK API")
page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")

DEFAULT_TEXT = "NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources."

# ════════════════════════════════════════════════════════════════════════════
# 1. TOKENIZATION
# ════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("Tokenization")
    st.markdown("`nltk.tokenize` — split text into words, sentences, or custom units")

    text = st.text_area("Input text", DEFAULT_TEXT, height=120)

    tabs = st.tabs([
        "word_tokenize", "sent_tokenize", "TweetTokenizer",
        "RegexpTokenizer", "WhitespaceTokenizer", "WordPunctTokenizer",
        "BlanklineTokenizer", "MWETokenizer", "SyllableTokenizer",
        "ToktokTokenizer", "NLTKWordTokenizer",
    ])

    with tabs[0]:
        st.subheader("word_tokenize")
        tokens = nltk.word_tokenize(text)
        st.write(tokens)
        runnable_code("""import nltk
text = "NLTK is a leading platform for NLP. It provides easy-to-use interfaces."
tokens = nltk.word_tokenize(text)
print(tokens)
print("Total tokens:", len(tokens))""")

    with tabs[1]:
        st.subheader("sent_tokenize")
        sents = nltk.sent_tokenize(text)
        for i, s in enumerate(sents):
            st.write(f"**Sentence {i+1}:** {s}")
        runnable_code("""import nltk
text = (
    "Natural Language Processing is a subfield of linguistics and AI. "
    "It focuses on the interaction between computers and human language. "
    "NLTK is one of the most popular libraries for NLP in Python. "
    "It provides tools for tokenization, parsing, classification, and more."
)
sentences = nltk.sent_tokenize(text)
print(f"Found {len(sentences)} sentences:\\n")
for i, s in enumerate(sentences, 1):
    print(f"Sentence {i}: {s}")""")

    with tabs[2]:
        st.subheader("TweetTokenizer")
        col1, col2 = st.columns(2)
        strip_handles = col1.checkbox("strip_handles", False)
        reduce_len = col2.checkbox("reduce_len", False)
        tweet = st.text_input("Tweet text", "@nltk is AMAZINGGG!!! 😊 #NLP #Python")
        from nltk.tokenize import TweetTokenizer
        tt = TweetTokenizer(strip_handles=strip_handles, reduce_len=reduce_len)
        st.write(tt.tokenize(tweet))
        runnable_code("""from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer(strip_handles=True, reduce_len=True)
tweets = [
    "@nltk is AMAZINGGG!!! 😊 #NLP #Python",
    "I looooove this library!!! @user can't believe it 😂😂",
    "#MachineLearning and #AI are revolutionizing the world!!!",
]
for tweet in tweets:
    tokens = tt.tokenize(tweet)
    print(f"Tweet : {tweet}")
    print(f"Tokens: {tokens}\\n")""")

    with tabs[3]:
        st.subheader("RegexpTokenizer")
        pattern = st.text_input("Regex pattern", r"\w+")
        from nltk.tokenize import RegexpTokenizer
        try:
            rt = RegexpTokenizer(pattern)
            st.write(rt.tokenize(text))
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.tokenize import RegexpTokenizer
text = "Hello, world! This is NLTK 3.9. It costs $0.00 and works 100% great!"

# Only words (no punctuation or numbers)
tok_words = RegexpTokenizer(r'[a-zA-Z]+')
print("Words only:        ", tok_words.tokenize(text))

# Words and numbers
tok_wn = RegexpTokenizer(r'\\w+')
print("Words + numbers:   ", tok_wn.tokenize(text))

# Only numbers
tok_nums = RegexpTokenizer(r'\\d+\\.?\\d*')
print("Numbers only:      ", tok_nums.tokenize(text))""")

    with tabs[4]:
        st.subheader("WhitespaceTokenizer")
        from nltk.tokenize import WhitespaceTokenizer
        st.write(WhitespaceTokenizer().tokenize(text))
        runnable_code("""from nltk.tokenize import WhitespaceTokenizer
texts = [
    "Hello world  foo",
    "split   on   any   whitespace",
    "tabs\\there\\ttoo",
    "newlines\\nalso\\ncounted",
]
tok = WhitespaceTokenizer()
for t in texts:
    print(f"Input : {repr(t)}")
    print(f"Output: {tok.tokenize(t)}\\n")""")

    with tabs[5]:
        st.subheader("WordPunctTokenizer")
        from nltk.tokenize import WordPunctTokenizer
        st.write(WordPunctTokenizer().tokenize(text))
        runnable_code("""from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
sentences = [
    "Hello, world! It's NLTK.",
    "Dr. Smith went to Washington D.C. on Jan. 5th.",
    "She said: 'I can't believe it!' and left.",
]
for s in sentences:
    print(f"Input : {s}")
    print(f"Output: {tok.tokenize(s)}\\n")""")

    with tabs[6]:
        st.subheader("BlanklineTokenizer")
        multi = st.text_area("Text with blank lines", "Paragraph one.\n\nParagraph two.\n\nParagraph three.", height=100)
        from nltk.tokenize import BlanklineTokenizer
        st.write(BlanklineTokenizer().tokenize(multi))
        runnable_code("""from nltk.tokenize import BlanklineTokenizer
text = "Para one.\\n\\nPara two.\\n\\nPara three."
print(BlanklineTokenizer().tokenize(text))""")

    with tabs[7]:
        st.subheader("MWETokenizer — Multi-Word Expression Tokenizer")
        from nltk.tokenize import MWETokenizer
        mwes = [("New", "York"), ("natural", "language", "processing")]
        mwet = MWETokenizer(mwes)
        sample = "New York is great for natural language processing"
        st.write("Input:", sample)
        st.write("Output:", mwet.tokenize(sample.split()))
        runnable_code("""from nltk.tokenize import MWETokenizer
tokenizer = MWETokenizer([('New', 'York'), ('natural', 'language', 'processing')])
tokens = "New York is great for natural language processing".split()
print(tokenizer.tokenize(tokens))""")

    with tabs[8]:
        st.subheader("SyllableTokenizer (Sonority Sequencing)")
        from nltk.tokenize import SyllableTokenizer
        sst = SyllableTokenizer()
        word = st.text_input("Word to syllabify", "basketball")
        st.write(sst.tokenize(word))
        runnable_code("""from nltk.tokenize import SyllableTokenizer
sst = SyllableTokenizer()
print(sst.tokenize("basketball"))  # ['bas', 'ket', 'ball']""")

    with tabs[9]:
        st.subheader("ToktokTokenizer")
        from nltk.tokenize import ToktokTokenizer
        st.write(ToktokTokenizer().tokenize(text))
        runnable_code("""from nltk.tokenize import ToktokTokenizer
print(ToktokTokenizer().tokenize("Hello, world! It's NLTK."))""")

    with tabs[10]:
        st.subheader("NLTKWordTokenizer")
        from nltk.tokenize import NLTKWordTokenizer
        st.write(list(NLTKWordTokenizer().tokenize(text)))
        runnable_code("""from nltk.tokenize import NLTKWordTokenizer
print(list(NLTKWordTokenizer().tokenize("Hello, world! It's great.")))""")


# ════════════════════════════════════════════════════════════════════════════
# 2. STEMMING & LEMMATIZATION
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("Stemming & Lemmatization")
    st.markdown("`nltk.stem` — reduce words to their root or base form")

    words_input = st.text_input("Enter words (comma-separated)", "running, flies, studies, better, caring, happiness")
    words = [w.strip() for w in words_input.split(",") if w.strip()]

    tabs = st.tabs([
        "PorterStemmer", "LancasterStemmer", "SnowballStemmer",
        "RegexpStemmer", "WordNetLemmatizer", "ARLSTem (Arabic)",
    ])

    with tabs[0]:
        st.subheader("PorterStemmer")
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        results = {w: ps.stem(w) for w in words}
        st.table(pd.DataFrame(results.items(), columns=["Word", "Stem"]))
        runnable_code("""from nltk.stem import PorterStemmer
ps = PorterStemmer()
words = [
    "running", "runner", "runs", "ran",
    "flies", "flying", "flied",
    "studies", "studying", "studied",
    "happiness", "happy", "happily",
    "caring", "cared", "carefully",
]
print(f"{'Word':<15} {'Stem'}")
print("-" * 25)
for w in words:
    print(f"{w:<15} {ps.stem(w)}")""")

    with tabs[1]:
        st.subheader("LancasterStemmer")
        from nltk.stem import LancasterStemmer
        ls = LancasterStemmer()
        results = {w: ls.stem(w) for w in words}
        st.table(pd.DataFrame(results.items(), columns=["Word", "Stem"]))
        runnable_code("""from nltk.stem import LancasterStemmer, PorterStemmer
ls = LancasterStemmer()
ps = PorterStemmer()
words = [
    "running", "runner", "happiness",
    "studies", "generously", "eating",
    "classification", "presumably", "multiply",
]
print(f"{'Word':<20} {'Lancaster':<15} {'Porter'}")
print("-" * 50)
for w in words:
    print(f"{w:<20} {ls.stem(w):<15} {ps.stem(w)}")""")

    with tabs[2]:
        st.subheader("SnowballStemmer")
        from nltk.stem import SnowballStemmer
        lang = st.selectbox("Language", sorted(SnowballStemmer.languages))
        try:
            sb = SnowballStemmer(lang)
            results = {w: sb.stem(w) for w in words}
            st.table(pd.DataFrame(results.items(), columns=["Word", "Stem"]))
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.stem import SnowballStemmer
words = ["running", "happiness", "generously", "studies", "beautiful", "quickly"]

# Compare across 5 languages (same concept words)
langs_words = {
    "english":    ["running", "happiness", "generously", "studies"],
    "french":     ["courant", "bonheur", "généreusement", "études"],
    "german":     ["laufen", "glück", "großzügig", "studien"],
    "spanish":    ["corriendo", "felicidad", "generosamente", "estudios"],
    "portuguese": ["correndo", "felicidade", "generosamente", "estudos"],
}
for lang, ws in langs_words.items():
    sb = SnowballStemmer(lang)
    stems = [sb.stem(w) for w in ws]
    print(f"{lang:<12}: {ws} -> {stems}")""")

    with tabs[3]:
        st.subheader("RegexpStemmer")
        from nltk.stem import RegexpStemmer
        suffix = st.text_input("Suffix pattern to strip", "ing$|s$|ed$")
        try:
            rs = RegexpStemmer(suffix)
            results = {w: rs.stem(w) for w in words}
            st.table(pd.DataFrame(results.items(), columns=["Word", "Stem"]))
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.stem import RegexpStemmer
stemmer = RegexpStemmer('ing$|s$|ed$', min=4)
print(stemmer.stem("running"))  # runn
print(stemmer.stem("flies"))    # flie""")

    with tabs[4]:
        st.subheader("WordNetLemmatizer")
        from nltk.stem import WordNetLemmatizer
        wl = WordNetLemmatizer()
        pos_map = {"noun (n)": "n", "verb (v)": "v", "adjective (a)": "a", "adverb (r)": "r"}
        pos_label = st.selectbox("POS", list(pos_map.keys()))
        pos = pos_map[pos_label]
        results = {w: wl.lemmatize(w, pos=pos) for w in words}
        st.table(pd.DataFrame(results.items(), columns=["Word", "Lemma"]))
        runnable_code("""from nltk.stem import WordNetLemmatizer
wl = WordNetLemmatizer()

# Without POS: defaults to noun
print("=== Default (noun) ===")
for w in ["running", "better", "flies", "studies", "went"]:
    print(f"  {w:<12} -> {wl.lemmatize(w)}")

# With POS specified
print("\\n=== With POS ===")
examples = [
    ("running",  'v'),  # verb  -> run
    ("better",   'a'),  # adj   -> good
    ("flies",    'n'),  # noun  -> fly
    ("flies",    'v'),  # verb  -> fly
    ("went",     'v'),  # verb  -> go
    ("studies",  'v'),  # verb  -> study
    ("studies",  'n'),  # noun  -> study
    ("happily",  'r'),  # adverb-> happily
]
for word, pos in examples:
    print(f"  {word:<12} (pos={pos}) -> {wl.lemmatize(word, pos=pos)}")""")

    with tabs[5]:
        st.subheader("ARLSTem — Arabic Light Stemmer")
        st.info("ARLSTem and ARLSTem2 are designed for Arabic text.")
        from nltk.stem import ARLSTem, ARLSTem2
        arabic_word = st.text_input("Arabic word", "يكتبون")
        a1 = ARLSTem()
        a2 = ARLSTem2()
        st.write(f"**ARLSTem stem:** `{a1.stem(arabic_word)}`")
        st.write(f"**ARLSTem2 stem:** `{a2.stem(arabic_word)}`")
        runnable_code("""from nltk.stem import ARLSTem, ARLSTem2
a1 = ARLSTem()
a2 = ARLSTem2()
word = "يكتبون"
print(a1.stem(word))
print(a2.stem(word))""")


# ════════════════════════════════════════════════════════════════════════════
# 3. POS TAGGING
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("POS Tagging")
    st.markdown("`nltk.tag` — assign Part-of-Speech tags to tokens")

    text = st.text_area("Input text", DEFAULT_TEXT, height=100)
    tokens = nltk.word_tokenize(text)

    tabs = st.tabs([
        "pos_tag (default)", "Universal Tagset", "PerceptronTagger",
        "UnigramTagger", "BigramTagger", "DefaultTagger",
        "RegexpTagger", "AffixTagger", "map_tag",
    ])

    with tabs[0]:
        st.subheader("nltk.pos_tag — Penn Treebank tagset")
        tagged = nltk.pos_tag(tokens)
        st.table(pd.DataFrame(tagged, columns=["Token", "POS Tag"]))
        runnable_code("""import nltk
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She has been studying natural language processing since 2020.",
    "NLTK provides easy-to-use interfaces for over 50 corpora and lexical resources.",
]
for sent in sentences:
    tokens = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(tokens)
    print(f"Sentence: {sent}")
    print(f"Tagged  : {tagged}\\n")""")

    with tabs[1]:
        st.subheader("Universal Tagset (simplified)")
        tagged_u = nltk.pos_tag(tokens, tagset="universal")
        st.table(pd.DataFrame(tagged_u, columns=["Token", "Universal Tag"]))
        st.markdown("""
| Tag | Meaning |
|-----|---------|
| NOUN | Nouns |
| VERB | Verbs |
| ADJ | Adjectives |
| ADV | Adverbs |
| PRON | Pronouns |
| DET | Determiners |
| ADP | Prepositions |
| NUM | Numbers |
| CONJ | Conjunctions |
| PRT | Particles |
| . | Punctuation |
| X | Other |
""")
        runnable_code("""import nltk
sentences = [
    "She quickly runs every morning.",
    "The big brown dog chased the small cat.",
    "Scientists discovered three new planets beyond our solar system.",
]
for sent in sentences:
    tokens = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(tokens, tagset='universal')
    print(f"Sentence : {sent}")
    print(f"Universal: {tagged}\\n")""")

    with tabs[2]:
        st.subheader("PerceptronTagger (same as default pos_tag)")
        from nltk.tag import PerceptronTagger
        tagger = PerceptronTagger()
        tagged = tagger.tag(tokens)
        st.table(pd.DataFrame(tagged, columns=["Token", "Tag"]))
        runnable_code("""from nltk.tag import PerceptronTagger
tagger = PerceptronTagger()
tokens = ["The", "dog", "runs", "fast"]
print(tagger.tag(tokens))""")

    with tabs[3]:
        st.subheader("UnigramTagger — trained on Brown corpus")
        from nltk.tag import UnigramTagger
        from nltk.corpus import brown
        train_sents = brown.tagged_sents(categories="news")[:500]
        ut = UnigramTagger(train_sents)
        tagged = ut.tag(tokens)
        st.table(pd.DataFrame(tagged, columns=["Token", "Tag"]))
        runnable_code("""from nltk.tag import UnigramTagger
from nltk.corpus import brown
train = brown.tagged_sents(categories='news')[:500]
tagger = UnigramTagger(train)
print(tagger.tag(["The", "cat", "sat"]))""")

    with tabs[4]:
        st.subheader("BigramTagger — trained on Brown corpus")
        from nltk.tag import BigramTagger, UnigramTagger
        from nltk.corpus import brown
        train_sents = brown.tagged_sents(categories="news")[:500]
        t0 = UnigramTagger(train_sents)
        bt = BigramTagger(train_sents, backoff=t0)
        tagged = bt.tag(tokens)
        st.table(pd.DataFrame(tagged, columns=["Token", "Tag"]))
        runnable_code("""from nltk.tag import BigramTagger, UnigramTagger
from nltk.corpus import brown
train = brown.tagged_sents(categories='news')[:500]
t0 = UnigramTagger(train)
tagger = BigramTagger(train, backoff=t0)
print(tagger.tag(["The", "big", "dog", "runs"]))""")

    with tabs[5]:
        st.subheader("DefaultTagger — tags everything with one tag")
        from nltk.tag import DefaultTagger
        default_tag = st.text_input("Default tag", "NN")
        dt = DefaultTagger(default_tag)
        st.table(pd.DataFrame(dt.tag(tokens), columns=["Token", "Tag"]))
        runnable_code("""from nltk.tag import DefaultTagger
tagger = DefaultTagger('NN')
print(tagger.tag(["The", "cat", "sat"]))
# [('The', 'NN'), ('cat', 'NN'), ('sat', 'NN')]""")

    with tabs[6]:
        st.subheader("RegexpTagger — pattern-based tagging")
        from nltk.tag import RegexpTagger
        patterns = [
            (r'.*ing$', 'VBG'), (r'.*ed$', 'VBD'), (r'.*es$', 'VBZ'),
            (r'.*ould$', 'MD'), (r'.*\'s$', 'NN$'), (r'.*s$', 'NNS'),
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), (r'.*', 'NN'),
        ]
        rt = RegexpTagger(patterns)
        st.table(pd.DataFrame(rt.tag(tokens), columns=["Token", "Tag"]))
        runnable_code("""from nltk.tag import RegexpTagger
patterns = [
    (r'.*ing$', 'VBG'), (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'), (r'.*s$', 'NNS'), (r'.*', 'NN'),
]
tagger = RegexpTagger(patterns)
print(tagger.tag(["running", "watched", "goes"]))""")

    with tabs[7]:
        st.subheader("AffixTagger — uses word prefixes/suffixes")
        from nltk.tag import AffixTagger
        from nltk.corpus import brown
        train_sents = brown.tagged_sents(categories="news")[:500]
        at = AffixTagger(train_sents)
        tagged = at.tag(tokens)
        st.table(pd.DataFrame(tagged, columns=["Token", "Tag"]))
        runnable_code("""from nltk.tag import AffixTagger
from nltk.corpus import brown
train = brown.tagged_sents(categories='news')[:500]
tagger = AffixTagger(train)
print(tagger.tag(["running", "quickly", "cats"]))""")

    with tabs[8]:
        st.subheader("map_tag — convert between tagsets")
        from nltk.tag import map_tag
        tagged = nltk.pos_tag(tokens)
        mapped = [(tok, map_tag("en-ptb", "universal", tag)) for tok, tag in tagged]
        st.table(pd.DataFrame(mapped, columns=["Token", "Universal Tag"]))
        runnable_code("""from nltk.tag import map_tag
import nltk
tagged = nltk.pos_tag(nltk.word_tokenize("She runs quickly"))
universal = [(w, map_tag('en-ptb', 'universal', t)) for w, t in tagged]
print(universal)""")


# ════════════════════════════════════════════════════════════════════════════
# 4. NAMED ENTITY RECOGNITION & CHUNKING
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("Named Entity Recognition & Chunking")
    st.markdown("`nltk.chunk` — identify named entities and noun phrase chunks")

    text = st.text_area("Input text", "Apple Inc. was founded by Steve Jobs in Cupertino, California. Barack Obama visited New York City.")

    tabs = st.tabs(["ne_chunk", "RegexpChunkParser", "ChunkScore"])

    with tabs[0]:
        st.subheader("ne_chunk — Named Entity Recognition")
        binary = st.checkbox("Binary mode (just NE / not-NE)", False)
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tagged, binary=binary)
        entities = []
        for subtree in tree:
            if hasattr(subtree, "label"):
                entity = " ".join(w for w, t in subtree.leaves())
                entities.append({"Entity": entity, "Type": subtree.label()})
        if entities:
            st.table(pd.DataFrame(entities))
        else:
            st.info("No named entities found.")
        st.text(str(tree))
        runnable_code("""import nltk
texts = [
    "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California.",
    "Barack Obama visited New York City and met with the United Nations Secretary-General.",
    "Amazon's CEO Andy Jassy announced a new headquarters in Arlington, Virginia.",
    "The Eiffel Tower in Paris, France attracts millions of tourists every year.",
]
for text in texts:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(tagged)
    entities = [(\" \".join(w for w, t in sub.leaves()), sub.label())
                for sub in tree if hasattr(sub, 'label')]
    print(f"Text    : {text}")
    print(f"Entities: {entities}\\n")""")

    with tabs[1]:
        st.subheader("RegexpChunkParser — rule-based NP chunking")
        grammar = st.text_area("Chunk grammar", "NP: {<DT>?<JJ>*<NN.*>+}", height=80)
        try:
            from nltk.chunk import RegexpChunkParser
            from nltk.chunk.regexp import ChunkRule
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            cp = nltk.RegexpParser(grammar)
            result = cp.parse(tagged)
            chunks = []
            for subtree in result:
                if hasattr(subtree, "label"):
                    phrase = " ".join(w for w, t in subtree.leaves())
                    chunks.append({"Chunk": phrase, "Type": subtree.label()})
            if chunks:
                st.table(pd.DataFrame(chunks))
            st.text(str(result))
        except Exception as e:
            st.error(str(e))
        runnable_code("""import nltk
grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
cp = nltk.RegexpParser(grammar)
tagged = nltk.pos_tag(nltk.word_tokenize("The big brown dog sat."))
result = cp.parse(tagged)
print(result)""")

    with tabs[2]:
        st.subheader("ChunkScore — evaluate chunker accuracy")
        runnable_code("""from nltk.chunk.util import ChunkScore
# ChunkScore compares chunker output to a gold-standard
# and reports precision, recall, and F-measure.
scorer = ChunkScore()
# scorer.score(chunk_tree)  # called per-sentence
print("Precision:", scorer.precision())
print("Recall:   ", scorer.recall())
print("F-Measure:", scorer.f_measure())""")
        st.info("ChunkScore requires a gold-standard annotated corpus to evaluate against. The code above shows how it works — load a corpus like `conll2000`, parse each sentence, and compare.")


# ════════════════════════════════════════════════════════════════════════════
# 5. SENTIMENT ANALYSIS (VADER)
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("Sentiment Analysis — VADER")
    st.markdown("`nltk.sentiment.vader.SentimentIntensityAnalyzer` — rule-based sentiment scoring")

    text = st.text_area("Enter text to analyze", "NLTK is absolutely amazing! I love working with natural language processing. It's not terrible at all.")

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    tabs = st.tabs(["SentimentIntensityAnalyzer", "SentimentAnalyzer (ML)", "mark_negation"])

    with tabs[0]:
        st.subheader("SentimentIntensityAnalyzer")
        scores = sia.polarity_scores(text)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Negative", f"{scores['neg']:.3f}")
        col2.metric("Neutral", f"{scores['neu']:.3f}")
        col3.metric("Positive", f"{scores['pos']:.3f}")
        col4.metric("Compound", f"{scores['compound']:.3f}")

        sentiment = "Positive 😊" if scores["compound"] >= 0.05 else ("Negative 😞" if scores["compound"] <= -0.05 else "Neutral 😐")
        st.success(f"Overall sentiment: **{sentiment}**")

        st.subheader("Sentence-by-sentence analysis")
        sents = nltk.sent_tokenize(text)
        rows = []
        for s in sents:
            sc = sia.polarity_scores(s)
            rows.append({"Sentence": s, "Neg": sc["neg"], "Neu": sc["neu"], "Pos": sc["pos"], "Compound": sc["compound"]})
        st.dataframe(pd.DataFrame(rows))

        runnable_code("""from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

sentences = [
    "NLTK is absolutely amazing! I love it!",
    "This is the worst library I have ever used.",
    "The documentation is okay, nothing special.",
    "I can't believe how incredibly powerful this tool is!",
    "The API is not bad, but it could be better.",
    "Terrible performance. Completely unusable. Very disappointed.",
]
print(f"{'Sentence':<50} {'neg':>6} {'neu':>6} {'pos':>6} {'compound':>9} {'Verdict'}")
print("-" * 90)
for sent in sentences:
    sc = sia.polarity_scores(sent)
    verdict = "Positive" if sc['compound'] >= 0.05 else ("Negative" if sc['compound'] <= -0.05 else "Neutral")
    print(f"{sent[:48]:<50} {sc['neg']:>6.3f} {sc['neu']:>6.3f} {sc['pos']:>6.3f} {sc['compound']:>9.4f} {verdict}")""")

    with tabs[1]:
        st.subheader("SentimentAnalyzer (trainable ML-based)")
        runnable_code("""from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import (
    extract_unigram_feats, extract_bigram_feats,
    mark_negation, split_train_test
)
# The SentimentAnalyzer wraps a trainable classifier.
# Example with movie_reviews corpus:
from nltk.corpus import movie_reviews
import random

docs = [(list(movie_reviews.words(fileid)), cat)
        for cat in movie_reviews.categories()
        for fileid in movie_reviews.fileids(cat)]
random.shuffle(docs)
train, test = docs[:1500], docs[1500:]

sa = SentimentAnalyzer()
all_words = sa.all_words([doc for doc, _ in train], low_freq_thresh=10)
unigram_feats = sa.unigram_word_feats(all_words, top_n=200)
sa.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

train_set = sa.apply_features(train)
test_set  = sa.apply_features(test)

trainer = nltk.classify.NaiveBayesClassifier.train
classifier = sa.train(trainer, train_set)
print(sa.evaluate(test_set))""")
        st.info("Training on `movie_reviews` takes ~30-60 seconds. Run this code in a Python script for the full interactive result.")

    with tabs[2]:
        st.subheader("mark_negation")
        from nltk.sentiment.util import mark_negation
        sample = st.text_input("Sentence", "I do not like this product at all")
        tokens = nltk.word_tokenize(sample)
        marked = mark_negation(tokens)
        st.write(marked)
        runnable_code("""from nltk.sentiment.util import mark_negation
import nltk
tokens = nltk.word_tokenize("I do not like this product at all")
marked = mark_negation(tokens)
print(marked)
# ['I', 'do', 'not', 'like_NEG', 'this_NEG', 'product_NEG', 'at_NEG', 'all_NEG', '.']""")


# ════════════════════════════════════════════════════════════════════════════
# 6. FREQUENCY DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("Frequency Distribution")
    st.markdown("`nltk.FreqDist`, `nltk.ConditionalFreqDist` — count and explore word frequencies")

    text = st.text_area("Input text", "To be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles")
    tokens = nltk.word_tokenize(text.lower())
    top_n = st.slider("Show top N words", 5, 30, 15)

    tabs = st.tabs(["FreqDist", "ConditionalFreqDist", "hapaxes & Nr"])

    with tabs[0]:
        st.subheader("FreqDist")
        fd = nltk.FreqDist(tokens)
        top = fd.most_common(top_n)
        st.bar_chart(pd.DataFrame(top, columns=["Word", "Count"]).set_index("Word"))
        st.write(f"**Total tokens (N):** {fd.N()}  |  **Unique types (B):** {fd.B()}")
        st.table(pd.DataFrame(top, columns=["Word", "Count"]))
        runnable_code("""import nltk
text = (
    "To be or not to be, that is the question. "
    "Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles, and by opposing end them. "
    "To die, to sleep, no more; and by a sleep to say we end the heartache."
)
tokens = nltk.word_tokenize(text.lower())
fd = nltk.FreqDist(tokens)

print("Top 10 most common:", fd.most_common(10))
print("Total tokens (N):", fd.N())
print("Unique types (B):", fd.B())
print("Most common word:", fd.max())
print("Freq of 'to':   ", round(fd.freq('to'), 4))
print("Count of 'be':  ", fd['be'])
print("Nr(1) hapax count:", fd.Nr(1))
print("Hapaxes:", fd.hapaxes())""")

    with tabs[1]:
        st.subheader("ConditionalFreqDist — frequency by category")
        from nltk.corpus import brown
        genres = st.multiselect("Brown corpus genres", brown.categories(), default=["news", "fiction", "romance"])
        if genres:
            cfd = nltk.ConditionalFreqDist(
                (genre, word.lower())
                for genre in genres
                for word in brown.words(categories=genre)
            )
            words_to_check = st.text_input("Words to compare", "the, a, said, man, time").split(",")
            words_to_check = [w.strip() for w in words_to_check if w.strip()]
            rows = []
            for w in words_to_check:
                row = {"Word": w}
                for g in genres:
                    row[g] = cfd[g][w]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows))
        runnable_code("""import nltk
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word.lower())
    for genre in ['news', 'fiction']
    for word in brown.words(categories=genre)
)
cfd.tabulate(conditions=['news', 'fiction'],
             samples=['the', 'a', 'man', 'said'])""")

    with tabs[2]:
        st.subheader("hapaxes() and Nr()")
        fd = nltk.FreqDist(tokens)
        hapaxes = fd.hapaxes()
        st.write(f"**Hapaxes** (words appearing exactly once): `{hapaxes}`")
        st.write(f"**Nr(1)** = {fd.Nr(1)}  |  **Nr(2)** = {fd.Nr(2)}")
        runnable_code("""import nltk
text = (
    "To be or not to be, that is the question. "
    "Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles and by opposing end them."
)
fd = nltk.FreqDist(nltk.word_tokenize(text.lower()))

print("Hapaxes (appear once):", fd.hapaxes())
print("Nr(1) - count of once-words:", fd.Nr(1))
print("Nr(2) - count of twice-words:", fd.Nr(2))
print("Most common word:", fd.max())
print("Relative freq of 'to':", round(fd.freq('to'), 4))
print("Raw count of 'to':", fd['to'])""")


# ════════════════════════════════════════════════════════════════════════════
# 7. COLLOCATIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    st.title("Collocations")
    st.markdown("`nltk.collocations` — find statistically significant word pairs/triples")

    from nltk.corpus import inaugural
    text_src = st.selectbox("Corpus", ["Custom text", "Inaugural speeches", "Brown corpus (news)"])
    if text_src == "Custom text":
        raw = st.text_area("Enter text", DEFAULT_TEXT * 10, height=120)
        tokens = nltk.word_tokenize(raw.lower())
    elif text_src == "Inaugural speeches":
        tokens = [w.lower() for w in inaugural.words()]
    else:
        from nltk.corpus import brown
        tokens = [w.lower() for w in brown.words(categories="news")]

    top_n = st.slider("Top N collocations", 5, 20, 10)
    tabs = st.tabs(["BigramCollocationFinder", "TrigramCollocationFinder", "QuadgramCollocationFinder"])

    with tabs[0]:
        st.subheader("BigramCollocationFinder")
        from nltk.collocations import BigramCollocationFinder
        from nltk.metrics import BigramAssocMeasures
        bcf = BigramCollocationFinder.from_words(tokens)
        bcf.apply_freq_filter(2)
        measure = st.selectbox("Association measure", ["pmi", "likelihood_ratio", "raw_freq", "student_t"], key="bi")
        try:
            scored = bcf.score_ngrams(getattr(BigramAssocMeasures(), measure))[:top_n]
            st.table(pd.DataFrame(scored, columns=["Bigram", "Score"]))
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import nltk
tokens = nltk.word_tokenize("the quick brown fox jumps over the lazy dog " * 10)
bcf = BigramCollocationFinder.from_words(tokens)
bcf.apply_freq_filter(2)
print(bcf.nbest(BigramAssocMeasures.pmi, 10))""")

    with tabs[1]:
        st.subheader("TrigramCollocationFinder")
        from nltk.collocations import TrigramCollocationFinder
        from nltk.metrics import TrigramAssocMeasures
        tcf = TrigramCollocationFinder.from_words(tokens)
        tcf.apply_freq_filter(2)
        try:
            scored = tcf.score_ngrams(TrigramAssocMeasures.likelihood_ratio)[:top_n]
            st.table(pd.DataFrame(scored, columns=["Trigram", "Score"]))
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
tokens = "the big brown fox jumps over the lazy dog".split() * 20
tcf = TrigramCollocationFinder.from_words(tokens)
tcf.apply_freq_filter(2)
print(tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 5))""")

    with tabs[2]:
        st.subheader("QuadgramCollocationFinder")
        from nltk.collocations import QuadgramCollocationFinder
        from nltk.metrics import QuadgramAssocMeasures
        qcf = QuadgramCollocationFinder.from_words(tokens)
        qcf.apply_freq_filter(3)
        try:
            scored = qcf.score_ngrams(QuadgramAssocMeasures.likelihood_ratio)[:top_n]
            st.table(pd.DataFrame(scored, columns=["Quadgram", "Score"]))
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.collocations import QuadgramCollocationFinder
from nltk.metrics import QuadgramAssocMeasures
tokens = "the quick brown fox jumps over the lazy dog".split() * 30
qcf = QuadgramCollocationFinder.from_words(tokens)
qcf.apply_freq_filter(3)
print(qcf.nbest(QuadgramAssocMeasures.likelihood_ratio, 5))""")


# ════════════════════════════════════════════════════════════════════════════
# 8. N-GRAMS & TEXT UTILITIES
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    st.title("N-grams & Text Utilities")
    st.markdown("`nltk.util` — bigrams, trigrams, ngrams, skipgrams, everygrams, pad_sequence")

    text = st.text_area("Input text", "The quick brown fox jumps over the lazy dog")
    tokens = nltk.word_tokenize(text)

    tabs = st.tabs(["bigrams", "trigrams", "ngrams", "skipgrams", "everygrams", "pad_sequence", "FreqDist on ngrams"])

    with tabs[0]:
        st.subheader("bigrams()")
        result = list(nltk.bigrams(tokens))
        st.write(result)
        runnable_code("""import nltk
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Natural language processing is fascinating",
    "NLTK makes NLP easy and fun",
]
for sent in sentences:
    tokens = nltk.word_tokenize(sent)
    bigrams = list(nltk.bigrams(tokens))
    print(f"Sentence : {sent}")
    print(f"Bigrams  : {bigrams}")
    print(f"Count    : {len(bigrams)}\\n")""")

    with tabs[1]:
        st.subheader("trigrams()")
        result = list(nltk.trigrams(tokens))
        st.write(result)
        runnable_code("""import nltk
tokens = "the quick brown fox".split()
print(list(nltk.trigrams(tokens)))
# [('the','quick','brown'), ('quick','brown','fox')]""")

    with tabs[2]:
        st.subheader("ngrams()")
        n = st.slider("n", 2, 6, 3, key="ngram_n")
        result = list(nltk.ngrams(tokens, n))
        st.write(result)
        runnable_code("""import nltk
tokens = "the quick brown fox jumps".split()
print(list(nltk.ngrams(tokens, 4)))
# [('the','quick','brown','fox'), ('quick','brown','fox','jumps')]""")

    with tabs[3]:
        st.subheader("skipgrams()")
        n = st.slider("n", 2, 4, 2, key="skip_n")
        k = st.slider("k (max skip)", 1, 4, 2, key="skip_k")
        result = list(nltk.skipgrams(tokens, n, k))
        st.write(result)
        runnable_code("""import nltk
tokens = "the quick brown fox".split()
# skipgrams(tokens, n=2, k=2): bigrams allowing up to 2 skips
print(list(nltk.skipgrams(tokens, 2, 2)))""")

    with tabs[4]:
        st.subheader("everygrams()")
        max_n = st.slider("max_len", 2, 5, 3, key="every_n")
        result = list(nltk.everygrams(tokens, max_len=max_n))
        st.write(result)
        runnable_code("""import nltk
tokens = "the quick brown".split()
print(list(nltk.everygrams(tokens, max_len=3)))
# all n-grams from 1 to max_len""")

    with tabs[5]:
        st.subheader("pad_sequence()")
        from nltk.util import pad_sequence
        n = st.slider("n", 2, 4, 2, key="pad_n")
        result = list(pad_sequence(tokens, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
        st.write(result)
        runnable_code("""from nltk.util import pad_sequence
tokens = "the quick brown fox".split()
padded = list(pad_sequence(tokens, n=2,
    pad_left=True, pad_right=True,
    left_pad_symbol='<s>', right_pad_symbol='</s>'))
print(padded)""")

    with tabs[6]:
        st.subheader("FreqDist on n-grams")
        n = st.slider("n-gram size", 2, 4, 2, key="fdist_n")
        ng = list(nltk.ngrams(tokens, n))
        fd = nltk.FreqDist(ng)
        top = fd.most_common(10)
        if top:
            labels = [str(g) for g, _ in top]
            counts = [c for _, c in top]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(labels, counts)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        runnable_code("""import nltk
tokens = nltk.word_tokenize("to be or not to be that is the question")
bigrams = list(nltk.bigrams(tokens))
fd = nltk.FreqDist(bigrams)
print(fd.most_common(5))""")


# ════════════════════════════════════════════════════════════════════════════
# 9. WORDNET
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[8]:
    st.title("WordNet")
    st.markdown("`nltk.corpus.wordnet` — lexical database of English")

    from nltk.corpus import wordnet as wn

    word = st.text_input("Look up word", "bank")
    pos_options = {"All": None, "Noun (n)": wn.NOUN, "Verb (v)": wn.VERB, "Adjective (a)": wn.ADJ, "Adverb (r)": wn.ADV}
    pos_label = st.selectbox("Part of Speech filter", list(pos_options.keys()))
    pos_filter = pos_options[pos_label]

    tabs = st.tabs(["Synsets", "Definitions & Examples", "Lemmas", "Hypernyms/Hyponyms", "Similarity", "Morphy", "Lesk WSD"])

    synsets = wn.synsets(word, pos=pos_filter)

    with tabs[0]:
        st.subheader(f"Synsets for '{word}'")
        if synsets:
            for ss in synsets:
                st.write(f"- `{ss.name()}` — {ss.definition()}")
        else:
            st.info("No synsets found.")
        runnable_code("""from nltk.corpus import wordnet as wn
synsets = wn.synsets('bank')
for ss in synsets:
    print(ss.name(), '-', ss.definition())""")

    with tabs[1]:
        st.subheader("Definitions & Examples")
        for ss in synsets:
            st.markdown(f"**`{ss.name()}`**")
            st.write(f"Definition: {ss.definition()}")
            if ss.examples():
                st.write(f"Examples: {ss.examples()}")
        runnable_code("""from nltk.corpus import wordnet as wn
ss = wn.synset('bank.n.01')
print(ss.definition())
print(ss.examples())""")

    with tabs[2]:
        st.subheader("Lemmas")
        rows = []
        for ss in synsets:
            for lemma in ss.lemmas():
                rows.append({"Synset": ss.name(), "Lemma": lemma.name(), "Antonyms": [a.name() for a in lemma.antonyms()]})
        if rows:
            st.dataframe(pd.DataFrame(rows))
        runnable_code("""from nltk.corpus import wordnet as wn
for ss in wn.synsets('good'):
    for lemma in ss.lemmas():
        print(lemma.name(), '| antonyms:', [a.name() for a in lemma.antonyms()])""")

    with tabs[3]:
        st.subheader("Hypernyms & Hyponyms")
        if synsets:
            ss = st.selectbox("Choose synset", [s.name() for s in synsets])
            chosen = wn.synset(ss)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Hypernyms** (more general)")
                for h in chosen.hypernyms():
                    st.write(f"- `{h.name()}` — {h.definition()}")
            with col2:
                st.markdown("**Hyponyms** (more specific)")
                for h in chosen.hyponyms()[:10]:
                    st.write(f"- `{h.name()}` — {h.definition()}")
        runnable_code("""from nltk.corpus import wordnet as wn
ss = wn.synset('dog.n.01')
print("Hypernyms:", ss.hypernyms())
print("Hyponyms:",  ss.hyponyms()[:5])
print("Root hypernym:", ss.root_hypernyms())""")

    with tabs[4]:
        st.subheader("Semantic Similarity")
        word2 = st.text_input("Compare with word", "river")
        ss1_list = wn.synsets(word, pos=wn.NOUN)
        ss2_list = wn.synsets(word2, pos=wn.NOUN)
        if ss1_list and ss2_list:
            ss1, ss2 = ss1_list[0], ss2_list[0]
            wup = ss1.wup_similarity(ss2)
            path = ss1.path_similarity(ss2)
            lch = ss1.lch_similarity(ss2) if ss1.pos() == ss2.pos() else None
            col1, col2, col3 = st.columns(3)
            col1.metric("Wu-Palmer", f"{wup:.4f}" if wup else "N/A")
            col2.metric("Path", f"{path:.4f}" if path else "N/A")
            col3.metric("Leacock-Chodorow", f"{lch:.4f}" if lch else "N/A")
        runnable_code("""from nltk.corpus import wordnet as wn
ss1 = wn.synset('car.n.01')
ss2 = wn.synset('bus.n.01')
print(ss1.wup_similarity(ss2))   # Wu-Palmer
print(ss1.path_similarity(ss2))  # Path similarity
print(ss1.lch_similarity(ss2))   # Leacock-Chodorow""")

    with tabs[5]:
        st.subheader("morphy() — morphological analysis")
        morphy_word = st.text_input("Word for morphy", "dogs")
        result = wn.morphy(morphy_word)
        st.write(f"morphy('{morphy_word}') → `{result}`")
        runnable_code("""from nltk.corpus import wordnet as wn
print(wn.morphy('dogs'))    # dog
print(wn.morphy('running')) # run
print(wn.morphy('better', wn.ADJ))  # good""")

    with tabs[6]:
        st.subheader("Lesk Algorithm — Word Sense Disambiguation")
        from nltk.wsd import lesk
        sentence = st.text_input("Sentence", "I went to the bank to deposit money")
        target = st.text_input("Target word", "bank")
        sent_tokens = nltk.word_tokenize(sentence)
        result = lesk(sent_tokens, target)
        if result:
            st.write(f"**Best synset:** `{result.name()}`")
            st.write(f"**Definition:** {result.definition()}")
        runnable_code("""from nltk.wsd import lesk
import nltk
sentence = nltk.word_tokenize("I went to the bank to deposit money")
sense = lesk(sentence, 'bank')
print(sense.name())      # bank.n.02
print(sense.definition())""")


# ════════════════════════════════════════════════════════════════════════════
# 10. LANGUAGE MODELS
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[9]:
    st.title("Language Models")
    st.markdown("`nltk.lm` — train and evaluate n-gram language models")

    from nltk.lm import MLE, Laplace, Lidstone, KneserNeyInterpolated, WittenBellInterpolated, StupidBackoff
    from nltk.lm.preprocessing import padded_everygram_pipeline
    from nltk.util import ngrams as nltk_ngrams

    RICH_CORPUS = (
        "The cat sat on the mat. The cat ate a fat rat. The rat ran from the cat. "
        "A big dog chased the cat up the tree. The dog barked loudly at the cat. "
        "The cat jumped down and ran into the house. The house was warm and cozy. "
        "Natural language processing is a fascinating field of computer science. "
        "Machine learning models can learn from large amounts of text data. "
        "The quick brown fox jumps over the lazy dog every single morning. "
        "Scientists discovered new methods for training language models efficiently. "
        "Deep learning has revolutionized the way computers understand human language. "
        "Words carry meaning and context that models try to learn and represent. "
        "The model predicts the next word based on the previous words in a sequence. "
    )

    col_left, col_right = st.columns([3, 1])
    with col_left:
        training_text = st.text_area("Training corpus", RICH_CORPUS, height=130)
    with col_right:
        n = st.slider("N-gram order", 2, 5, 2)
        model_name = st.selectbox("Model", ["MLE", "Laplace", "Lidstone",
                                            "KneserNeyInterpolated",
                                            "WittenBellInterpolated",
                                            "StupidBackoff"])

    train_sents = [nltk.word_tokenize(s.lower())
                   for s in nltk.sent_tokenize(training_text) if s.strip()]

    # show live n-gram stats so user can see slider effect immediately
    all_tokens = [tok for sent in train_sents for tok in sent]
    unique_ngrams = len(set(nltk_ngrams(all_tokens, n)))
    unique_unigrams = len(set(all_tokens))
    st.info(
        f"**Current settings →** n = **{n}** | model = **{model_name}** | "
        f"tokens = **{len(all_tokens)}** | unique {n}-grams = **{unique_ngrams}** | "
        f"vocabulary = **{unique_unigrams}** unique words"
    )

    train_data, vocab = padded_everygram_pipeline(n, train_sents)

    model_map = {
        "MLE": MLE, "Laplace": Laplace,
        "KneserNeyInterpolated": KneserNeyInterpolated,
        "WittenBellInterpolated": WittenBellInterpolated,
    }

    try:
        if model_name == "Lidstone":
            gamma = st.slider("Gamma (Lidstone smoothing)", 0.01, 1.0, 0.1)
            model = Lidstone(gamma, n)
        elif model_name == "StupidBackoff":
            alpha = st.slider("Alpha", 0.1, 1.0, 0.4)
            model = StupidBackoff(alpha, n)
        else:
            model = model_map[model_name](n)

        model.fit(train_data, vocab)

        tabs = st.tabs(["Score / Perplexity", "Text Generation", "Vocabulary", "Sample Code"])

        with tabs[0]:
            st.subheader(f"Score — P(word | context)  [n={n}]")
            test_words = [
                ("cat",     ["the"]),
                ("dog",     ["the"]),
                ("language",["natural"]),
                ("model",   ["language"]),
                ("sat",     ["cat"]),
                ("unknown", ["the"]),
            ]
            rows = []
            for word, ctx in test_words:
                context = ctx[-(n-1):] if n > 1 else []
                try:
                    prob = model.score(word, context if context else None)
                    rows.append({"Word": word, "Context": str(context), f"P(word|ctx) n={n}": f"{prob:.6f}"})
                except Exception:
                    rows.append({"Word": word, "Context": str(context), f"P(word|ctx) n={n}": "N/A"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("**Custom scoring:**")
            col1, col2 = st.columns(2)
            test_word = col1.text_input("Word to score", "cat")
            test_ctx  = col2.text_input(f"Context (last {n-1} word(s))", "the")
            ctx_tokens = nltk.word_tokenize(test_ctx.lower())[-(n-1):] if n > 1 else []
            try:
                score = model.score(test_word.lower(), ctx_tokens if ctx_tokens else None)
                st.metric(f"P('{test_word}' | {ctx_tokens})", f"{score:.6f}")
            except Exception as e:
                st.error(str(e))

            runnable_code("""from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
import nltk

corpus = (
    "The cat sat on the mat. The cat ate a fat rat. "
    "The rat ran from the cat. A big dog chased the cat. "
    "Natural language processing is fascinating. "
    "Machine learning models learn from text data. "
) * 5

# Try n=2, then n=3 and see how scores change!
for n in [2, 3]:
    sents = [nltk.word_tokenize(s.lower()) for s in nltk.sent_tokenize(corpus)]
    train_data, vocab = padded_everygram_pipeline(n, sents)
    model = Laplace(n)
    model.fit(train_data, vocab)

    print(f"=== n={n} ===")
    ctx = ['the'] if n == 2 else ['the', 'cat']
    print(f"  P('cat'  | {ctx}) = {model.score('cat',  ctx[-(n-1):]):.4f}")
    print(f"  P('dog'  | {ctx}) = {model.score('dog',  ctx[-(n-1):]):.4f}")
    print(f"  P('fish' | {ctx}) = {model.score('fish', ctx[-(n-1):]):.4f}  (unseen)")
    print()
""", key="lm_score_example")

        with tabs[1]:
            st.subheader(f"Text Generation  [n={n}]")
            num_words = st.slider("Words to generate", 5, 40, 20)
            seed_text = st.text_input("Seed context", "the cat")
            seed = nltk.word_tokenize(seed_text.lower())
            st.markdown("**Generated text from different random seeds:**")
            for rs in [0, 1, 42, 99]:
                try:
                    gen = model.generate(num_words, text_seed=seed if seed else None, random_seed=rs)
                    st.write(f"seed={rs}: `{' '.join(str(w) for w in gen)}`")
                except Exception as e:
                    st.write(f"seed={rs}: _{e}_")

            runnable_code("""from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import nltk

corpus = (
    "The cat sat on the mat. The cat ate a fat rat. "
    "The rat ran from the cat. A big dog chased the cat. "
    "Natural language processing is fascinating. "
    "Machine learning models learn from large text data. "
    "Deep learning changed the way we understand language. "
) * 8

# Compare generation quality at different n values
for n in [2, 3, 4]:
    sents = [nltk.word_tokenize(s.lower()) for s in nltk.sent_tokenize(corpus)]
    train_data, vocab = padded_everygram_pipeline(n, sents)
    model = MLE(n)
    model.fit(train_data, vocab)
    gen = model.generate(15, text_seed=['the'], random_seed=42)
    print(f"n={n}: {' '.join(str(w) for w in gen)}")
""", key="lm_gen_example")

        with tabs[2]:
            st.subheader("Vocabulary")
            vocab_list = sorted(list(model.vocab))
            st.metric("Vocabulary size", len(vocab_list))
            st.metric(f"Unique {n}-grams in corpus", unique_ngrams)
            st.write(vocab_list)
            runnable_code("""from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
import nltk

corpus = "The cat sat on the mat. The dog ran in the park. A bird sang in the tree." * 5
sents = [nltk.word_tokenize(s.lower()) for s in nltk.sent_tokenize(corpus)]
train_data, vocab = padded_everygram_pipeline(2, sents)
model = Laplace(2)
model.fit(train_data, vocab)

print(f"Vocabulary size : {len(list(model.vocab))}")
print(f"All vocab words : {sorted(list(model.vocab))}")
print(f"'cat' in vocab  : {'cat' in model.vocab}")
print(f"UNK lookup      : {model.vocab.lookup('elephant')}")
print(f"Vocab cutoff    : {model.vocab.cutoff}")
""", key="lm_vocab_example")

        with tabs[3]:
            sample_code = f"""from nltk.lm import {model_name}
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams
import nltk

n = {n}   # ← change this to see the effect!

corpus = (
    "The cat sat on the mat. The cat ate a fat rat. "
    "The rat ran from the cat. A big dog chased the cat. "
    "Natural language processing is fascinating. "
    "Machine learning models learn from text data. "
    "Deep learning changed the way we understand language. "
) * 8

sents = [nltk.word_tokenize(s.lower()) for s in nltk.sent_tokenize(corpus)]
train_data, vocab = padded_everygram_pipeline(n, sents)

model = {model_name}(n)
model.fit(train_data, vocab)

# Stats that change with n
all_tokens = [t for s in sents for t in s]
unique_ng = len(set(ngrams(all_tokens, n)))
print(f"n = {{n}}  |  vocab = {{len(list(model.vocab))}}  |  unique {{n}}-grams = {{unique_ng}}")

# Scoring — context length grows with n
ctx = ['the'] * (n - 1)
for word in ['cat', 'dog', 'language', 'model', 'elephant']:
    prob = model.score(word, ctx[-(n-1):] if n > 1 else None)
    print(f"  P('{word}'  | {ctx}) = {prob:.6f}")

# Generate text
print("\\nGenerated:")
gen = model.generate(20, text_seed=['the'], random_seed=42)
print(' '.join(str(w) for w in gen))"""
            runnable_code(sample_code, key=f"lm_sample_{model_name}_{n}")

    except Exception as e:
        st.error(f"Model error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# 11. TEXT METRICS & DISTANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[10]:
    st.title("Text Metrics & Distance")
    st.markdown("`nltk.metrics` — string distances, scores, and annotation agreement")

    tabs = st.tabs([
        "edit_distance", "Jaro / Jaro-Winkler", "Jaccard Distance",
        "MASI Distance", "Precision / Recall / F-measure",
        "ConfusionMatrix", "Spearman Correlation",
    ])

    with tabs[0]:
        st.subheader("edit_distance (Levenshtein)")
        from nltk.metrics.distance import edit_distance, edit_distance_align
        s1 = st.text_input("String 1", "intention")
        s2 = st.text_input("String 2", "execution")
        transpositions = st.checkbox("Allow transpositions (Damerau-Levenshtein)", False)
        dist = edit_distance(s1, s2, transpositions=transpositions)
        st.metric("Edit Distance", dist)
        try:
            alignment = edit_distance_align(s1, s2)
            st.write("Alignment path:", alignment)
        except Exception:
            pass
        runnable_code("""from nltk.metrics.distance import edit_distance
pairs = [
    ("intention",  "execution"),
    ("kitten",     "sitting"),
    ("cat",        "cut"),
    ("saturday",   "sunday"),
    ("algorithm",  "altruistic"),
    ("python",     "python"),   # identical
    ("nlp",        ""),         # empty string
]
print(f"{'String 1':<15} {'String 2':<15} {'Edit Distance'}")
print("-" * 45)
for s1, s2 in pairs:
    dist = edit_distance(s1, s2)
    print(f"{s1:<15} {s2:<15} {dist}")""")

    with tabs[1]:
        st.subheader("Jaro & Jaro-Winkler Similarity")
        from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity
        s1 = st.text_input("String 1", "MARTHA", key="j1")
        s2 = st.text_input("String 2", "MARHTA", key="j2")
        p = st.slider("Winkler prefix scale (p)", 0.0, 0.25, 0.1)
        col1, col2 = st.columns(2)
        col1.metric("Jaro Similarity", f"{jaro_similarity(s1, s2):.4f}")
        col2.metric("Jaro-Winkler Similarity", f"{jaro_winkler_similarity(s1, s2, p=p):.4f}")
        runnable_code("""from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity
print(jaro_similarity("MARTHA", "MARHTA"))         # 0.9444
print(jaro_winkler_similarity("MARTHA", "MARHTA")) # 0.9611""")

    with tabs[2]:
        st.subheader("Jaccard Distance")
        from nltk.metrics.distance import jaccard_distance
        s1 = st.text_input("Set 1 (space-separated)", "cat dog bird fish", key="jac1")
        s2 = st.text_input("Set 2 (space-separated)", "cat dog lion tiger", key="jac2")
        set1, set2 = set(s1.split()), set(s2.split())
        dist = jaccard_distance(set1, set2)
        st.metric("Jaccard Distance", f"{dist:.4f}")
        st.write(f"Intersection: `{set1 & set2}`  |  Union: `{set1 | set2}`")
        runnable_code("""from nltk.metrics.distance import jaccard_distance
A = set(['cat', 'dog', 'bird'])
B = set(['cat', 'dog', 'lion'])
print(jaccard_distance(A, B))  # 0.5""")

    with tabs[3]:
        st.subheader("MASI Distance (for semantic similarity)")
        from nltk.metrics.distance import masi_distance
        s1 = st.text_input("Set 1", "the cat sat on the mat", key="masi1")
        s2 = st.text_input("Set 2", "the cat sat", key="masi2")
        set1, set2 = frozenset(s1.split()), frozenset(s2.split())
        dist = masi_distance(set1, set2)
        st.metric("MASI Distance", f"{dist:.4f}")
        runnable_code("""from nltk.metrics.distance import masi_distance
A = frozenset(['the', 'cat', 'sat', 'on', 'the', 'mat'])
B = frozenset(['the', 'cat', 'sat'])
print(masi_distance(A, B))""")

    with tabs[4]:
        st.subheader("Precision, Recall & F-Measure")
        from nltk.metrics.scores import precision, recall, f_measure
        ref_input = st.text_input("Reference set", "the cat sat on the mat")
        test_input = st.text_input("Test set", "the cat sat on floor")
        ref_set = set(ref_input.split())
        test_set = set(test_input.split())
        p = precision(ref_set, test_set)
        r = recall(ref_set, test_set)
        f = f_measure(ref_set, test_set)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{p:.4f}" if p else "N/A")
        col2.metric("Recall", f"{r:.4f}" if r else "N/A")
        col3.metric("F-Measure", f"{f:.4f}" if f else "N/A")
        runnable_code("""from nltk.metrics.scores import precision, recall, f_measure
reference = {'the', 'cat', 'sat', 'on', 'mat'}
test      = {'the', 'cat', 'sat', 'on', 'floor'}
print("Precision:", precision(reference, test))
print("Recall:   ", recall(reference, test))
print("F-measure:", f_measure(reference, test))""")

    with tabs[5]:
        st.subheader("ConfusionMatrix")
        from nltk.metrics.confusionmatrix import ConfusionMatrix
        ref  = ['DT', 'NN', 'VB', 'JJ', 'NN', 'VB', 'DT', 'NN']
        test = ['DT', 'NN', 'NN', 'JJ', 'VB', 'VB', 'DT', 'JJ']
        cm = ConfusionMatrix(ref, test)
        st.text(str(cm))
        runnable_code("""from nltk.metrics.confusionmatrix import ConfusionMatrix
reference = ['DT','NN','VB','JJ','NN','VB']
test      = ['DT','NN','NN','JJ','VB','VB']
cm = ConfusionMatrix(reference, test)
print(cm)
print(cm.pretty_format(sort_by_count=True))""")

    with tabs[6]:
        st.subheader("Spearman Correlation")
        from nltk.metrics.spearman import spearman_correlation, ranks_from_scores
        runnable_code("""from nltk.metrics.spearman import (
    spearman_correlation, ranks_from_scores, ranks_from_sequence
)
# ranks_from_scores: convert score dict to rank pairs
scores = {'a': 10, 'b': 7, 'c': 3, 'd': 1}
ranks = list(ranks_from_scores(scores.items()))

# spearman_correlation: compare two ranked lists
r = spearman_correlation(
    ranks_from_sequence(['a','b','c','d']),
    ranks_from_sequence(['a','c','b','d'])
)
print("Spearman r:", r)""")
        st.info("Spearman correlation is used to compare two ranked word lists, e.g. for evaluating distributional similarity models.")


# ════════════════════════════════════════════════════════════════════════════
# 12. TRANSLATION METRICS
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[11]:
    st.title("Translation Metrics")
    st.markdown("`nltk.translate` — BLEU, METEOR, NIST, RIBES, ChrF, GLEU and more")

    hypothesis = st.text_input("Hypothesis (machine translation)", "The cat is on the mat")
    reference  = st.text_input("Reference (human translation)",    "The cat sat on the mat")

    hyp_tokens = hypothesis.lower().split()
    ref_tokens  = reference.lower().split()

    tabs = st.tabs(["BLEU", "sentence_bleu", "METEOR", "NIST", "ChrF", "GLEU", "RIBES", "alignment_error_rate"])

    with tabs[0]:
        st.subheader("corpus_bleu")
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoother = SmoothingFunction()
        score = corpus_bleu([[ref_tokens]], [hyp_tokens], smoothing_function=smoother.method1)
        st.metric("Corpus BLEU", f"{score:.4f}")
        runnable_code("""from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
smooth = SmoothingFunction().method1

# Multiple hypotheses vs references
references = [
    [["the", "cat", "sat", "on", "the", "mat"]],
    [["the", "dog", "ran", "in", "the", "park"]],
    [["she", "sings", "a", "beautiful", "song"]],
]
hypotheses = [
    ["the", "cat", "is",  "on", "the", "mat"],   # close
    ["a",   "dog", "ran", "in", "the", "garden"], # slightly off
    ["she", "sang", "a", "nice", "melody"],        # more different
]
score = corpus_bleu(references, hypotheses, smoothing_function=smooth)
print(f"Corpus BLEU: {score:.4f}\\n")

# Show per-sentence BLEU for comparison
from nltk.translate.bleu_score import sentence_bleu
for ref, hyp in zip(references, hypotheses):
    s = sentence_bleu(ref, hyp, smoothing_function=smooth)
    print(f"  Ref: {ref[0]}")
    print(f"  Hyp: {hyp}")
    print(f"  BLEU: {s:.4f}\\n")""")

    with tabs[1]:
        st.subheader("sentence_bleu")
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        weights_label = st.selectbox("Weights", ["1-gram (1,0,0,0)", "2-gram (0.5,0.5,0,0)", "4-gram BLEU (0.25 each)"])
        weights_map = {
            "1-gram (1,0,0,0)": (1, 0, 0, 0),
            "2-gram (0.5,0.5,0,0)": (0.5, 0.5, 0, 0),
            "4-gram BLEU (0.25 each)": (0.25, 0.25, 0.25, 0.25),
        }
        weights = weights_map[weights_label]
        score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=SmoothingFunction().method1)
        st.metric("Sentence BLEU", f"{score:.4f}")
        runnable_code("""from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
ref = ["the", "cat", "sat", "on", "the", "mat"]
hyp = ["the", "cat", "is",  "on", "the", "mat"]
# Unigram BLEU
print(sentence_bleu([ref], hyp, weights=(1,0,0,0)))
# 4-gram BLEU
print(sentence_bleu([ref], hyp, weights=(0.25,)*4))""")

    with tabs[2]:
        st.subheader("METEOR Score")
        from nltk.translate.meteor_score import single_meteor_score
        try:
            score = single_meteor_score(reference.lower(), hypothesis.lower())
            st.metric("METEOR", f"{score:.4f}")
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.translate.meteor_score import single_meteor_score, meteor_score
ref = "the cat sat on the mat"
hyp = "the cat is on the mat"
print(single_meteor_score(ref, hyp))""")

    with tabs[3]:
        st.subheader("NIST Score")
        from nltk.translate.nist_score import sentence_nist
        try:
            score = sentence_nist([ref_tokens], hyp_tokens)
            st.metric("Sentence NIST", f"{score:.4f}")
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.translate.nist_score import sentence_nist, corpus_nist
ref = ["the", "cat", "sat", "on", "the", "mat"]
hyp = ["the", "cat", "is",  "on", "the", "mat"]
print(sentence_nist([ref], hyp))""")

    with tabs[4]:
        st.subheader("ChrF Score (Character n-gram F-score)")
        from nltk.translate.chrf_score import sentence_chrf
        score = sentence_chrf(reference, hypothesis)
        st.metric("ChrF", f"{score:.4f}")
        runnable_code("""from nltk.translate.chrf_score import sentence_chrf, corpus_chrf
ref = "the cat sat on the mat"
hyp = "the cat is on the mat"
print(sentence_chrf(ref, hyp))""")

    with tabs[5]:
        st.subheader("GLEU Score (Google's BLEU variant)")
        from nltk.translate.gleu_score import sentence_gleu
        score = sentence_gleu([ref_tokens], hyp_tokens)
        st.metric("Sentence GLEU", f"{score:.4f}")
        runnable_code("""from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
ref = ["the", "cat", "sat", "on", "the", "mat"]
hyp = ["the", "cat", "is",  "on", "the", "mat"]
print(sentence_gleu([ref], hyp))""")

    with tabs[6]:
        st.subheader("RIBES Score")
        from nltk.translate.ribes_score import sentence_ribes
        try:
            score = sentence_ribes([ref_tokens], hyp_tokens)
            st.metric("Sentence RIBES", f"{score:.4f}")
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.translate.ribes_score import sentence_ribes, corpus_ribes
ref = ["the", "cat", "sat", "on", "the", "mat"]
hyp = ["the", "cat", "is",  "on", "the", "mat"]
print(sentence_ribes([ref], hyp))""")

    with tabs[7]:
        st.subheader("alignment_error_rate")
        from nltk.translate.metrics import alignment_error_rate
        from nltk.translate.api import Alignment
        runnable_code("""from nltk.translate.metrics import alignment_error_rate
from nltk.translate.api import Alignment
# AER compares reference alignments to hypothesis alignments
ref  = Alignment([(0,0),(1,1),(2,2)])
hyp  = Alignment([(0,0),(1,1),(2,3)])
sure = Alignment([(0,0),(1,1)])
print(alignment_error_rate(ref, hyp, possible=sure))""")
        st.info("alignment_error_rate evaluates word alignment quality between parallel sentences. Requires an `Alignment` object from gold-standard data.")


# ════════════════════════════════════════════════════════════════════════════
# 13. GRAMMAR & PARSING
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[12]:
    st.title("Grammar & Parsing")
    st.markdown("`nltk.grammar`, `nltk.parse` — define and parse with CFG/PCFG grammars")

    tabs = st.tabs(["CFG", "PCFG", "RecursiveDescentParser", "ShiftReduceParser", "ChartParser", "EarleyChartParser", "DependencyGraph"])

    default_grammar = """
S -> NP VP
VP -> V NP | V NP PP
PP -> P NP
V -> "saw" | "ate" | "walked"
NP -> Det N | Det N PP | "I"
Det -> "a" | "an" | "the" | "my"
N -> "man" | "dog" | "cat" | "telescope" | "park"
P -> "in" | "on" | "by" | "with"
"""

    with tabs[0]:
        st.subheader("CFG — Context Free Grammar")
        grammar_str = st.text_area("Grammar", default_grammar.strip(), height=200)
        sentence = st.text_input("Sentence to parse", "I saw a man with a telescope")
        try:
            grammar = nltk.CFG.fromstring(grammar_str)
            st.write(f"**Productions:** {len(grammar.productions())}")
            st.write(f"**Start symbol:** `{grammar.start()}`")
            parser = nltk.ChartParser(grammar)
            trees = list(parser.parse(sentence.split()))
            if trees:
                st.success(f"Found {len(trees)} parse tree(s)")
                for i, tree in enumerate(trees[:3]):
                    st.text(str(tree))
            else:
                st.warning("No parse found for this sentence with the given grammar.")
        except Exception as e:
            st.error(str(e))
        runnable_code("""import nltk
grammar = nltk.CFG.fromstring('''
    S -> NP VP
    VP -> V NP
    NP -> Det N
    V -> "saw" | "ate"
    Det -> "the" | "a"
    N -> "cat" | "dog"
''')
parser = nltk.ChartParser(grammar)
for tree in parser.parse("the cat saw a dog".split()):
    print(tree)""")

    with tabs[1]:
        st.subheader("PCFG — Probabilistic Context Free Grammar")
        pcfg_str = """
S -> NP VP [1.0]
NP -> Det N [0.5] | NP PP [0.25] | 'I' [0.25]
VP -> V NP [0.6] | VP PP [0.4]
PP -> P NP [1.0]
Det -> 'the' [0.6] | 'a' [0.4]
N -> 'dog' [0.4] | 'cat' [0.4] | 'park' [0.2]
V -> 'saw' [0.5] | 'chased' [0.5]
P -> 'in' [0.5] | 'on' [0.5]
"""
        try:
            pcfg = nltk.PCFG.fromstring(pcfg_str.strip())
            st.write(f"**Start:** `{pcfg.start()}`  |  **Productions:** {len(pcfg.productions())}")
            for prod in pcfg.productions()[:8]:
                st.write(f"- `{prod}`")
        except Exception as e:
            st.error(str(e))
        runnable_code("""import nltk
grammar = nltk.PCFG.fromstring('''
    S  -> NP VP   [1.0]
    NP -> Det N   [0.6] | 'I' [0.4]
    VP -> V NP    [1.0]
    Det -> 'the'  [0.5] | 'a' [0.5]
    N  -> 'cat'   [0.5] | 'dog' [0.5]
    V  -> 'saw'   [1.0]
''')
from nltk.parse import ViterbiParser
viterbi = ViterbiParser(grammar)
for tree in viterbi.parse("I saw a cat".split()):
    print(tree)""")

    with tabs[2]:
        st.subheader("RecursiveDescentParser")
        simple_grammar = "S -> NP VP\nNP -> Det N\nVP -> V NP\nDet -> 'the'\nN -> 'cat' | 'dog'\nV -> 'saw'"
        try:
            g = nltk.CFG.fromstring(simple_grammar)
            rdp = nltk.RecursiveDescentParser(g)
            trees = list(rdp.parse("the cat saw the dog".split()))
            for t in trees:
                st.text(str(t))
        except Exception as e:
            st.error(str(e))
        runnable_code("""import nltk
grammar = nltk.CFG.fromstring('''
    S -> NP VP
    NP -> Det N
    VP -> V NP
    Det -> 'the'
    N -> 'cat' | 'dog'
    V -> 'saw'
''')
rd_parser = nltk.RecursiveDescentParser(grammar)
for tree in rd_parser.parse("the cat saw the dog".split()):
    print(tree)""")

    with tabs[3]:
        st.subheader("ShiftReduceParser")
        simple_grammar = "S -> NP VP\nNP -> Det N\nVP -> V NP\nDet -> 'the'\nN -> 'cat' | 'dog'\nV -> 'saw'"
        try:
            g = nltk.CFG.fromstring(simple_grammar)
            srp = nltk.ShiftReduceParser(g)
            trees = list(srp.parse("the cat saw the dog".split()))
            for t in trees:
                st.text(str(t))
        except Exception as e:
            st.error(str(e))
        runnable_code("""import nltk
grammar = nltk.CFG.fromstring('''
    S -> NP VP
    NP -> Det N
    VP -> V NP
    Det -> 'the'
    N -> 'cat' | 'dog'
    V -> 'saw'
''')
sr_parser = nltk.ShiftReduceParser(grammar)
for tree in sr_parser.parse("the cat saw the dog".split()):
    print(tree)""")

    with tabs[4]:
        st.subheader("ChartParser (Earley/CYK)")
        try:
            grammar_str2 = st.text_area("Grammar (ChartParser)", default_grammar.strip(), height=180, key="cp_gram")
            sentence2 = st.text_input("Sentence", "the dog saw a cat", key="cp_sent")
            g = nltk.CFG.fromstring(grammar_str2)
            cp = nltk.ChartParser(g)
            trees = list(cp.parse(sentence2.split()))
            st.success(f"{len(trees)} parse(s) found")
            for t in trees[:2]:
                st.text(str(t))
        except Exception as e:
            st.error(str(e))
        runnable_code("""import nltk
grammar = nltk.CFG.fromstring('''
    S -> NP VP
    NP -> Det N
    VP -> V NP
    Det -> 'the'
    N  -> 'cat' | 'dog'
    V  -> 'saw'
''')
parser = nltk.ChartParser(grammar)
for tree in parser.parse("the cat saw the dog".split()):
    print(tree)""")

    with tabs[5]:
        st.subheader("EarleyChartParser")
        from nltk.parse.earleychart import EarleyChartParser
        try:
            g = nltk.CFG.fromstring(default_grammar.strip())
            ep = EarleyChartParser(g)
            trees = list(ep.parse("I saw a man".split()))
            for t in trees[:2]:
                st.text(str(t))
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.parse.earleychart import EarleyChartParser
import nltk
grammar = nltk.CFG.fromstring('''
    S -> NP VP
    NP -> Det N | 'I'
    VP -> V NP
    Det -> 'a' | 'the'
    N -> 'man' | 'dog'
    V -> 'saw'
''')
parser = EarleyChartParser(grammar)
for tree in parser.parse("I saw a man".split()):
    print(tree)""")

    with tabs[6]:
        st.subheader("DependencyGraph")
        from nltk.parse.dependencygraph import DependencyGraph
        conll_data = """1\tI\t_\tPRON\tPRP\t_\t2\tnsubj\t_\t_
2\tsaw\t_\tVERB\tVBD\t_\t0\troot\t_\t_
3\tthe\t_\tDET\tDT\t_\t4\tdet\t_\t_
4\tman\t_\tNOUN\tNN\t_\t2\tdobj\t_\t_
"""
        try:
            dg = DependencyGraph(conll_data)
            st.text(str(dg.tree()))
            st.write("Nodes:")
            for n in list(dg.nodes.values())[1:]:
                st.write(f"  `{n['word']}` ← {n['rel']} ← head[{n['head']}]")
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.parse.dependencygraph import DependencyGraph
conll = '''1\\tI\\t_\\tPRON\\tPRP\\t_\\t2\\tnsubj\\t_\\t_
2\\tsaw\\t_\\tVERB\\tVBD\\t_\\t0\\troot\\t_\\t_
3\\tthe\\t_\\tDET\\tDT\\t_\\t4\\tdet\\t_\\t_
4\\tman\\t_\\tNOUN\\tNN\\t_\\t2\\tdobj\\t_\\t_
'''
dg = DependencyGraph(conll)
print(dg.tree())""")


# ════════════════════════════════════════════════════════════════════════════
# 14. CLASSIFICATION (NAIVE BAYES)
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[13]:
    st.title("Classification — Naive Bayes")
    st.markdown("`nltk.classify` — text classification with Naive Bayes, Decision Tree, and more")

    tabs = st.tabs(["NaiveBayesClassifier", "DecisionTreeClassifier", "SklearnClassifier", "TextCat"])

    with tabs[0]:
        st.subheader("NaiveBayesClassifier — Movie Review Sentiment")
        st.info("Training on 200 movie reviews (100 pos / 100 neg). Takes ~5 seconds.")
        if st.button("Train & Evaluate NaiveBayesClassifier"):
            from nltk.corpus import movie_reviews
            from nltk.classify import NaiveBayesClassifier, accuracy

            def word_features(words):
                return {w: True for w in words}

            with st.spinner("Training..."):
                docs = [(word_features(movie_reviews.words(fileid)), cat)
                        for cat in movie_reviews.categories()
                        for fileid in movie_reviews.fileids(cat)]
                import random; random.seed(42); random.shuffle(docs)
                train, test = docs[:150], docs[150:]
                classifier = NaiveBayesClassifier.train(train)
                acc = accuracy(classifier, test)

            st.success(f"Accuracy: **{acc:.2%}**")
            st.subheader("Most Informative Features")
            buf = io.StringIO()
            classifier.show_most_informative_features(10)
            feats = classifier.most_informative_features(10)
            rows = [{"Feature": f, "Ratio": f"{r:.1f}:1"} for f, r in feats]
            st.table(pd.DataFrame(rows))

            st.subheader("Classify your own text")
            custom = st.text_area("Custom review", "This movie was absolutely wonderful and brilliant!")
            custom_feats = word_features(nltk.word_tokenize(custom.lower()))
            pred = classifier.classify(custom_feats)
            prob = classifier.prob_classify(custom_feats)
            st.write(f"**Prediction:** {pred}  |  **P(pos):** {prob.prob('pos'):.3f}  |  **P(neg):** {prob.prob('neg'):.3f}")

        runnable_code("""from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier, accuracy

def word_features(words):
    return {word: True for word in words}

docs = [(word_features(movie_reviews.words(f)), cat)
        for cat in movie_reviews.categories()
        for f in movie_reviews.fileids(cat)]

train, test = docs[200:], docs[:200]
classifier = NaiveBayesClassifier.train(train)
print("Accuracy:", accuracy(classifier, test))
classifier.show_most_informative_features(10)""")

    with tabs[1]:
        st.subheader("DecisionTreeClassifier")
        runnable_code("""from nltk.classify import DecisionTreeClassifier

# Feature extraction function
def gender_features(name):
    return {'last_letter': name[-1], 'first_letter': name[0],
            'length': len(name)}

from nltk.corpus import names
import random
labeled = ([(n, 'male')   for n in names.words('male.txt')] +
           [(n, 'female') for n in names.words('female.txt')])
random.shuffle(labeled)

featuresets = [(gender_features(n), g) for n, g in labeled]
train, test = featuresets[500:], featuresets[:500]

dt = DecisionTreeClassifier.train(train, entropy_cutoff=0.8, depth_cutoff=5)
print("Accuracy:", nltk.classify.accuracy(dt, test))
print(dt.classify(gender_features("Alice")))   # female
print(dt.classify(gender_features("James")))   # male""")
        st.info("Run this in a script — training on the full names corpus takes a few seconds.")

    with tabs[2]:
        st.subheader("SklearnClassifier — wrap scikit-learn models")
        runnable_code("""from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.corpus import movie_reviews

def word_features(words):
    return {word: True for word in words}

docs = [(word_features(movie_reviews.words(f)), cat)
        for cat in movie_reviews.categories()
        for f in movie_reviews.fileids(cat)]
train, test = docs[200:], docs[:200]

# Use any sklearn classifier
svm_clf = SklearnClassifier(LinearSVC())
svm_clf.train(train)
print("SVM Accuracy:", nltk.classify.accuracy(svm_clf, test))

nb_clf = SklearnClassifier(MultinomialNB())
nb_clf.train(train)
print("MNB Accuracy:", nltk.classify.accuracy(nb_clf, test))""")
        st.warning("Requires `scikit-learn` installed: `pip install scikit-learn`")

    with tabs[3]:
        st.subheader("TextCat — language identification")
        from nltk.classify import TextCat
        text_to_identify = st.text_input("Enter text in any language", "Bonjour tout le monde, comment allez-vous?")
        try:
            tc = TextCat()
            lang = tc.guess_language(text_to_identify)
            st.metric("Detected Language", lang)
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.classify import TextCat
tc = TextCat()
print(tc.guess_language("Bonjour tout le monde"))  # fra
print(tc.guess_language("Hello world"))             # eng
print(tc.guess_language("Hola mundo"))              # spa""")


# ════════════════════════════════════════════════════════════════════════════
# 15. CONCORDANCE & TEXT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[14]:
    st.title("Concordance & Text Analysis")
    st.markdown("`nltk.text.Text` — concordance, collocations, similar words, dispersion plots")

    from nltk.text import Text, ConcordanceIndex, TextCollection

    corpus_choice = st.selectbox("Choose corpus", ["Custom text", "Inaugural speeches", "Gutenberg - Moby Dick"])
    if corpus_choice == "Custom text":
        raw = st.text_area("Your text", DEFAULT_TEXT * 20, height=120)
        tokens = nltk.word_tokenize(raw.lower())
    elif corpus_choice == "Inaugural speeches":
        from nltk.corpus import inaugural
        tokens = [w.lower() for w in inaugural.words()]
    else:
        from nltk.corpus import gutenberg
        tokens = [w.lower() for w in gutenberg.words("melville-moby_dick.txt")]

    text_obj = Text(tokens)

    tabs = st.tabs(["concordance", "similar", "common_contexts", "dispersion_plot", "collocations", "vocab / FreqDist", "TF-IDF"])

    with tabs[0]:
        st.subheader("concordance()")
        word = st.text_input("Word to search", "freedom")
        width = st.slider("Context width", 40, 100, 75)
        lines = st.slider("Max lines", 5, 30, 10)
        results = text_obj.concordance_list(word, width=width, lines=lines)
        if results:
            for entry in results:
                st.code(entry.line, language=None)
        else:
            st.info(f"'{word}' not found in corpus.")
        runnable_code("""import nltk
from nltk.text import Text
from nltk.corpus import inaugural
text = Text(inaugural.words())
text.concordance('freedom', width=75, lines=10)""")

    with tabs[1]:
        st.subheader("similar() — words used in similar contexts")
        word = st.text_input("Target word", "nation", key="sim_word")
        try:
            similar = text_obj.similar(word, num=20)
            st.write(similar)
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.text import Text
from nltk.corpus import inaugural
text = Text(inaugural.words())
text.similar('nation')  # words used in similar contexts""")

    with tabs[2]:
        st.subheader("common_contexts() — shared context for multiple words")
        words_in = st.text_input("Words (comma-separated)", "man, woman")
        word_list = [w.strip().lower() for w in words_in.split(",")]
        try:
            result = text_obj.common_contexts(word_list, num=10)
            st.write(result)
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.text import Text
from nltk.corpus import inaugural
text = Text([w.lower() for w in inaugural.words()])
text.common_contexts(['man', 'woman'])""")

    with tabs[3]:
        st.subheader("dispersion_plot() — word usage across the corpus")
        disp_words = st.text_input("Words to plot (comma-separated)", "freedom, democracy, liberty, peace, war")
        word_list = [w.strip().lower() for w in disp_words.split(",") if w.strip()]
        if word_list:
            try:
                fig, ax = plt.subplots(figsize=(12, max(3, len(word_list))))
                positions = {w: [] for w in word_list}
                for i, tok in enumerate(tokens):
                    if tok in positions:
                        positions[tok].append(i)
                for j, w in enumerate(word_list):
                    ax.plot(positions[w], [j] * len(positions[w]), "|", color="steelblue", markersize=8)
                ax.set_yticks(range(len(word_list)))
                ax.set_yticklabels(word_list)
                ax.set_xlabel("Word offset")
                ax.set_title("Dispersion Plot")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(str(e))
        runnable_code("""import nltk
from nltk.text import Text
from nltk.corpus import inaugural
text = Text(inaugural.words())
text.dispersion_plot(['freedom', 'democracy', 'liberty', 'peace'])""")

    with tabs[4]:
        st.subheader("collocations()")
        try:
            result = text_obj.collocation_list(num=20)
            st.write([" ".join(c) for c in result])
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.text import Text
from nltk.corpus import inaugural
text = Text(inaugural.words())
text.collocations()""")

    with tabs[5]:
        st.subheader("vocab() — FreqDist of the text")
        fd = text_obj.vocab()
        top_n = st.slider("Top N", 10, 50, 20, key="vocab_n")
        top = fd.most_common(top_n)
        st.bar_chart(pd.DataFrame(top, columns=["Word", "Count"]).set_index("Word"))
        runnable_code("""from nltk.text import Text
from nltk.corpus import inaugural
text = Text(inaugural.words())
vocab = text.vocab()
print(vocab.most_common(10))""")

    with tabs[6]:
        st.subheader("TextCollection — TF-IDF")
        doc1 = "the cat sat on the mat"
        doc2 = "the dog sat on the log"
        doc3 = "cats and dogs are pets"
        tc = TextCollection([doc1.split(), doc2.split(), doc3.split()])
        query_word = st.text_input("Query word for TF-IDF", "cat")
        doc_choice = st.selectbox("Document", [doc1, doc2, doc3])
        doc_tokens = doc_choice.split()
        tf = tc.tf(query_word, doc_tokens)
        idf = tc.idf(query_word)
        tfidf = tc.tf_idf(query_word, doc_tokens)
        col1, col2, col3 = st.columns(3)
        col1.metric("TF", f"{tf:.4f}")
        col2.metric("IDF", f"{idf:.4f}")
        col3.metric("TF-IDF", f"{tfidf:.4f}")
        runnable_code("""from nltk.text import TextCollection
d1 = "the cat sat on the mat".split()
d2 = "the dog sat on the log".split()
d3 = "cats and dogs are pets".split()
tc = TextCollection([d1, d2, d3])
print("TF:",     tc.tf('cat', d1))
print("IDF:",    tc.idf('cat'))
print("TF-IDF:", tc.tf_idf('cat', d1))""")


# ════════════════════════════════════════════════════════════════════════════
# 16. PROBABILITY DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[15]:
    st.title("Probability Distributions")
    st.markdown("`nltk.probability` — FreqDist-backed probability estimators")

    text = st.text_area("Input text", DEFAULT_TEXT, height=100)
    tokens = nltk.word_tokenize(text.lower())
    fd = nltk.FreqDist(tokens)

    tabs = st.tabs(["MLEProbDist", "LaplaceProbDist", "LidstoneProbDist",
                    "KneserNeyProbDist", "WittenBellProbDist", "ConditionalProbDist"])

    query_word = st.text_input("Query word", "nltk")

    with tabs[0]:
        st.subheader("MLEProbDist — Maximum Likelihood Estimate")
        from nltk.probability import MLEProbDist
        mle = MLEProbDist(fd)
        st.metric(f"P('{query_word}')", f"{mle.prob(query_word):.6f}")
        st.write(f"Max probability word: `{mle.max()}`")
        runnable_code("""from nltk.probability import MLEProbDist, FreqDist
import nltk
fd = FreqDist(nltk.word_tokenize("the cat sat on the mat the cat"))
mle = MLEProbDist(fd)
print(mle.prob('the'))  # 3/8 = 0.375
print(mle.max())        # 'the'""")

    with tabs[1]:
        st.subheader("LaplaceProbDist — Add-one smoothing")
        from nltk.probability import LaplaceProbDist
        lp = LaplaceProbDist(fd)
        st.metric(f"P('{query_word}')", f"{lp.prob(query_word):.6f}")
        st.metric("Discount", f"{lp.discount():.6f}")
        runnable_code("""from nltk.probability import LaplaceProbDist, FreqDist
fd = FreqDist(['cat', 'dog', 'cat', 'bird'])
lp = LaplaceProbDist(fd)
print(lp.prob('cat'))   # smoothed probability
print(lp.prob('fish'))  # nonzero even for unseen words""")

    with tabs[2]:
        st.subheader("LidstoneProbDist — Lidstone smoothing")
        from nltk.probability import LidstoneProbDist
        gamma = st.slider("Gamma", 0.001, 1.0, 0.1)
        lids = LidstoneProbDist(fd, gamma)
        st.metric(f"P('{query_word}')", f"{lids.prob(query_word):.6f}")
        runnable_code("""from nltk.probability import LidstoneProbDist, FreqDist
fd = FreqDist(['cat', 'dog', 'cat', 'bird'])
# gamma=1.0 is Laplace, gamma→0 is MLE
lidstone = LidstoneProbDist(fd, 0.1)
print(lidstone.prob('cat'))
print(lidstone.prob('fish'))  # unseen but nonzero""")

    with tabs[3]:
        st.subheader("KneserNeyProbDist")
        from nltk.probability import KneserNeyProbDist
        discount = st.slider("Discount", 0.1, 0.99, 0.75)
        try:
            kn = KneserNeyProbDist(fd, discount=discount)
            st.metric(f"P('{query_word}')", f"{kn.prob(query_word):.6f}")
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.probability import KneserNeyProbDist, FreqDist
fd = FreqDist(['cat', 'dog', 'cat', 'bird', 'cat', 'fish'])
kn = KneserNeyProbDist(fd, discount=0.75)
print(kn.prob('cat'))
print(kn.prob('dog'))""")

    with tabs[4]:
        st.subheader("WittenBellProbDist")
        from nltk.probability import WittenBellProbDist
        try:
            wb = WittenBellProbDist(fd, bins=len(fd) + 100)
            st.metric(f"P('{query_word}')", f"{wb.prob(query_word):.6f}")
        except Exception as e:
            st.error(str(e))
        runnable_code("""from nltk.probability import WittenBellProbDist, FreqDist
fd = FreqDist(['cat', 'dog', 'cat', 'bird'])
wb = WittenBellProbDist(fd, bins=1000)
print(wb.prob('cat'))
print(wb.prob('unseen_word'))""")

    with tabs[5]:
        st.subheader("ConditionalProbDist")
        runnable_code("""from nltk.probability import (
    ConditionalFreqDist, ConditionalProbDist, MLEProbDist
)
from nltk.corpus import brown
# Build a conditional freq dist: P(word | genre)
cfd = ConditionalFreqDist(
    (genre, word.lower())
    for genre in ['news', 'fiction']
    for word in brown.words(categories=genre)
)
# Wrap with a probability estimator
cpd = ConditionalProbDist(cfd, MLEProbDist)
print(cpd['news'].prob('the'))
print(cpd['fiction'].prob('said'))""")
        from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist
        from nltk.corpus import brown
        genres = ["news", "fiction"]
        cfd = ConditionalFreqDist(
            (genre, word.lower())
            for genre in genres
            for word in brown.words(categories=genre)
        )
        cpd = ConditionalProbDist(cfd, MLEProbDist)
        words_check = ["the", "said", "freedom", "was", "a"]
        rows = []
        for w in words_check:
            rows.append({"Word": w, "P(word|news)": f"{cpd['news'].prob(w):.5f}", "P(word|fiction)": f"{cpd['fiction'].prob(w):.5f}"})
        st.dataframe(pd.DataFrame(rows))


# ════════════════════════════════════════════════════════════════════════════
# 17. NOT AVAILABLE IN STREAMLIT
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[16]:
    st.title("❌ APIs Not Available in Streamlit")
    st.markdown("These NLTK modules cannot run inside a Streamlit app, but here's **why** and **how to use them** in a regular Python script.")

    sections = {
        "nltk.app (Tkinter GUI apps)": {
            "reason": "All `nltk.app.*` modules launch **Tkinter** desktop windows (`tk.Tk()`). Streamlit runs in a browser — there is no display server to render native GUI windows.",
            "modules": [
                "nltk.app.chartparser_app",
                "nltk.app.chunkparser_app",
                "nltk.app.collocations_app",
                "nltk.app.concordance_app",
                "nltk.app.nemo_app",
                "nltk.app.rdparser_app",
                "nltk.app.srparser_app",
                "nltk.app.wordfreq_app",
                "nltk.app.wordnet_app",
            ],
            "code": """# Run from a regular terminal (NOT inside Streamlit):
import nltk

nltk.app.chartparser()    # Interactive chart parser GUI
nltk.app.chunkparser()    # Chunker GUI
nltk.app.collocations()   # Collocations browser
nltk.app.concordance()    # Concordance browser
nltk.app.rdparser()       # Recursive descent parser GUI
nltk.app.srparser()       # Shift-reduce parser GUI
nltk.app.wordfreq()       # Word frequency browser
nltk.app.wordnet()        # WordNet browser GUI""",
        },
        "nltk.draw (Tkinter drawing)": {
            "reason": "`nltk.draw` uses **Tkinter canvas** widgets to render parse trees, CFG diagrams, and tables. Same issue — requires a native display.",
            "modules": [
                "nltk.draw.cfg — CFGEditor, CFGDemo",
                "nltk.draw.dispersion — dispersion_plot (Tkinter version)",
                "nltk.draw.table — MultiListbox, Table",
                "nltk.draw.tree — TreeWidget, TreeView, draw_trees()",
                "nltk.draw.util — CanvasFrame, CanvasWidget, ...",
            ],
            "code": """# Run from a regular terminal:
from nltk.draw.tree import draw_trees
from nltk.tree import Tree

t = Tree.fromstring("(S (NP (Det the) (N cat)) (VP (V sat)))")
draw_trees(t)  # Opens a Tkinter window with the parse tree

# Or use tree.draw() directly:
t.draw()""",
        },
        "Stanford NLP / CoreNLP (Java required)": {
            "reason": "Stanford parsers, NER taggers, and CoreNLP require a **Java runtime** and large `.jar` files downloaded separately. They cannot run in a sandboxed web environment.",
            "modules": [
                "nltk.parse.stanford — StanfordParser, StanfordDependencyParser",
                "nltk.parse.corenlp — CoreNLPParser, CoreNLPDependencyParser, CoreNLPServer",
                "nltk.tag.stanford — StanfordNERTagger, StanfordPOSTagger",
                "nltk.tokenize.stanford — StanfordTokenizer",
            ],
            "code": """# Requires: Java 8+, Stanford CoreNLP jar files
import os
os.environ['STANFORD_PARSER'] = '/path/to/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/path/to/stanford-parser-models.jar'

from nltk.parse.stanford import StanfordParser
parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
list(parser.raw_parse("The quick brown fox jumps"))""",
        },
        "MaltParser (Java required)": {
            "reason": "MaltParser is a Java-based dependency parser. NLTK wraps it but it needs a Java executable and model files on disk.",
            "modules": [
                "nltk.parse.malt — MaltParser",
            ],
            "code": """# Requires: Java, maltparser.jar, and a trained model file
from nltk.parse.malt import MaltParser
mp = MaltParser(working_dir='/tmp', mco='engmalt.linear-1.7')
print(list(mp.parse("I saw the man with a telescope".split())))""",
        },
        "Prover9 / Mace4 / Tableau (External binaries)": {
            "reason": "These are **formal logic theorem provers** that require external binary executables (Prover9, Mace4) compiled for your OS.",
            "modules": [
                "nltk.inference.prover9 — Prover9, Prover9Command",
                "nltk.inference.mace — Mace, MaceCommand",
                "nltk.inference.tableau — TableauProver",
                "nltk.inference.resolution — ResolutionProver",
            ],
            "code": """# Requires: Prover9 binary installed at /usr/local/bin/prover9
from nltk.inference.prover9 import Prover9
from nltk.sem.logic import Expression
read_expr = Expression.fromstring

goal    = read_expr('mortal(socrates)')
premise = read_expr('all x.(man(x) -> mortal(x))')
man     = read_expr('man(socrates)')

prover = Prover9()
print(prover.prove(goal, [premise, man]))  # True""",
        },
        "nltk.twitter (Twitter API credentials)": {
            "reason": "The Twitter/X API now requires **paid API credentials** (Bearer token, API key/secret). Without them, all Twitter NLTK functions fail with authentication errors.",
            "modules": [
                "nltk.twitter.twitterclient — Twitter, Streamer",
                "nltk.twitter.common — json2csv, extract_fields",
                "nltk.twitter.api — TweetHandlerI",
            ],
            "code": """# Requires a Twitter Developer Account and approved API access
from nltk.twitter import Twitter
from nltk.twitter.util import credsfromfile

oauth = credsfromfile()  # reads from ~/.twitter-creds
tw = Twitter(oauth)
tw.search(keywords="NLP", limit=10)  # fetches 10 tweets about NLP""",
        },
        "BLLIP Parser (C++ binary)": {
            "reason": "BllipParser is a statistical parser implemented in C++ that requires the `bllipparser` package and pre-trained models. Incompatible with Streamlit's sandboxed environment.",
            "modules": [
                "nltk.parse.bllip — BllipParser",
            ],
            "code": """# pip install bllipparser  (then download models)
from nltk.parse.bllip import BllipParser
# model_dir from: pip install bllipparser; python -c "import bllipparser; bllipparser.download_model()"
parser = BllipParser.from_unified_model_dir('/path/to/model')
for tree in parser.parse("The cat sat on the mat"):
    print(tree)""",
        },
        "SENNA Tagger (external binary)": {
            "reason": "SENNA is a C-based NLP pipeline. NLTK wraps it but it requires downloading the SENNA binary separately.",
            "modules": [
                "nltk.tag.senna — SennaTagger, SennaChunkTagger, SennaNERTagger",
                "nltk.classify.senna — Senna",
            ],
            "code": """# Download SENNA from: http://ronan.collobert.com/senna/
from nltk.tag.senna import SennaTagger
tagger = SennaTagger('/path/to/senna/')
print(tagger.tag("the cat sat on the mat".split()))""",
        },
        "Weka / TADM / MEGAM (External Java/binary)": {
            "reason": "These are external machine learning tools that need their own executables or JARs and cannot be bundled into a Streamlit app.",
            "modules": [
                "nltk.classify.weka — WekaClassifier",
                "nltk.classify.tadm — TadmMaxentClassifier",
                "nltk.classify.megam — call_megam()",
                "nltk.classify.maxent — TadmEventMaxentFeatureEncoding",
            ],
            "code": """# Weka requires weka.jar
from nltk.classify.weka import WekaClassifier, config_weka
config_weka('/path/to/weka.jar')
# ... train and classify

# TADM (Toolkit for Advanced Discriminative Modeling)
from nltk.classify.tadm import TadmMaxentClassifier
# Requires tadm binary: https://tadm.sourceforge.net/

# MEGAM (Maximum Entropy)
from nltk.classify.megam import config_megam
config_megam('/path/to/megam')""",
        },
        "nltk.sem.boxer / DRT demos with GUI": {
            "reason": "Boxer is an external semantic parser (requires Prolog + C&C parser). `DrsDrawer` and `DrtGlueDemo` use Tkinter for visualization.",
            "modules": [
                "nltk.sem.boxer — Boxer, BoxerDrsParser",
                "nltk.sem.drt_glue_demo — DrtGlueDemo (Tkinter)",
            ],
            "code": """# Boxer requires: candc + boxer binaries
from nltk.sem.boxer import Boxer
boxer = Boxer('/path/to/candc/', '/path/to/boxer/')
result = boxer.interpret("A dog sees every cat")
print(result)

# DRT GUI (Tkinter)
from nltk.sem.drt_glue_demo import demo
demo()  # Opens Tkinter window""",
        },
    }

    for title, info in sections.items():
        with st.expander(f"❌ {title}", expanded=False):
            st.error(f"**Why not available:** {info['reason']}")
            st.markdown("**Affected modules:**")
            for m in info["modules"]:
                st.markdown(f"- `{m}`")
            st.markdown("**How to use it outside Streamlit:**")
            st.code(info["code"], language="python")

    st.divider()
    st.subheader("Summary Table")
    summary = [
        {"Module", "Reason", "Workaround"},
    ]
    rows = [
        {"Module": "nltk.app.*", "Reason": "Tkinter GUI", "Workaround": "Run in terminal"},
        {"Module": "nltk.draw.*", "Reason": "Tkinter canvas", "Workaround": "Use matplotlib / SVG"},
        {"Module": "Stanford NLP", "Reason": "Needs Java JDK", "Workaround": "Use spaCy or stanza"},
        {"Module": "MaltParser", "Reason": "Needs Java + model", "Workaround": "Use spaCy dependency parser"},
        {"Module": "Prover9/Mace4", "Reason": "External C binaries", "Workaround": "Install locally"},
        {"Module": "nltk.twitter", "Reason": "Paid API credentials", "Workaround": "Get Twitter developer account"},
        {"Module": "BllipParser", "Reason": "C++ binary required", "Workaround": "Use Stanford / spaCy"},
        {"Module": "SENNA", "Reason": "External C binary", "Workaround": "Install locally"},
        {"Module": "Weka/TADM/MEGAM", "Reason": "External JARs/binaries", "Workaround": "Use nltk.classify.SklearnClassifier"},
        {"Module": "Boxer/DRT GUI", "Reason": "Prolog + Tkinter", "Workaround": "Install C&C + Boxer separately"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
