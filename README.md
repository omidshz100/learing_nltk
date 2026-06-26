# NLTK Interactive Explorer

An interactive Streamlit app covering all major NLTK APIs — run code live, explore examples, and learn NLP step by step.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng \
  maxent_ne_chunker_tab words stopwords vader_lexicon wordnet omw-1.4 \
  brown gutenberg movie_reviews names universal_tagset conll2000 treebank inaugural webtext
streamlit run app.py
```

Then open http://localhost:8501

---

## Learning Roadmap

### 🟢 Beginner — Start Here

| # | Section |
|---|---------|
| 1 | Tokenization |
| 2 | Stemming & Lemmatization |
| 3 | POS Tagging |
| 4 | NER & Chunking |
| 16 | Frequency Distribution |
| 10 | Sentiment Analysis |

### 🟡 Intermediate — Learn Next

| # | Section |
|---|---------|
| 7 | Classification (Naive Bayes) |
| 15 | Corpus Access |
| 18 | Collocations |
| 19 | Text Analysis / TF-IDF |
| 14 | Metrics & Evaluation |

### 🔴 Advanced — When Ready

| # | Section |
|---|---------|
| 5 | Parsing |
| 6 | Grammar |
| 8 | Clustering |
| 9 | Language Models |
| 11 | Semantics |
| 12 | Inference |
| 13 | Translation |

---

## All 24 NLTK Categories

### 1. Text Preprocessing & Tokenization — `nltk.tokenize`

| Group | APIs |
|---|---|
| Sentence/Word | `sent_tokenize()`, `word_tokenize()`, `PunktSentenceTokenizer`, `NLTKWordTokenizer` |
| Specialized | `TweetTokenizer`, `RegexpTokenizer`, `MWETokenizer`, `TreebankWordTokenizer`, `NISTTokenizer`, `ToktokTokenizer` |
| Syllable | `LegalitySyllableTokenizer`, `SyllableTokenizer` |
| Utilities | `align_tokens()`, `regexp_span_tokenize()`, `TreebankWordDetokenizer` |

### 2. Stemming & Lemmatization — `nltk.stem`

| Group | APIs |
|---|---|
| Stemmers | `PorterStemmer`, `LancasterStemmer`, `SnowballStemmer`, `ISRIStemmer`, `ARLSTem`, `ARLSTem2`, `Cistem`, `RSLPStemmer`, `RegexpStemmer` |
| Multilingual Snowball | `ArabicStemmer`, `FrenchStemmer`, `GermanStemmer`, `SpanishStemmer`, `ItalianStemmer`, `PortugueseStemmer` (+ 9 more) |
| Lemmatizer | `WordNetLemmatizer` |

### 3. Part-of-Speech Tagging — `nltk.tag`

| Group | APIs |
|---|---|
| Taggers | `PerceptronTagger`, `HiddenMarkovModelTagger`, `BrillTagger`, `CRFTagger`, `TnT` |
| Sequential | `UnigramTagger`, `BigramTagger`, `TrigramTagger`, `AffixTagger`, `DefaultTagger`, `RegexpTagger` |
| External ⚠️ | `StanfordPOSTagger`, `HunposTagger`, `SennaTagger` |
| Utilities | `pos_tag()`, `pos_tag_sents()`, `map_tag()`, `str2tuple()`, `tuple2str()` |

### 4. Chunking & Named Entity Recognition — `nltk.chunk`

| Group | APIs |
|---|---|
| Chunkers | `RegexpChunkParser`, `RegexpParser`, `NEChunkParser` |
| Chunk Rules | `ChunkRule`, `MergeRule`, `SplitRule`, `ExpandLeftRule`, `ExpandRightRule`, `UnChunkRule` |
| NE Chunking | `ne_chunk()`, `ne_chunk_sents()`, `Maxent_NE_Chunker` |
| Utilities | `conllstr2tree()`, `tree2conllstr()`, `conlltags2tree()`, `ChunkScore` |

### 5. Parsing — `nltk.parse`

| Group | APIs |
|---|---|
| Chart Parsers | `ChartParser`, `BottomUpChartParser`, `TopDownChartParser`, `EarleyChartParser`, `FeatureChartParser` |
| Probabilistic | `ViterbiParser`, `InsideChartParser`, `BottomUpProbabilisticChartParser` |
| Dependency | `ProjectiveDependencyParser`, `NonprojectiveDependencyParser`, `MaltParser` ⚠️, `StanfordDependencyParser` ⚠️ |
| Other | `RecursiveDescentParser`, `ShiftReduceParser`, `BllipParser` ⚠️, `CoreNLPParser` ⚠️ |
| Utilities | `generate()` |

### 6. Grammar & Formal Languages — `nltk.grammar`, `nltk.ccg`

| Group | APIs |
|---|---|
| Grammar Types | `CFG`, `PCFG`, `DependencyGrammar`, `ProbabilisticDependencyGrammar` |
| Productions | `Production`, `ProbabilisticProduction`, `Nonterminal` |
| CCG | `CCGChartParser`, `CCGLexicon`, `FunctionalCategory`, `PrimitiveCategory` |
| CCG Combinators | `ForwardCombinator`, `BackwardCombinator`, `UndirectedComposition` |
| Utilities | `induce_pcfg()`, `read_grammar()`, `nonterminals()` |

### 7. Classification — `nltk.classify`

| Group | APIs |
|---|---|
| Classifiers | `NaiveBayesClassifier`, `DecisionTreeClassifier`, `MaxentClassifier`, `PositiveNaiveBayesClassifier` |
| External ⚠️ | `SklearnClassifier`, `SvmClassifier`, `WekaClassifier`, `TadmMaxentClassifier` |
| Utilities | `accuracy()`, `apply_features()`, `log_likelihood()`, `attested_labels()` |

### 8. Clustering — `nltk.cluster`

| Group | APIs |
|---|---|
| Algorithms | `KMeansClusterer`, `EMClusterer`, `GAAClusterer` |
| Distance Metrics | `cosine_distance()`, `euclidean_distance()` |
| Base Classes | `VectorSpaceClusterer`, `ClusterI` |

### 9. Language Models — `nltk.lm`

| Group | APIs |
|---|---|
| Models | `MLE`, `Laplace`, `Lidstone`, `StupidBackoff`, `KneserNeyInterpolated`, `WittenBellInterpolated`, `AbsoluteDiscountingInterpolated` |
| Smoothing | `KneserNey`, `WittenBell`, `AbsoluteDiscounting` |
| Data Prep | `padded_everygram_pipeline()`, `padded_everygrams()`, `flatten()` |
| Vocab & Count | `Vocabulary`, `NgramCounter` |

### 10. Sentiment Analysis — `nltk.sentiment`

| Group | APIs |
|---|---|
| Analyzers | `SentimentIntensityAnalyzer` (VADER), `SentimentAnalyzer` |
| Feature Extraction | `extract_unigram_feats()`, `extract_bigram_feats()`, `mark_negation()` |
| Demos | `demo_movie_reviews()`, `demo_vader_tweets()`, `demo_liu_hu_lexicon()` |

### 11. Semantics & Logic — `nltk.sem`

| Group | APIs |
|---|---|
| Logic | `LogicParser`, `LambdaExpression`, `AllExpression`, `ExistsExpression`, `ApplicationExpression` |
| DRT | `DrtParser`, `DRS`, `DrtExpression`, `resolve_anaphora()` |
| Model Evaluation | `Model`, `Valuation`, `Assignment`, `evaluate_sents()` |
| Relation Extraction | `extract_rels()`, `tree2semi_rel()` |
| Glue Semantics | `Glue`, `GlueFormula`, `DrtGlue` |
| Boxer ⚠️ | `Boxer`, `BoxerDrs`, `BoxerDrsParser` |

### 12. Inference & Theorem Proving — `nltk.inference`

| Group | APIs |
|---|---|
| Provers ⚠️ | `Prover9`, `TableauProver`, `ResolutionProver` |
| Model Builders | `Mace`, `ParallelProverBuilder` |
| Discourse | `DiscourseTester`, `CfgReadingCommand`, `DrtGlueReadingCommand` |
| Nonmonotonic | `ClosedWorldProver`, `UniqueNamesProver`, `ClosedDomainProver` |

### 13. Machine Translation & Evaluation — `nltk.translate`

| Group | APIs |
|---|---|
| IBM Alignment | `IBMModel1`, `IBMModel2`, `IBMModel3`, `IBMModel4`, `IBMModel5` |
| Scores | `corpus_bleu()`, `sentence_bleu()`, `corpus_chrf()`, `corpus_nist()`, `meteor_score()`, `corpus_ribes()` |
| Phrase-based | `PhraseTable`, `phrase_extraction()`, `StackDecoder` |
| Alignment Utils | `AlignedSent`, `Alignment`, `grow_diag_final_and()` |

### 14. Metrics & Evaluation — `nltk.metrics`

| Group | APIs |
|---|---|
| Scoring | `accuracy()`, `precision()`, `recall()`, `f_measure()`, `log_likelihood()` |
| Distance | `edit_distance()`, `jaccard_distance()`, `jaro_similarity()`, `jaro_winkler_similarity()`, `masi_distance()`, `binary_distance()` |
| Agreement | `AnnotationTask` |
| Segmentation | `pk()`, `windowdiff()`, `ghd()` |
| Association | `BigramAssocMeasures`, `TrigramAssocMeasures`, `NgramAssocMeasures`, `QuadgramAssocMeasures` |
| Confusion Matrix | `ConfusionMatrix` |

### 15. Corpus Access — `nltk.corpus`

| Group | APIs |
|---|---|
| Loader | `LazyCorpusLoader` |
| Built-in Corpora | `brown`, `gutenberg`, `inaugural`, `movie_reviews`, `treebank`, `conll2000`, `wordnet`, `names`, `stopwords`, `webtext`, `reuters`, `nps_chat` |

### 16. Probability & Statistics — `nltk.probability`

| Group | APIs |
|---|---|
| Frequency | `FreqDist`, `ConditionalFreqDist` |
| Probability Dists | `MLEProbDist`, `LaplaceProbDist`, `LidstoneProbDist`, `KneserNeyProbDist`, `WittenBellProbDist`, `SimpleGoodTuringProbDist`, `HeldoutProbDist`, `CrossValidationProbDist` |
| Conditional | `ConditionalProbDist`, `DictionaryConditionalProbDist` |
| Utilities | `entropy()`, `log_likelihood()`, `add_logs()`, `sum_logs()` |

### 17. Tree Structures — `nltk.tree`

| Group | APIs |
|---|---|
| Tree Types | `Tree`, `ProbabilisticTree`, `ImmutableTree`, `ParentedTree`, `MultiParentedTree` |
| Transforms | `chomsky_normal_form()`, `collapse_unary()`, `un_chomsky_normal_form()` |
| Pretty Printing | `TreePrettyPrinter` |
| Parsing | `bracket_parse()`, `sinica_parse()` |

### 18. Collocations — `nltk.collocations`

`BigramCollocationFinder`, `TrigramCollocationFinder`, `QuadgramCollocationFinder`

### 19. Text Analysis — `nltk.text`

| Group | APIs |
|---|---|
| Text Exploration | `Text` — `concordance()`, `similar()`, `dispersion_plot()`, `collocations()`, `generate()` |
| Indexing | `ConcordanceIndex`, `ContextIndex`, `TokenSearcher` |
| TF-IDF | `TextCollection` — `tf()`, `idf()`, `tf_idf()` |

### 20. Chat / Dialogue Systems — `nltk.chat`

Built-in chatbots: `eliza_chat()`, `iesha_chat()`, `rude_chat()`, `zen_chat()`, `suntsu_chat()`  
Base class: `Chat`

### 21. Data & Downloader — `nltk.data`, `nltk.downloader`

| Group | APIs |
|---|---|
| Downloader | `Downloader`, `DownloaderGUI`, `DownloaderShell`, `download()` |
| Data Loading | `load()`, `find()`, `retrieve()`, `clear_cache()` |
| Path Pointers | `FileSystemPathPointer`, `GzipFileSystemPathPointer` |

### 22. Drawing & Visualization ❌ — `nltk.draw`

> **Not available in web apps** — requires Tkinter desktop GUI.

`TreeWidget`, `TreeView`, `draw_trees()`, `CFGEditor`, `CFGDemo`, `dispersion_plot()`, `CanvasFrame`, `CanvasWidget`, `Table`

### 23. Applications (GUI Tools) ❌ — `nltk.app`

> **Not available in web apps** — requires Tkinter desktop GUI.

`chartparser_app`, `chunkparser_app`, `collocations_app`, `concordance_app`, `rdparser_app`, `srparser_app`, `wordfreq_app`, `wordnet_app`

### 24. Utilities & Misc — `nltk.util`, `nltk.misc`, `nltk.collections`

| Group | APIs |
|---|---|
| N-grams | `bigrams()`, `trigrams()`, `ngrams()`, `everygrams()`, `skipgrams()` |
| Data Structures | `Trie`, `LazyMap`, `LazyConcatenation`, `Index` |
| Graph Utilities | `breadth_first()`, `transitive_closure()`, `unweighted_minimum_spanning_tree()` |
| Misc | `MinimalSet`, `generate_chomsky()`, `babelize_shell()` |

---

## Summary

| Category | Package | Primary Use |
|---|---|---|
| Tokenization | `nltk.tokenize` | Split text into tokens |
| Stemming/Lemma | `nltk.stem` | Reduce words to roots |
| POS Tagging | `nltk.tag` | Label grammatical roles |
| Chunking/NER | `nltk.chunk` | Extract entities/phrases |
| Parsing | `nltk.parse` | Syntactic tree building |
| Grammar | `nltk.grammar`, `nltk.ccg` | Formal grammar handling |
| Classification | `nltk.classify` | Text categorization |
| Clustering | `nltk.cluster` | Unsupervised grouping |
| Language Models | `nltk.lm` | N-gram probability |
| Sentiment | `nltk.sentiment` | Opinion/emotion detection |
| Semantics | `nltk.sem` | Logical meaning repr. |
| Inference | `nltk.inference` | Theorem proving |
| Translation | `nltk.translate` | MT alignment & scoring |
| Metrics | `nltk.metrics` | Evaluation & distance |
| Corpus | `nltk.corpus` | Access built-in datasets |
| Probability | `nltk.probability` | Freq & probability dists |
| Trees | `nltk.tree` | Parse tree manipulation |
| Collocations | `nltk.collocations` | Word co-occurrence |
| Text Analysis | `nltk.text` | Concordance, TF-IDF |
| Chat | `nltk.chat` | Rule-based chatbots |
| Data/Download | `nltk.data`, `nltk.downloader` | Load/download resources |
| Visualization ❌ | `nltk.draw` | GUI tree/grammar display |
| Applications ❌ | `nltk.app` | Interactive GUI demos |
| Utilities | `nltk.util`, `nltk.misc` | N-grams, graphs, helpers |

> ❌ Not available in web environments (requires Tkinter desktop GUI)  
> ⚠️ Requires external tool (Java, C++ binary, or API credentials)
