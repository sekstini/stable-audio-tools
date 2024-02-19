## MLS
### 1. Extract transcriptions
```bash
cd data/mls
tar xf mls_english_opus.tar.gz --wildcards "*.txt"
cd ../.. # get back to root
```

This will take an eternity, but when done you're left with:
```
data/mls/mls_english_opus
├── dev
│   ├── segments.txt
│   └── transcripts.txt
├── metainfo.txt
├── test
│   ├── segments.txt
│   └── transcripts.txt
└── train
    ├── limited_supervision
    ├── segments.txt
    └── transcripts.txt
```

### 2. Run preprocessing script
```python
python scripts/datasets/preprocess_mts.py data/mls/mls_english_opus.tar.gz
```
