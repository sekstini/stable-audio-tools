set -x

# Set up the data directory
mkdir -p data

# Download failed speech restoration examples
wget -O - https://www.openslr.org/resources/141/libritts_r_failed_speech_restoration_examples.tar.gz | tar -xz -C data

# Download the LibriTTS dataset (this will take a while)
wget -P data https://www.openslr.org/resources/141/dev_clean.tar.gz
wget -P data https://www.openslr.org/resources/141/dev_other.tar.gz
wget -P data https://www.openslr.org/resources/141/train_clean_100.tar.gz
wget -P data https://www.openslr.org/resources/141/train_clean_360.tar.gz
wget -P data https://www.openslr.org/resources/141/train_other_500.tar.gz