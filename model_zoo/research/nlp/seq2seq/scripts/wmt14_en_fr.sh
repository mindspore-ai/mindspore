#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

OUTPUT_DIR=${1:-"data_new/wmt14_en_fr"}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

mkdir -p $OUTPUT_DIR_DATA

echo "Downloading Europarl v7. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz \
  http://www.statmt.org/europarl/v7/fr-en.tgz

echo "Downloading Common Crawl corpus. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/common-crawl.tgz \
  http://www.statmt.org/wmt14/training-parallel-commoncrawl.tgz

echo "Downloading News Commentary v11. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/nc-v9.tgz \
  http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz

echo "Downloading test sets"
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz \
  http://www.statmt.org/wmt14/test-full.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-fr-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-fr-en"
mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
mkdir -p "${OUTPUT_DIR_DATA}/nc-v9"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v9.tgz" -C "${OUTPUT_DIR_DATA}/nc-v9"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-fr-en/europarl-v7.fr-en.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.fr-en.en" \
  "${OUTPUT_DIR_DATA}/nc-v9/training-parallel-nc-v9/news-commentary-v9.fr-en.en" \
  > "${OUTPUT_DIR}/train.en"
wc -l "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.fr-en.fr" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.fr-en.fr" \
  "${OUTPUT_DIR_DATA}/nc-v9/training-parallel-nc-v9/news-commentary-v9.fr-en.fr" \
  > "${OUTPUT_DIR}/train.fr"
wc -l "${OUTPUT_DIR}/train.fr"

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
  cd ${OUTPUT_DIR}/mosesdecoder
  git reset --hard 8c5eaa1a122236bbf927bde4ec610906fea599e6
  cd -
fi

# Convert newstest2014 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2014-fren-src.fr.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2014.fr
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2014-fren-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2014.en

cp ${OUTPUT_DIR_DATA}/test/test/newstest2014.fr ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test/newstest2014.en ${OUTPUT_DIR}

# Tokenize data
for f in ${OUTPUT_DIR}/*.fr; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l fr -threads 8 < $f > ${f%.*}.tok.fr
done

for f in ${OUTPUT_DIR}/*.en; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean all corpora
for f in ${OUTPUT_DIR}/*.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase fr en "${fbase}.clean" 1 80
done

# Filter datasets
python filter_dataset.py \
   -f1 ${OUTPUT_DIR}/train.tok.clean.en \
   -f2 ${OUTPUT_DIR}/train.tok.clean.fr

# Learn Shared BPE
merge_ops=32000
echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
cat "${OUTPUT_DIR}/train.tok.clean.fr" "${OUTPUT_DIR}/train.tok.clean.en" | \
    subword-nmt learn-bpe -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
for lang in en fr; do
  for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
    outfile="${f%.*}.bpe.${merge_ops}.${lang}"
    subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
    echo ${outfile}
  done
done
# Create vocabulary file for BPE
cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.en" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.fr" | \
subword-nmt get-vocab | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
