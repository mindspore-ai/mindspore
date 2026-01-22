# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Third-party modules."""

# List common third-party modules for JIT Fallback.
third_party_modules_whitelist = (
    # Python built-in modules.
    "datetime", "re", "difflib", "math", "cmath", "random",
    # Machine Learning.
    "ml_metrics", "nupic", "sklearn", "pyspark", "vowpal_porpoise", "xgboost", "tensorflow", "torch", "jax",
    # Natual Language Processing.
    "gensim", "jieba", "langid", "nltk", "pattern", "polyglot", "snownlp", "spacy", "textblob", "quepy", "yalign",
    "spammy", "genius", "konlpy", "nut", "rosetta", "bllipparser", "pynlpl", "ucto", "frog", "zpar", "colibricore",
    "StanfordDependencies", "distance", "thefuzz", "jellyfish", "editdistance", "textacy", "pycorenlp", "cltk",
    "rasa_nlu", "drqa", "dedupe",
    # Text Processing.
    "chardet", "ftfy", "Levenshtein", "pangu", "pyfiglet", "pypinyin", "shortuuid", "unidecode", "uniout",
    "xpinyin", "slugify", "phonenumbers", "ply", "pygments", "pyparsing", "nameparser", "user_agents", "sqlparse",
    # Web Content Extracting.
    "haul", "html2text", "lassie", "micawber", "newspaper", "goose", "readability", "requests_html", "sanitize",
    "sumy", "textract", "toapi",
    # Web Crawling.
    "cola", "demiurge", "feedparser", "grab", "mechanicalsoup", "pyspider", "robobrowser", "scrapy",
    # Algorithms and Design Patterns.
    "algorithms", "pypattyrn", "sortedcontainers", "scoop",
    # Cryptography.
    "cryptography", "hashids", "paramiko", "passlib", "nacl",
    # Data Analysis (without plotting).
    "numpy", "scipy", "blaze", "pandas", "numba", "pymc", "zipline", "pydy", "sympy", "statsmodels", "astropy",
    "vincent", "pygal", "pycascading", "emcee", "windml", "vispy", "Dora", "ruffus", "sompy", "somoclu", "hdbscan",
    # Computer Vision.
    "PIL", "skimage", "SimpleCV", "PCV", "face_recognition",
    # General-Purpose Machine Learning.
    "cntk", "auto_ml", "xgboost", "featureforge", "scikits", "metric_learn", "simpleai", "bigml", "pylearn2", "keras",
    "lasagne", "hebel", "topik", "pybrain", "surprise", "recsys", "thinkbayes", "nilearn", "neuropredict", "pyevolve",
    "pyhsmm", "mrjob", "neurolab", "pebl", "yahmm", "timbl", "deap", "deeppy", "mlxtend", "neon", "optunity", "topt",
    "pgmpy", "milk", "rep", "rgf", "FukuML", "stacked_generalization", "modAL", "cogitare", "gym",
)
