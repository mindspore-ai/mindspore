/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/text/ir/kernels/text_ir.h"
#include "minddata/dataset/text/kernels/wordpiece_tokenizer_op.h"
#include "minddata/dataset/text/sentence_piece_vocab.h"
#include "minddata/dataset/text/vocab.h"

namespace mindspore {
namespace dataset {
#ifdef ENABLE_ICU4C

PYBIND_REGISTER(
  BasicTokenizerOperation, 1, ([](const py::module *m) {
    (void)py::class_<text::BasicTokenizerOperation, TensorOperation, std::shared_ptr<text::BasicTokenizerOperation>>(
      *m, "BasicTokenizerOperation")
      .def(py::init([](bool lower_case, bool keep_whitespace, const NormalizeForm normalize_form,
                       bool preserve_unused_token, bool with_offsets) {
        auto basic_tokenizer = std::make_shared<text::BasicTokenizerOperation>(
          lower_case, keep_whitespace, normalize_form, preserve_unused_token, with_offsets);
        THROW_IF_ERROR(basic_tokenizer->ValidateParams());
        return basic_tokenizer;
      }));
  }));

PYBIND_REGISTER(
  BertTokenizerOperation, 1, ([](const py::module *m) {
    (void)py::class_<text::BertTokenizerOperation, TensorOperation, std::shared_ptr<text::BertTokenizerOperation>>(
      *m, "BertTokenizerOperation")
      .def(py::init([](const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator,
                       int32_t max_bytes_per_token, const std::string &unknown_token, bool lower_case,
                       bool keep_whitespace, const NormalizeForm normalize_form, bool preserve_unused_token,
                       bool with_offsets) {
        auto bert_tokenizer = std::make_shared<text::BertTokenizerOperation>(
          vocab, suffix_indicator, max_bytes_per_token, unknown_token, lower_case, keep_whitespace, normalize_form,
          preserve_unused_token, with_offsets);
        THROW_IF_ERROR(bert_tokenizer->ValidateParams());
        return bert_tokenizer;
      }));
  }));

PYBIND_REGISTER(CaseFoldOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::CaseFoldOperation, TensorOperation, std::shared_ptr<text::CaseFoldOperation>>(
                    *m, "CaseFoldOperation")
                    .def(py::init([]() {
                      auto case_fold = std::make_shared<text::CaseFoldOperation>();
                      THROW_IF_ERROR(case_fold->ValidateParams());
                      return case_fold;
                    }));
                }));

PYBIND_REGISTER(
  NormalizeUTF8Operation, 1, ([](const py::module *m) {
    (void)py::class_<text::NormalizeUTF8Operation, TensorOperation, std::shared_ptr<text::NormalizeUTF8Operation>>(
      *m, "NormalizeUTF8Operation")
      .def(py::init([](NormalizeForm normalize_form) {
        auto normalize_utf8 = std::make_shared<text::NormalizeUTF8Operation>(normalize_form);
        THROW_IF_ERROR(normalize_utf8->ValidateParams());
        return normalize_utf8;
      }));
  }));

PYBIND_REGISTER(
  RegexReplaceOperation, 1, ([](const py::module *m) {
    (void)py::class_<text::RegexReplaceOperation, TensorOperation, std::shared_ptr<text::RegexReplaceOperation>>(
      *m, "RegexReplaceOperation")
      .def(py::init([](std::string pattern, std::string replace, bool replace_all) {
        auto regex_replace = std::make_shared<text::RegexReplaceOperation>(pattern, replace, replace_all);
        THROW_IF_ERROR(regex_replace->ValidateParams());
        return regex_replace;
      }));
  }));

PYBIND_REGISTER(
  RegexTokenizerOperation, 1, ([](const py::module *m) {
    (void)py::class_<text::RegexTokenizerOperation, TensorOperation, std::shared_ptr<text::RegexTokenizerOperation>>(
      *m, "RegexTokenizerOperation")
      .def(
        py::init([](const std::string &delim_pattern, const std::string &keep_delim_pattern, const bool &with_offsets) {
          auto regex_tokenizer =
            std::make_shared<text::RegexTokenizerOperation>(delim_pattern, keep_delim_pattern, with_offsets);
          THROW_IF_ERROR(regex_tokenizer->ValidateParams());
          return regex_tokenizer;
        }));
  }));

PYBIND_REGISTER(UnicodeScriptTokenizerOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::UnicodeScriptTokenizerOperation, TensorOperation,
                                   std::shared_ptr<text::UnicodeScriptTokenizerOperation>>(
                    *m, "UnicodeScriptTokenizerOperation")
                    .def(py::init([](bool keep_whitespace, bool with_offsets) {
                      auto unicode_script_tokenizer =
                        std::make_shared<text::UnicodeScriptTokenizerOperation>(keep_whitespace, with_offsets);
                      THROW_IF_ERROR(unicode_script_tokenizer->ValidateParams());
                      return unicode_script_tokenizer;
                    }));
                }));

PYBIND_REGISTER(WhitespaceTokenizerOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::WhitespaceTokenizerOperation, TensorOperation,
                                   std::shared_ptr<text::WhitespaceTokenizerOperation>>(*m,
                                                                                        "WhitespaceTokenizerOperation")
                    .def(py::init([](bool with_offsets) {
                      auto whitespace_tokenizer = std::make_shared<text::WhitespaceTokenizerOperation>(with_offsets);
                      THROW_IF_ERROR(whitespace_tokenizer->ValidateParams());
                      return whitespace_tokenizer;
                    }));
                }));

PYBIND_REGISTER(NormalizeForm, 0, ([](const py::module *m) {
                  (void)py::enum_<NormalizeForm>(*m, "NormalizeForm", py::arithmetic())
                    .value("DE_NORMALIZE_NONE", NormalizeForm::kNone)
                    .value("DE_NORMALIZE_NFC", NormalizeForm::kNfc)
                    .value("DE_NORMALIZE_NFKC", NormalizeForm::kNfkc)
                    .value("DE_NORMALIZE_NFD", NormalizeForm::kNfd)
                    .value("DE_NORMALIZE_NFKD", NormalizeForm::kNfkd)
                    .export_values();
                }));
#endif

PYBIND_REGISTER(
  JiebaTokenizerOperation, 1, ([](const py::module *m) {
    (void)py::class_<text::JiebaTokenizerOperation, TensorOperation, std::shared_ptr<text::JiebaTokenizerOperation>>(
      *m, "JiebaTokenizerOperation")
      .def(
        py::init([](const std::string &hmm_path, const std::string &mp_path, const JiebaMode &mode, bool with_offsets) {
          auto jieba_tokenizer = std::make_shared<text::JiebaTokenizerOperation>(hmm_path, mp_path, mode, with_offsets);
          THROW_IF_ERROR(jieba_tokenizer->ValidateParams());
          return jieba_tokenizer;
        }))
      .def("add_word", [](text::JiebaTokenizerOperation &self, const std::string word, int64_t freq) {
        THROW_IF_ERROR(self.AddWord(word, freq));
      });
  }));

PYBIND_REGISTER(LookupOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::LookupOperation, TensorOperation, std::shared_ptr<text::LookupOperation>>(
                    *m, "LookupOperation")
                    .def(py::init([](const std::shared_ptr<Vocab> &vocab,
                                     const std::optional<std::string> &unknown_token, const std::string &data_type) {
                      auto lookup = std::make_shared<text::LookupOperation>(vocab, unknown_token, data_type);
                      THROW_IF_ERROR(lookup->ValidateParams());
                      return lookup;
                    }));
                }));

PYBIND_REGISTER(NgramOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::NgramOperation, TensorOperation, std::shared_ptr<text::NgramOperation>>(
                    *m, "NgramOperation")
                    .def(
                      py::init([](const std::vector<int32_t> &ngrams, const std::pair<std::string, int32_t> &left_pad,
                                  const std::pair<std::string, int32_t> &right_pad, const std::string &separator) {
                        auto ngram = std::make_shared<text::NgramOperation>(ngrams, left_pad, right_pad, separator);
                        THROW_IF_ERROR(ngram->ValidateParams());
                        return ngram;
                      }));
                }));

PYBIND_REGISTER(
  SentencePieceTokenizerOperation, 1, ([](const py::module *m) {
    (void)py::class_<text::SentencePieceTokenizerOperation, TensorOperation,
                     std::shared_ptr<text::SentencePieceTokenizerOperation>>(*m, "SentencePieceTokenizerOperation")
      .def(py::init([](const std::shared_ptr<SentencePieceVocab> &vocab, SPieceTokenizerOutType out_type) {
        auto SentencePieceTokenizer = std::make_shared<text::SentencePieceTokenizerOperation>(vocab, out_type);
        THROW_IF_ERROR(SentencePieceTokenizer->ValidateParams());
        return SentencePieceTokenizer;
      }))
      .def(py::init([](const std::string &vocab_path, SPieceTokenizerOutType out_type) {
        auto sentence_piece_tokenizer = std::make_shared<text::SentencePieceTokenizerOperation>(vocab_path, out_type);
        THROW_IF_ERROR(sentence_piece_tokenizer->ValidateParams());
        return sentence_piece_tokenizer;
      }));
  }));

PYBIND_REGISTER(
  SlidingWindowOperation, 1, ([](const py::module *m) {
    (void)py::class_<text::SlidingWindowOperation, TensorOperation, std::shared_ptr<text::SlidingWindowOperation>>(
      *m, "SlidingWindowOperation")
      .def(py::init([](const int32_t width, const int32_t axis) {
        auto sliding_window = std::make_shared<text::SlidingWindowOperation>(width, axis);
        THROW_IF_ERROR(sliding_window->ValidateParams());
        return sliding_window;
      }));
  }));

PYBIND_REGISTER(ToNumberOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::ToNumberOperation, TensorOperation, std::shared_ptr<text::ToNumberOperation>>(
                    *m, "ToNumberOperation")
                    .def(py::init([](std::string data_type) {
                      auto to_number = std::make_shared<text::ToNumberOperation>(data_type);
                      THROW_IF_ERROR(to_number->ValidateParams());
                      return to_number;
                    }));
                }));

PYBIND_REGISTER(TruncateSequencePairOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::TruncateSequencePairOperation, TensorOperation,
                                   std::shared_ptr<text::TruncateSequencePairOperation>>(
                    *m, "TruncateSequencePairOperation")
                    .def(py::init([](int32_t max_length) {
                      auto truncate_sequence_pair = std::make_shared<text::TruncateSequencePairOperation>(max_length);
                      THROW_IF_ERROR(truncate_sequence_pair->ValidateParams());
                      return truncate_sequence_pair;
                    }));
                }));

PYBIND_REGISTER(UnicodeCharTokenizerOperation, 1, ([](const py::module *m) {
                  (void)py::class_<text::UnicodeCharTokenizerOperation, TensorOperation,
                                   std::shared_ptr<text::UnicodeCharTokenizerOperation>>(
                    *m, "UnicodeCharTokenizerOperation")
                    .def(py::init([](bool with_offsets) {
                      auto unicode_char_tokenizer = std::make_shared<text::UnicodeCharTokenizerOperation>(with_offsets);
                      THROW_IF_ERROR(unicode_char_tokenizer->ValidateParams());
                      return unicode_char_tokenizer;
                    }));
                }));

// TODO(alexyuyue): Need to decouple WordpieceTokenizerOp to WordpieceTokenizerOperation after it's supported in C++
PYBIND_REGISTER(WordpieceTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<WordpieceTokenizerOp, TensorOp, std::shared_ptr<WordpieceTokenizerOp>>(
                    *m, "WordpieceTokenizerOp")
                    .def(py::init<const std::shared_ptr<Vocab> &, const std::string &, const int &, const std::string &,
                                  const bool &>());
                }));

PYBIND_REGISTER(JiebaMode, 0, ([](const py::module *m) {
                  (void)py::enum_<JiebaMode>(*m, "JiebaMode", py::arithmetic())
                    .value("DE_JIEBA_MIX", JiebaMode::kMix)
                    .value("DE_JIEBA_MP", JiebaMode::kMp)
                    .value("DE_JIEBA_HMM", JiebaMode::kHmm)
                    .export_values();
                }));

PYBIND_REGISTER(SPieceTokenizerLoadType, 0, ([](const py::module *m) {
                  (void)py::enum_<SPieceTokenizerLoadType>(*m, "SPieceTokenizerLoadType", py::arithmetic())
                    .value("DE_SPIECE_TOKENIZER_LOAD_KFILE", SPieceTokenizerLoadType::kFile)
                    .value("DE_SPIECE_TOKENIZER_LOAD_KMODEL", SPieceTokenizerLoadType::kModel)
                    .export_values();
                }));

PYBIND_REGISTER(SPieceTokenizerOutType, 0, ([](const py::module *m) {
                  (void)py::enum_<SPieceTokenizerOutType>(*m, "SPieceTokenizerOutType", py::arithmetic())
                    .value("DE_SPIECE_TOKENIZER_OUTTYPE_KString", SPieceTokenizerOutType::kString)
                    .value("DE_SPIECE_TOKENIZER_OUTTYPE_KINT", SPieceTokenizerOutType::kInt)
                    .export_values();
                }));
}  // namespace dataset
}  // namespace mindspore
