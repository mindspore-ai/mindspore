/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/text/kernels/jieba_tokenizer_op.h"
#include "minddata/dataset/text/kernels/lookup_op.h"
#include "minddata/dataset/text/kernels/ngram_op.h"
#include "minddata/dataset/text/kernels/sliding_window_op.h"
#include "minddata/dataset/text/kernels/to_number_op.h"
#include "minddata/dataset/text/kernels/unicode_char_tokenizer_op.h"
#include "minddata/dataset/text/kernels/wordpiece_tokenizer_op.h"
#include "minddata/dataset/text/kernels/sentence_piece_tokenizer_op.h"
#include "minddata/dataset/text/kernels/truncate_sequence_pair_op.h"

#ifdef ENABLE_ICU4C
#include "minddata/dataset/text/kernels/basic_tokenizer_op.h"
#include "minddata/dataset/text/kernels/bert_tokenizer_op.h"
#include "minddata/dataset/text/kernels/case_fold_op.h"
#include "minddata/dataset/text/kernels/normalize_utf8_op.h"
#include "minddata/dataset/text/kernels/regex_replace_op.h"
#include "minddata/dataset/text/kernels/regex_tokenizer_op.h"
#include "minddata/dataset/text/kernels/unicode_script_tokenizer_op.h"
#include "minddata/dataset/text/kernels/whitespace_tokenizer_op.h"
#endif

namespace mindspore {
namespace dataset {

#ifdef ENABLE_ICU4C

PYBIND_REGISTER(BasicTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<BasicTokenizerOp, TensorOp, std::shared_ptr<BasicTokenizerOp>>(*m,
                                                                                                  "BasicTokenizerOp")
                    .def(py::init<const bool &, const bool &, const NormalizeForm &, const bool &, const bool &>());
                }));

PYBIND_REGISTER(WhitespaceTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<WhitespaceTokenizerOp, TensorOp, std::shared_ptr<WhitespaceTokenizerOp>>(
                    *m, "WhitespaceTokenizerOp")
                    .def(py::init<const bool &>());
                }));

PYBIND_REGISTER(UnicodeScriptTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<UnicodeScriptTokenizerOp, TensorOp, std::shared_ptr<UnicodeScriptTokenizerOp>>(
                    *m, "UnicodeScriptTokenizerOp")
                    .def(py::init<>())
                    .def(py::init<const bool &, const bool &>());
                }));

PYBIND_REGISTER(
  CaseFoldOp, 1, ([](const py::module *m) {
    (void)py::class_<CaseFoldOp, TensorOp, std::shared_ptr<CaseFoldOp>>(*m, "CaseFoldOp").def(py::init<>());
  }));

PYBIND_REGISTER(NormalizeUTF8Op, 1, ([](const py::module *m) {
                  (void)py::class_<NormalizeUTF8Op, TensorOp, std::shared_ptr<NormalizeUTF8Op>>(*m, "NormalizeUTF8Op")
                    .def(py::init<>())
                    .def(py::init<NormalizeForm>());
                }));

PYBIND_REGISTER(RegexReplaceOp, 1, ([](const py::module *m) {
                  (void)py::class_<RegexReplaceOp, TensorOp, std::shared_ptr<RegexReplaceOp>>(*m, "RegexReplaceOp")
                    .def(py::init<const std::string &, const std::string &, bool>());
                }));

PYBIND_REGISTER(RegexTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<RegexTokenizerOp, TensorOp, std::shared_ptr<RegexTokenizerOp>>(*m,
                                                                                                  "RegexTokenizerOp")
                    .def(py::init<const std::string &, const std::string &, const bool &>());
                }));

PYBIND_REGISTER(BertTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<BertTokenizerOp, TensorOp, std::shared_ptr<BertTokenizerOp>>(*m, "BertTokenizerOp")
                    .def(py::init<const std::shared_ptr<Vocab> &, const std::string &, const int &, const std::string &,
                                  const bool &, const bool &, const NormalizeForm &, const bool &, const bool &>());
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

PYBIND_REGISTER(JiebaTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<JiebaTokenizerOp, TensorOp, std::shared_ptr<JiebaTokenizerOp>>(*m,
                                                                                                  "JiebaTokenizerOp")
                    .def(py::init<const std::string &, const std::string &, const JiebaMode &, const bool &>())
                    .def("add_word", [](JiebaTokenizerOp &self, const std::string word, int freq) {
                      THROW_IF_ERROR(self.AddWord(word, freq));
                    });
                }));

PYBIND_REGISTER(UnicodeCharTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<UnicodeCharTokenizerOp, TensorOp, std::shared_ptr<UnicodeCharTokenizerOp>>(
                    *m, "UnicodeCharTokenizerOp")
                    .def(py::init<const bool &>());
                }));

PYBIND_REGISTER(LookupOp, 1, ([](const py::module *m) {
                  (void)py::class_<LookupOp, TensorOp, std::shared_ptr<LookupOp>>(*m, "LookupOp")
                    .def(py::init([](std::shared_ptr<Vocab> vocab, const py::object &py_word) {
                      if (vocab == nullptr) {
                        THROW_IF_ERROR(Status(StatusCode::kUnexpectedError, "vocab object type is incorrect or null."));
                      }
                      if (py_word.is_none()) {
                        return std::make_shared<LookupOp>(vocab, Vocab::kNoTokenExists);
                      }
                      std::string word = py::reinterpret_borrow<py::str>(py_word);
                      WordIdType default_id = vocab->Lookup(word);
                      if (default_id == Vocab::kNoTokenExists) {
                        THROW_IF_ERROR(Status(StatusCode::kUnexpectedError,
                                              "default unknown token: " + word + " doesn't exist in vocab."));
                      }
                      return std::make_shared<LookupOp>(vocab, default_id);
                    }));
                }));

PYBIND_REGISTER(NgramOp, 1, ([](const py::module *m) {
                  (void)py::class_<NgramOp, TensorOp, std::shared_ptr<NgramOp>>(*m, "NgramOp")
                    .def(py::init<const std::vector<int32_t> &, int32_t, int32_t, const std::string &,
                                  const std::string &, const std::string &>());
                }));

PYBIND_REGISTER(WordpieceTokenizerOp, 1, ([](const py::module *m) {
                  (void)py::class_<WordpieceTokenizerOp, TensorOp, std::shared_ptr<WordpieceTokenizerOp>>(
                    *m, "WordpieceTokenizerOp")
                    .def(py::init<const std::shared_ptr<Vocab> &, const std::string &, const int &, const std::string &,
                                  const bool &>());
                }));

PYBIND_REGISTER(SlidingWindowOp, 1, ([](const py::module *m) {
                  (void)py::class_<SlidingWindowOp, TensorOp, std::shared_ptr<SlidingWindowOp>>(*m, "SlidingWindowOp")
                    .def(py::init<uint32_t, int32_t>());
                }));

PYBIND_REGISTER(
  SentencePieceTokenizerOp, 1, ([](const py::module *m) {
    (void)py::class_<SentencePieceTokenizerOp, TensorOp, std::shared_ptr<SentencePieceTokenizerOp>>(
      *m, "SentencePieceTokenizerOp")
      .def(
        py::init<std::shared_ptr<SentencePieceVocab> &, const SPieceTokenizerLoadType, const SPieceTokenizerOutType>())
      .def(py::init<const std::string &, const std::string &, const SPieceTokenizerLoadType,
                    const SPieceTokenizerOutType>());
  }));

PYBIND_REGISTER(ToNumberOp, 1, ([](const py::module *m) {
                  (void)py::class_<ToNumberOp, TensorOp, std::shared_ptr<ToNumberOp>>(*m, "ToNumberOp")
                    .def(py::init<DataType>())
                    .def(py::init<std::string>());
                }));

PYBIND_REGISTER(TruncateSequencePairOp, 1, ([](const py::module *m) {
                  (void)py::class_<TruncateSequencePairOp, TensorOp, std::shared_ptr<TruncateSequencePairOp>>(
                    *m, "TruncateSequencePairOp")
                    .def(py::init<int64_t>());
                }));

PYBIND_REGISTER(JiebaMode, 0, ([](const py::module *m) {
                  (void)py::enum_<JiebaMode>(*m, "JiebaMode", py::arithmetic())
                    .value("DE_JIEBA_MIX", JiebaMode::kMix)
                    .value("DE_JIEBA_MP", JiebaMode::kMp)
                    .value("DE_JIEBA_HMM", JiebaMode::kHmm)
                    .export_values();
                }));

PYBIND_REGISTER(SPieceTokenizerOutType, 0, ([](const py::module *m) {
                  (void)py::enum_<SPieceTokenizerOutType>(*m, "SPieceTokenizerOutType", py::arithmetic())
                    .value("DE_SPIECE_TOKENIZER_OUTTYPE_KString", SPieceTokenizerOutType::kString)
                    .value("DE_SPIECE_TOKENIZER_OUTTYPE_KINT", SPieceTokenizerOutType::kInt)
                    .export_values();
                }));

PYBIND_REGISTER(SPieceTokenizerLoadType, 0, ([](const py::module *m) {
                  (void)py::enum_<SPieceTokenizerLoadType>(*m, "SPieceTokenizerLoadType", py::arithmetic())
                    .value("DE_SPIECE_TOKENIZER_LOAD_KFILE", SPieceTokenizerLoadType::kFile)
                    .value("DE_SPIECE_TOKENIZER_LOAD_KMODEL", SPieceTokenizerLoadType::kModel)
                    .export_values();
                }));

}  // namespace dataset
}  // namespace mindspore
