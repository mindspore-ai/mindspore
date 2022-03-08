/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/text.h"
#include "minddata/dataset/text/char_n_gram.h"
#include "minddata/dataset/text/fast_text.h"
#include "minddata/dataset/text/glove.h"
#include "minddata/dataset/text/vectors.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(Vocab, 0, ([](const py::module *m) {
                  (void)py::class_<Vocab, std::shared_ptr<Vocab>>(*m, "Vocab")
                    .def(py::init<>())
                    .def_static("from_list",
                                [](const std::vector<std::string> &words,
                                   const std::vector<std::string> &special_tokens, bool special_first) {
                                  std::shared_ptr<Vocab> v;
                                  THROW_IF_ERROR(Vocab::BuildFromVector(words, special_tokens, special_first, &v));
                                  return v;
                                })
                    .def_static(
                      "from_file",
                      [](const std::string &path, const std::string &dlm, int32_t vocab_size,
                         const std::vector<std::string> &special_tokens, bool special_first) {
                        std::shared_ptr<Vocab> v;
                        THROW_IF_ERROR(Vocab::BuildFromFile(path, dlm, vocab_size, special_tokens, special_first, &v));
                        return v;
                      })
                    .def_static("from_dict",
                                [](const std::unordered_map<WordType, WordIdType> &words) {
                                  std::shared_ptr<Vocab> v;
                                  THROW_IF_ERROR(Vocab::BuildFromUnorderedMap(words, &v));
                                  return v;
                                })
                    .def("tokens_to_ids",
                         [](Vocab &self, const std::vector<std::string> words) {
                           auto ids = self.TokensToIds(words);
                           py::object ret;
                           if (ids.size() == 1) {
                             ret = py::int_(ids[0]);
                           } else {
                             py::list ids_list = py::cast(ids);
                             ret = py::object(ids_list);
                           }
                           return ret;
                         })
                    .def("ids_to_tokens",
                         [](Vocab &self, const std::vector<int32_t> ids) {
                           auto words = self.IdsToTokens(ids);
                           py::object ret;
                           if (words.size() == 1) {
                             ret = py::str(words[0]);
                           } else {
                             py::list words_list = py::cast(words);
                             ret = py::object(words_list);
                           }
                           return ret;
                         })
                    .def("vocab", [](Vocab &self) { return self.GetVocab(); });
                }));

PYBIND_REGISTER(SentencePieceVocab, 0, ([](const py::module *m) {
                  (void)py::class_<SentencePieceVocab, std::shared_ptr<SentencePieceVocab>>(*m, "SentencePieceVocab")
                    .def(py::init<>())
                    .def_static("from_file",
                                [](const std::vector<std::string> &paths, const int32_t vocab_size,
                                   const float character_coverage, const SentencePieceModel model_type,
                                   const std::unordered_map<std::string, std::string> &params) {
                                  std::shared_ptr<SentencePieceVocab> v;
                                  THROW_IF_ERROR(SentencePieceVocab::BuildFromFile(
                                    paths, vocab_size, character_coverage, model_type, params, &v));
                                  return v;
                                })
                    .def_static("save_model", [](const std::shared_ptr<SentencePieceVocab> *vocab, std::string path,
                                                 std::string filename) {
                      THROW_IF_ERROR(SentencePieceVocab::SaveModel(vocab, path, filename));
                    });
                }));

PYBIND_REGISTER(SentencePieceModel, 0, ([](const py::module *m) {
                  (void)py::enum_<SentencePieceModel>(*m, "SentencePieceModel", py::arithmetic())
                    .value("DE_SENTENCE_PIECE_UNIGRAM", SentencePieceModel::kUnigram)
                    .value("DE_SENTENCE_PIECE_BPE", SentencePieceModel::kBpe)
                    .value("DE_SENTENCE_PIECE_CHAR", SentencePieceModel::kChar)
                    .value("DE_SENTENCE_PIECE_WORD", SentencePieceModel::kWord)
                    .export_values();
                }));

PYBIND_REGISTER(CharNGram, 1, ([](const py::module *m) {
                  (void)py::class_<CharNGram, Vectors, std::shared_ptr<CharNGram>>(*m, "CharNGram")
                    .def(py::init<>())
                    .def_static("from_file", [](const std::string &path, int32_t max_vectors) {
                      std::shared_ptr<CharNGram> char_n_gram;
                      THROW_IF_ERROR(CharNGram::BuildFromFile(&char_n_gram, path, max_vectors));
                      return char_n_gram;
                    });
                }));

PYBIND_REGISTER(FastText, 1, ([](const py::module *m) {
                  (void)py::class_<FastText, Vectors, std::shared_ptr<FastText>>(*m, "FastText")
                    .def(py::init<>())
                    .def_static("from_file", [](const std::string &path, int32_t max_vectors) {
                      std::shared_ptr<FastText> fast_text;
                      THROW_IF_ERROR(FastText::BuildFromFile(&fast_text, path, max_vectors));
                      return fast_text;
                    });
                }));

PYBIND_REGISTER(GloVe, 1, ([](const py::module *m) {
                  (void)py::class_<GloVe, Vectors, std::shared_ptr<GloVe>>(*m, "GloVe")
                    .def(py::init<>())
                    .def_static("from_file", [](const std::string &path, int32_t max_vectors) {
                      std::shared_ptr<GloVe> glove;
                      THROW_IF_ERROR(GloVe::BuildFromFile(&glove, path, max_vectors));
                      return glove;
                    });
                }));

PYBIND_REGISTER(Vectors, 0, ([](const py::module *m) {
                  (void)py::class_<Vectors, std::shared_ptr<Vectors>>(*m, "Vectors")
                    .def(py::init<>())
                    .def_static("from_file", [](const std::string &path, int32_t max_vectors) {
                      std::shared_ptr<Vectors> vectors;
                      THROW_IF_ERROR(Vectors::BuildFromFile(&vectors, path, max_vectors));
                      return vectors;
                    });
                }));
}  // namespace dataset
}  // namespace mindspore
