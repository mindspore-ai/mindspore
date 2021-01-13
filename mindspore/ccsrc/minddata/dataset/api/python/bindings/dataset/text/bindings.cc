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
#include "minddata/dataset/text/vocab.h"
#include "minddata/dataset/text/sentence_piece_vocab.h"
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(Vocab, 0, ([](const py::module *m) {
                  (void)py::class_<Vocab, std::shared_ptr<Vocab>>(*m, "Vocab")
                    .def(py::init<>())
                    .def_static("from_list",
                                [](const py::list &words, const py::list &special_tokens, bool special_first) {
                                  std::shared_ptr<Vocab> v;
                                  THROW_IF_ERROR(Vocab::BuildFromPyList(words, special_tokens, special_first, &v));
                                  return v;
                                })
                    .def_static(
                      "from_file",
                      [](const std::string &path, const std::string &dlm, int32_t vocab_size,
                         const py::list &special_tokens, bool special_first) {
                        std::shared_ptr<Vocab> v;
                        THROW_IF_ERROR(Vocab::BuildFromFile(path, dlm, vocab_size, special_tokens, special_first, &v));
                        return v;
                      })
                    .def_static("from_dict", [](const py::dict &words) {
                      std::shared_ptr<Vocab> v;
                      THROW_IF_ERROR(Vocab::BuildFromPyDict(words, &v));
                      return v;
                    });
                }));

PYBIND_REGISTER(SentencePieceVocab, 0, ([](const py::module *m) {
                  (void)py::class_<SentencePieceVocab, std::shared_ptr<SentencePieceVocab>>(*m, "SentencePieceVocab")
                    .def(py::init<>())
                    .def_static("from_file",
                                [](const py::list &paths, const int32_t vocab_size, const float character_coverage,
                                   const SentencePieceModel model_type, const py::dict &params) {
                                  std::shared_ptr<SentencePieceVocab> v;
                                  std::vector<std::string> path_list;
                                  for (auto path : paths) {
                                    path_list.emplace_back(py::str(path));
                                  }
                                  std::unordered_map<std::string, std::string> param_map;
                                  for (auto param : params) {
                                    std::string key = py::reinterpret_borrow<py::str>(param.first);
                                    if (key == "input" || key == "vocab_size" || key == "model_prefix" ||
                                        key == "character_coverage" || key == "model_type") {
                                      continue;
                                    }
                                    param_map[key] = py::reinterpret_borrow<py::str>(param.second);
                                  }
                                  THROW_IF_ERROR(SentencePieceVocab::BuildFromFile(
                                    path_list, vocab_size, character_coverage, model_type, param_map, &v));
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

}  // namespace dataset
}  // namespace mindspore
