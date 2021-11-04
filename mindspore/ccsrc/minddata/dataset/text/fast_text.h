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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_FAST_TEXT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_FAST_TEXT_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/include/dataset/iterator.h"
#include "minddata/dataset/text/vectors.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {
/// \brief Pre-train word vectors.
class FastText : public Vectors {
 public:
  /// Constructor.
  FastText() = default;

  /// Constructor.
  /// \param[in] map A map between string and vector.
  /// \param[in] dim Dimension of the vectors.
  FastText(const std::unordered_map<std::string, std::vector<float>> &map, int32_t dim);

  /// Destructor.
  ~FastText() = default;

  /// \brief Build Vectors from reading a pre-train vector file.
  /// \param[out] fast_text FastText object which contains the pre-train vectors.
  /// \param[in] path Path to the pre-trained word vector file. The suffix of set must be `*.vec`.
  /// \param[in] max_vectors This can be used to limit the number of pre-trained vectors loaded (default=0, no limit).
  static Status BuildFromFile(std::shared_ptr<FastText> *fast_text, const std::string &path, int32_t max_vectors = 0);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_FAST_TEXT_H_
