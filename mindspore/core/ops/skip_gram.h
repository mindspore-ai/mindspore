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

#ifndef MINDSPORE_CORE_OPS_SKIP_GRAM_H_
#define MINDSPORE_CORE_OPS_SKIP_GRAM_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSkipGram = "SkipGram";
/// \brief SkipGram defined the SkipGram operator prototype.
class MS_CORE_API SkipGram : public PrimitiveC {
 public:
  /// \brief Constructor.
  SkipGram() : PrimitiveC(kNameSkipGram) {}

  /// \brief Destructor.
  ~SkipGram() = default;

  MS_DECLARE_PARENT(SkipGram, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] include_all_grams Define a boolean value to indicate whether to include ngram with size lower than
  ///            ngram_size.
  /// \param[in] max_skip_size Define the maximum number of word to skip.
  /// \param[in] ngram_size Define the the number of word each output item.
  void Init(const bool include_all_grams, const int64_t max_skip_size, const int64_t ngram_size);

  /// \brief Method to set include_all_grams attribute.
  ///
  /// \param[in] include_all_grams Define a boolean value to indicate whether to include ngram with size lower than
  ///            ngram_size.
  void set_include_all_grams(const bool include_all_grams);

  /// \brief Method to get include_all_grams attribute.
  ///
  /// \return a boolean value.
  bool get_include_all_grams() const;

  /// \brief Method to set max_skip_size attribute.
  ///
  /// \param[in] max_skip_size Define the maximum number of word to skip.
  void set_max_skip_size(const int64_t max_skip_size);

  /// \brief Method to get max_skip_size attribute.
  ///
  /// \return an integer value.
  int64_t get_max_skip_size() const;

  /// \brief Method to set ngram_size attribute.
  ///
  /// \param[in] ngram_size Define the the number of word each output item.
  void set_ngram_size(const int64_t ngram_size);

  /// \brief Method to get ngram_size attribute.
  ///
  /// \return an integer value.
  int64_t get_ngram_size() const;
};
AbstractBasePtr SkipGramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args);
using PrimSkipGramPtr = std::shared_ptr<SkipGram>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SKIP_GRAM_H_
