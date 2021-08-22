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
class SkipGram : public PrimitiveC {
 public:
  SkipGram() : PrimitiveC(kNameSkipGram) {}
  ~SkipGram() = default;
  MS_DECLARE_PARENT(SkipGram, PrimitiveC);
  void Init(const bool include_all_grams, const int64_t max_skip_size, const int64_t ngram_size);
  void set_include_all_grams(const bool include_all_grams);
  bool get_include_all_grams() const;
  void set_max_skip_size(const int64_t max_skip_size);
  int64_t get_max_skip_size() const;
  void set_ngram_size(const int64_t ngram_size);
  int64_t get_ngram_size() const;
};
AbstractBasePtr SkipGramInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args);
using PrimSkipGramPtr = std::shared_ptr<SkipGram>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SKIP_GRAM_H_
