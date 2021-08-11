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

#ifndef MINDSPORE_CORE_OPS_SPLIT_WITH_OVERLAP_H_
#define MINDSPORE_CORE_OPS_SPLIT_WITH_OVERLAP_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
namespace mindspore {
namespace ops {
constexpr auto kNameSplitWithOverlap = "SplitWithOverlap";
class MS_CORE_API SplitWithOverlap : public PrimitiveC {
 public:
  SplitWithOverlap() : PrimitiveC(kNameSplitWithOverlap) {}
  ~SplitWithOverlap() = default;
  MS_DECLARE_PARENT(SplitWithOverlap, PrimitiveC);
  void Init(int64_t number_split, const std::vector<int64_t> &ratio, const std::vector<int64_t> &extend_top,
            const std::vector<int64_t> &extend_bottom, int64_t split_dim, int64_t stride, int64_t pad_top,
            bool trans_format);

  void set_ratio(const std::vector<int64_t> &ratio);
  void set_extend_top(const std::vector<int64_t> &extend_top);
  void set_extend_bottom(const std::vector<int64_t> &extend_bottom);
  void set_number_split(int64_t number_split);
  void set_split_dim(int64_t split_dim);
  void set_split_stride(int64_t stride);
  void set_pad_top(int64_t pad_top);
  void set_trans_format(bool trans_format);

  std::vector<int64_t> get_ratio() const;
  std::vector<int64_t> get_extend_top() const;
  std::vector<int64_t> get_extend_bottom() const;
  int64_t get_number_split() const;
  int64_t get_split_dim() const;
  int64_t get_split_stride() const;
  int64_t get_pad_top() const;
  bool get_trans_format() const;
};
AbstractBasePtr SplitWithOverlapInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args);
using PrimSplitWithOverlap = std::shared_ptr<SplitWithOverlap>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SPLIT_WITH_OVERLAP_H_
