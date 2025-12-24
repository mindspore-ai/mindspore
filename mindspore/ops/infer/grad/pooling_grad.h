/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_POOLING_GRAD_H_
#define MINDSPORE_CORE_OPS_POOLING_GRAD_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePoolingGrad = "PoolingGrad";
class OPS_API PoolingGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PoolingGrad);
  PoolingGrad() : BaseOperator(kNamePoolingGrad) {}
  void Init(const PoolMode &pool_mode, const std::vector<int64_t> &window, const std::vector<int64_t> &stride,
            const PadMode &pad_mode, const std::vector<int64_t> &pad_list, const RoundMode &round_mode,
            const Format &format = NCHW, const bool global = false);
  void set_pool_mode(const PoolMode &pool_mode);
  void set_window(const std::vector<int64_t> &window);
  void set_stride(const std::vector<int64_t> &stride);
  void set_pad_mode(const PadMode &pad_mode);
  void set_pad_list(const std::vector<int64_t> &pad_list);
  void set_round_mode(const RoundMode &round_mode);
  void set_format(const Format &format);
  void set_global(const bool global);
  //  window(h, w)
  //  stride(h, w)
  //  pad_list(up, down, left, right)

  PoolMode get_pool_mode() const;
  std::vector<int64_t> get_window() const;
  std::vector<int64_t> get_stride() const;
  PadMode get_pad_mode() const;
  std::vector<int64_t> get_pad_list() const;
  RoundMode get_round_mode() const;
  Format get_format() const;
  bool get_global() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_POOLING_GRAD_H_
