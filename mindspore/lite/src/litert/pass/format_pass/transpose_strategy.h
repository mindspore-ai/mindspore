/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_PASS_TRANSPOSE_STRATEGY_H_
#define MINDSPORE_LITE_SRC_RUNTIME_PASS_TRANSPOSE_STRATEGY_H_

#include <map>
#include <set>
#include <vector>
#include <functional>
#include <unordered_map>
#include "src/executor/kernel_exec.h"
#include "src/litert/pass/format_pass/pass_utils.h"

namespace mindspore::lite::pass {
static TransInfoPair kNHWC2NCHWTrans = {Format::NHWC, Format::NCHW};
static TransInfoPair kNCHW2NHWCTrans = {Format::NCHW, Format::NHWC};

template <typename T>
T TransFormAxis(T axis, const TransInfoPair &trans) {
  if (IsSameTranspose(trans, kNHWC2NCHWTrans)) {
    switch (axis) {
      case kNHWC_N:
        return kNCHW_N;
      case kNHWC_H:
        return kNCHW_H;
      case kNHWC_W:
        return kNCHW_W;
      case kNHWC_C:
        return kNCHW_C;
      default:
        return axis;
    }
  }
  if (IsSameTranspose(trans, kNCHW2NHWCTrans)) {
    switch (axis) {
      case kNCHW_N:
        return kNHWC_N;
      case kNCHW_H:
        return kNHWC_H;
      case kNCHW_W:
        return kNHWC_W;
      case kNCHW_C:
        return kNHWC_C;
      default:
        return axis;
    }
  }
  return axis;
}

class TransposeStrategy {
 public:
  TransposeStrategy() = default;
  ~TransposeStrategy() = default;

  size_t GetTransCount(const std::vector<kernel::KernelExec *> &kernels, TransInfoPair *trans_info);
  bool CrossKernelFusionPreCheck(const kernel::KernelExec *kernel, TransInfoPair *pre_trans, TransInfoPair *post_trans);
  static int TryTransKernelAxis(kernel::KernelExec *kernel, const TransInfoPair &trans);
};
}  // namespace mindspore::lite::pass
#endif  // MINDSPORE_LITE_SRC_RUNTIME_PASS_TRANSPOSE_STRATEGY_H_
