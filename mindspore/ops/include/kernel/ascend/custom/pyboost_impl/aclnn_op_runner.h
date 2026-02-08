/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_EXTENSION_ASCEND_ACLNN_OP_RUNNER_H_
#define MINDSPORE_CCSRC_EXTENSION_ASCEND_ACLNN_OP_RUNNER_H_

#include <vector>
#include <tuple>
#include <memory>
#include <utility>
#include <functional>
#include <string>
#include "include/pynative/utils/pyboost/custom/pyboost_extension.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace ms::pynative {
class OPS_ASCEND_API AclnnOpRunner final : public PyboostRunner {
 public:
  using AclnnLaunchFunc = std::function<void(mindspore::device::DeviceContext *, size_t)>;
  using PyboostRunner::PyboostRunner;
  void SetLaunchFunc(const AclnnLaunchFunc &func) { launch_func_ = func; }
  void LaunchKernel() override {}

 protected:
  void _DispatchLaunchTask() override;

  AclnnLaunchFunc launch_func_{nullptr};
};

inline mindspore::tensor::TensorPtr Arg(const ms::Tensor &t) { return t.is_defined() ? t.tensor() : nullptr; }
inline std::vector<mindspore::tensor::TensorPtr> Arg(const std::vector<ms::Tensor> &tensors) {
  std::vector<mindspore::tensor::TensorPtr> result;
  result.reserve(tensors.size());
  for (const auto &t : tensors) {
    result.push_back(t.tensor());
  }
  return result;
}
inline std::optional<mindspore::tensor::TensorPtr> Arg(const std::optional<ms::Tensor> &opt_tensor) {
  if (opt_tensor.has_value()) {
    return Arg(opt_tensor.value());
  }
  return std::nullopt;
}
template <typename T>
inline constexpr T Arg(const T &arg) {
  return arg;
}

#define LAUNCH_ACLNN_FUNC(aclnn_api, ...)                                                                    \
  [](auto &&...args) {                                                                                       \
    auto args_t = std::make_tuple(ms::pynative::Arg(std::forward<decltype(args)>(args))...);                 \
    return [args_t](auto __dev_ctx, auto __stream_id) {                                                      \
      std::apply([&](auto &&...args) { LAUNCH_ACLNN(aclnn_api, __dev_ctx, __stream_id, args...); }, args_t); \
    };                                                                                                       \
  }(__VA_ARGS__)
}  // namespace ms::pynative
#endif  // MINDSPORE_CCSRC_EXTENSION_ASCEND_ACLNN_OP_RUNNER_H_
