/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_
#define MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_

#include <map>
#include <memory>
#include <utility>
#include <string>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "ir/tensor.h"

namespace mindspore {
namespace tensor {
namespace py = pybind11;
using AutoGradMetaDataWeakPtr = std::weak_ptr<AutoGradMetaData>;

struct RegisterHook {
  /// \brief Register a backward hook
  ///
  /// \ void
  static uint64_t RegisterTensorBackwardHook(const Tensor &tensor, const py::function &hook);

  /// \brief Remove a backward hook
  ///
  /// \ void
  static void RemoveTensorBackwardHook(uint64_t id);

  /// \brief Update weight meta
  ///
  /// \ void
  static void UpdateTensorBackwardHook(const AutoGradMetaDataPtr &auto_grad_meta_data, const std::string &id);

  static void ClearHookMap() { hook_meta_fn_map_.clear(); }

  // For store hook
  static std::map<uint64_t, std::pair<AutoGradMetaDataWeakPtr, TensorBackwardHookPtr>> hook_meta_fn_map_;
};

}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_
