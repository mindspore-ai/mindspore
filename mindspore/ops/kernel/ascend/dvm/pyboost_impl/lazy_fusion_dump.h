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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_DUMP_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_DUMP_H

#include <vector>
#include <string>
#include <utility>
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
using TensorPtr = tensor::TensorPtr;

class LazyFusionDump {
 public:
  static LazyFusionDump &Instance();

  std::string ToString(const TypeId &t) { return TypeIdToString(t); }

  std::string ToString(const ValuePtr &t) {
    if (t == nullptr) {
      return "nullptr";
    }
    if (t->isa<Scalar>()) {
      auto scalar_value = t->cast<ScalarPtr>();
      MS_EXCEPTION_IF_NULL(scalar_value);
      return scalar_value->DumpText();
    } else if (t->isa<tensor::Tensor>()) {
      return TensorStr(t->cast<TensorPtr>());
    }
    return t->ToString();
  }

  std::string TensorStr(const TensorPtr &t);

  void DumpGraphInfo(std::stringstream *buf);
  void DumpKernelInfo(std::stringstream *buf);

 private:
  LazyFusionDump();
  ~LazyFusionDump() = default;

  void DumpToFile(const std::string &file_path, std::stringstream *buf);
  void CreateDumpDir();

  bool enable_dump_{true};
  std::string dump_dir_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_DVM_LAZY_FUSION_DUMP_H
