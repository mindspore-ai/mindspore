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

  std::string ToString(const ValueTuplePtr &t) { return (t == nullptr ? "nullptr" : t->ToString()); }

  std::string ToString(const ScalarPtr &t) { return (t == nullptr ? "nullptr" : t->DumpText()); }

  std::string ToString(const TensorPtr &t);

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
