/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_BUILD_H_

#include <string>
#include <unordered_map>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "ir/dtype.h"
#include "kernel/kernel.h"
#include "kernel/oplib/oplib.h"
#include "plugin/device/ascend/kernel/tbe/tbe_adapter.h"

namespace mindspore {
namespace kernel {

class TbeKernelBuild {
 public:
  static bool GetIOSize(const nlohmann::json &kernel_json, std::vector<size_t> *input_size_list,
                        std::vector<size_t> *output_size_list);
  static bool GetIOSize(const nlohmann::json &fusion_op_list, const std::vector<AnfNodePtr> &output_nodes,
                        std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list);
  static void CalInputSize(const nlohmann::json &fusion_op_list, std::vector<size_t> *input_size_list);
  static bool CalOutputSize(const nlohmann::json &fusion_op_list,
                            const std::vector<mindspore::AnfNodePtr> &output_nodes,
                            std::vector<size_t> *output_size_list);

 private:
  TbeKernelBuild() = default;
  ~TbeKernelBuild() = default;
  static size_t GetIOSizeImpl(const nlohmann::json &desc);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_BUILD_H_
