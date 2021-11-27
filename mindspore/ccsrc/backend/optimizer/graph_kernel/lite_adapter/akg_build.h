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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_LITE_ADAPTER_AKG_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_LITE_ADAPTER_AKG_BUILD_H_
#include <string>
#include "utils/anf_utils.h"

namespace mindspore::graphkernel {
constexpr size_t PROCESS_LIMIT = 8;
constexpr size_t TIME_OUT = 100;

class AkgKernelBuilder {
 public:
  AkgKernelBuilder() = default;
  ~AkgKernelBuilder() = default;

  bool CompileJsonsInAnfnodes(const AnfNodePtrList &node_list);
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_LITE_ADAPTER_AKG_BUILD_H_
