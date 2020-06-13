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

#ifndef MINDSPORE_CCSRC_KERNEL_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_KERNEL_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_

#include <string>
#include <memory>
#include <vector>
#include "ir/anf.h"
#include "kernel/kernel.h"
#include "kernel/akg/akgkernelbuild.h"

namespace mindspore {
namespace kernel {
class AkgAscendKernelBuilder : public AkgKernelBuild {
 public:
  AkgAscendKernelBuilder() = default;
  ~AkgAscendKernelBuilder() = default;

  bool CollectJson(const AnfNodePtr &anf_node);
  bool CollectFusedJson(const std::vector<AnfNodePtr> &anf_nodes, const std::vector<AnfNodePtr> &input_list,
                        const std::vector<AnfNodePtr> &output_list);
  std::string json_name() const { return json_name_; }
  std::string kernel_json() const { return kernel_json_; }
  const std::vector<size_t> &input_size_list() const { return input_size_list_; }
  const std::vector<size_t> &output_size_list() const { return output_size_list_; }

 private:
  std::string kernel_json_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
};

bool AkgAscendKernelParallelBuild(const std::vector<AnfNodePtr> &anf_nodes);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
