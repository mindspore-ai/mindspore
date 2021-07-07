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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_PARAMS_INFO_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_PARAMS_INFO_H_

#include <utility>
#include <string>
#include <vector>
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
// ParamsInfo is used for server computation kernel's register, e.g, ApplyMomentumKernel, FedAvgKernel, etc.
// Register of a server kernel needs every inputs/workspace/outputs parameters' name and type.
// For example:
// ParamsInfo()
//   .AddInputNameType("input1_name", kNumberTypeFloat32)
//   .AddInputNameType("input2_name", kNumberTypeUInt64)
//   .AddWorkspaceNameType("workspace1_name", kNumberTypeFloat32)
//   .AddOutputNameType("output1_name", kNumberTypeUInt64)
// This invocation describes a server kernel with parameters below:
//    An input with name "input1_name" and type float32.
//    An input with name "input1_name" and type uint_64.
//    A workspace with name "workspace1_name" and type float32.
//    An output with name "output1_name" and type float32.
class ParamsInfo {
 public:
  ParamsInfo() = default;
  ~ParamsInfo() = default;

  ParamsInfo &AddInputNameType(const std::string &name, TypeId type);
  ParamsInfo &AddWorkspaceNameType(const std::string &name, TypeId type);
  ParamsInfo &AddOutputNameType(const std::string &name, TypeId type);
  size_t inputs_num() const;
  size_t outputs_num() const;
  const std::pair<std::string, TypeId> &inputs_name_type(size_t index) const;
  const std::pair<std::string, TypeId> &outputs_name_type(size_t index) const;
  const std::vector<std::string> &inputs_names() const;
  const std::vector<std::string> &workspace_names() const;
  const std::vector<std::string> &outputs_names() const;

 private:
  std::vector<std::pair<std::string, TypeId>> inputs_name_type_;
  std::vector<std::pair<std::string, TypeId>> workspaces_name_type_;
  std::vector<std::pair<std::string, TypeId>> outputs_name_type_;
  std::vector<std::string> inputs_names_;
  std::vector<std::string> workspace_names_;
  std::vector<std::string> outputs_names_;
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_PARAMS_INFO_H_
