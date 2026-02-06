/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CONVERT_INPUT_AND_ATTR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CONVERT_INPUT_AND_ATTR_H_

#include <set>
#include <string>
#include <memory>
#include "include/backend/common/pass_manager/pass.h"
#include "ops/op_def.h"

namespace mindspore::graphkernel {
class ConvertFrontEndToGraphKernel : public opt::Pass {
 public:
  ConvertFrontEndToGraphKernel() : Pass("convert_input_to_attr") {}
  ~ConvertFrontEndToGraphKernel() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool Process(const CNodePtr &cnode, const ops::OpDefPtr &op_def, const PrimitivePtr &primitive);
  bool AddConstInputToAttr(const CNodePtr &cnode, const size_t input_index, const std::string &arg_name,
                           const std::string &arg_handler, const PrimitivePtr &primitive);
};
class ConvertGraphKernelToFrontEnd : public opt::Pass {
 public:
  ConvertGraphKernelToFrontEnd() : Pass("convert_attr_to_input") {}
  ~ConvertGraphKernelToFrontEnd() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  static bool Process(const AnfNodePtr &node);

 private:
  static void AddAttrToInput(const CNodePtr &cnode, const std::string &arg_name, const std::string &arg_handler,
                             const PrimitivePtr &primitive, size_t pos);
  static bool ConvertInputsType(const CNodePtr &cnode, size_t idx, ops::OP_DTYPE fe_arg_type);
};

class OpDefAdapter {
 public:
  /// \brief Check whether need to convert the node from GraphKernel format to frontend format,
  /// includes input/attr and kernel object.
  static bool NeedConvertGK2FE(const AnfNodePtr &node);
  static bool NeedConvertInputAndAttr(const AnfNodePtr &node);
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CONVERT_INPUT_AND_ATTR_H_
