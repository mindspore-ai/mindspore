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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUT_DATA_TYPE_TRANS_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUT_DATA_TYPE_TRANS_PASS_H_
#include <string>
#include "backend/common/optimizer/pass.h"
#include "include/api/data_type.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/api/types.h"

namespace mindspore::opt {
class InputDTypeTransPass : public Pass {
 public:
  explicit InputDTypeTransPass(DataType dst_input_data_type, DataType src_input_data_type)
      : Pass("input_data_type_trans_pass"),
        dst_input_data_type_(dst_input_data_type),
        src_input_data_type_(src_input_data_type) {}
  ~InputDTypeTransPass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS HandleGraphInput(const FuncGraphPtr &graph);
  CNodePtr GenCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node, const std::string &cnode_name);
  DataType dst_input_data_type_ = DataType::kTypeUnknown;
  DataType src_input_data_type_ = DataType::kTypeUnknown;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUT_DATA_TYPE_TRANS_PASS_H_
