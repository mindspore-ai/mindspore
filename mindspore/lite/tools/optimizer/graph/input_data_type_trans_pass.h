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
#include "include/backend/optimizer/pass.h"
#include "include/api/data_type.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/api/types.h"

namespace mindspore::opt {
class InOutDTypeTransPass : public Pass {
 public:
  explicit InOutDTypeTransPass(DataType dst_input_data_type, DataType dst_output_data_type)
      : Pass("inout_data_type_trans_pass"),
        dst_input_data_type_(static_cast<TypeId>(dst_input_data_type)),
        dst_output_data_type_(static_cast<TypeId>(dst_output_data_type)) {}
  ~InOutDTypeTransPass() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS HandleGraphInput(const FuncGraphPtr &graph);
  STATUS HandleGraphOutput(const FuncGraphPtr &graph);
  TypeId dst_input_data_type_ = kTypeUnknown;
  TypeId dst_output_data_type_ = kTypeUnknown;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUT_DATA_TYPE_TRANS_PASS_H_
