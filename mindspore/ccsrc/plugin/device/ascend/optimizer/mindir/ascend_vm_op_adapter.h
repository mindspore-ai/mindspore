/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ASCEND_CONVERT_CONST_INPUT_TO_ATTR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ASCEND_CONVERT_CONST_INPUT_TO_ATTR_H_
#include <string>
#include <memory>
#include <vector>
#include <map>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/op_adaptation_info_factory.h"

namespace mindspore {
namespace opt {
class AscendVmOpAdapter : public PatternProcessPass {
 public:
  explicit AscendVmOpAdapter(bool multigraph = true) : PatternProcessPass("ascend_vm_op_adapter", multigraph) {}
  ~AscendVmOpAdapter() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;

 private:
  CNodePtr ConvertToTargetOp(const CNodePtr &origin_op, mindspore::opt::OpAdaptationInfo *op_adaptation_info) const;
  CNodePtr CreateTargetOp(const CNodePtr &origin_op, const OpAdaptationInfo &op_adaptation_info) const;
  bool ConvertInputToAttr(const CNodePtr &origin_op, const string &target_op_name,
                          const std::vector<std::string> &input_names_vec, size_t i,
                          const std::shared_ptr<AnfNode> &input_node,
                          const std::map<size_t, InputAttrInfo>::iterator &iter,
                          const std::shared_ptr<Primitive> &target_primitive) const;
  string GetAttrName(const string &target_op_name, const std::map<size_t, InputAttrInfo>::iterator &iter,
                     const string &input_name) const;
  ValuePtr UpdateAttrValueByDtype(const ValuePtr &value, const string &attr_data_type) const;
  ValuePtr UpdateAttrValue(const CNodePtr &origin_op, const std::map<size_t, InputAttrInfo>::iterator &iter,
                           const ValuePtr &value, const string &attr_name) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_ASCEND_CONVERT_CONST_INPUT_TO_ATTR_H_
