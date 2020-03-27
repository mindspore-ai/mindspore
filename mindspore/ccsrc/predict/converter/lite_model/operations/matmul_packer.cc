/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/lite_model/op_attr_packer.h"

namespace mindspore {
namespace predict {
namespace convert {
bool MatMulPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op) {
  if (c_node_ptr == nullptr || ms_op == nullptr) {
    return false;
  }
  bool kernel_transpore_a = AnfAlgo::GetNodeAttr<bool>(c_node_ptr, "transpose_a");
  bool kernel_transpore_b = AnfAlgo::GetNodeAttr<bool>(c_node_ptr, "transpose_b");
  std::unique_ptr<MatMulT> attr(new MatMulT());
  MS_EXCEPTION_IF_NULL(attr);
  attr->transposeA = kernel_transpore_a;
  attr->transposeB = kernel_transpore_b;
  ms_op->name = c_node_ptr->fullname_with_scope();
  ms_op->attr.type = OpT_MatMul;
  ms_op->attr.value = attr.release();
  return true;
}
}  // namespace convert
}  // namespace predict
}  // namespace mindspore
