/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "src/delegate/npu/pass/npu_infer_format_pass.h"
#include <vector>
#include <queue>
#include <map>
#include "src/delegate/npu/pass/npu_pass_utils.h"
#include "src/delegate/npu/npu_converter_utils.h"
#include "src/tensor.h"
#include "src/cxx_api/tensor/tensor_impl.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
int NPUInferFormatPass::Run(NPUGraph *subgraph) {
  CHECK_NULL_RETURN(subgraph);
  all_ops_ = subgraph->GetOps();
  all_tensors_ = subgraph->GetInsertTensors();
  auto graph_inputs = subgraph->inputs();
  std::queue<NPUOp *> infer_ops;
  std::map<tensor::MSTensor *, bool> is_inferred;
  // initialization
  for (auto op : *all_ops_) {
    infer_ops.push(op);
  }
  for (auto tensor : *all_tensors_) {
    is_inferred[tensor->impl()->lite_tensor()] = false;
  }
  for (auto input_tensor : graph_inputs) {
    is_inferred[input_tensor.impl()->lite_tensor()] = true;
  }
  while (!infer_ops.empty()) {
    auto cur_op = infer_ops.front();
    infer_ops.pop();
    bool input_inferred = std::all_of(cur_op->inputs().begin(), cur_op->inputs().end(), [&](auto in_tensor) {
      return is_inferred[in_tensor.impl()->lite_tensor()] == true || in_tensor.IsConst();
    });
    if (input_inferred) {
      auto dst_format = cur_op->inputs().at(0).format();
      if (NPUPassUtils::IsNhwc2Nchw(cur_op) && dst_format == Format::NHWC) {
        dst_format = Format::NCHW;
      } else if (NPUPassUtils::IsNchw2Nhwc(cur_op) && dst_format == Format::NCHW) {
        dst_format = Format::NHWC;
      }
      for (auto &out_tensor : cur_op->outputs()) {
        const_cast<mindspore::MSTensor &>(out_tensor).SetFormat(dst_format);
        is_inferred[out_tensor.impl()->lite_tensor()] = true;
      }
    } else {
      infer_ops.push(cur_op);
    }
  }
  return RET_OK;
}
}  // namespace mindspore
