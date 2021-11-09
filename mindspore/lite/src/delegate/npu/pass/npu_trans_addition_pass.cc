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
#include "src/delegate/npu/pass/npu_trans_addition_pass.h"
#include <vector>
#include "src/delegate/npu/pass/npu_pass_utils.h"
#include "src/delegate/npu/npu_converter_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
int NPUTransAdditionPass::InsertTransNodes(NPUOp *op, std::vector<NPUOp *> *trans_ops,
                                           std::vector<mindspore::MSTensor> graph_outputs) {
  MS_ASSERT(op != nullptr);
  MS_ASSERT(trans_ops != nullptr);
  // Get the post ops, which have no need to insert trans op.
  auto post_non_insert_ops = op->out_ops();
  for (auto cur_output : op->outputs()) {
    if (std::find(graph_outputs.begin(), graph_outputs.end(), cur_output) != graph_outputs.end() &&
        cur_output.Shape().size() == NPU_SHAPE_SIZE && cur_output.format() == Format::NCHW) {
      // Create additional post transpose op's in tensor.
      auto name = op->name() + "_addition_trans" + "_Nchw2Nhwc" + std::to_string(total_++);
      auto nchw_shape = cur_output.Shape();
      std::vector<int64_t> nhwc_shape = {nchw_shape[NCHW_N], nchw_shape[NCHW_H], nchw_shape[NCHW_W],
                                         nchw_shape[NCHW_C]};
      auto nchw_tensor =
        mindspore::MSTensor::CreateTensor(name + "/input0", cur_output.DataType(), nchw_shape, nullptr, 0);
      if (nchw_tensor == nullptr) {
        MS_LOG(ERROR) << "New nchw tensor failed when inserting post nchw2nhwc op.";
        return RET_ERROR;
      }
      all_tensors_->push_back(nchw_tensor);

      // Change original shape and format of cur tensor
      cur_output.SetShape(nhwc_shape);
      cur_output.SetFormat(Format::NHWC);

      std::vector<mindspore::MSTensor> nc2nh_inputs{*nchw_tensor};
      std::vector<mindspore::MSTensor> nc2nh_outputs{cur_output};
      // Create post transAddition op: Nchw2Nhwc
      auto *post_trans_op = NPUPassUtils::CreateNchw2NhwcOp(nc2nh_inputs, nc2nh_outputs, name);
      if (post_trans_op == nullptr) {
        MS_LOG(ERROR) << "Create Nchw2Nhwc transpose op failed.";
        return RET_ERROR;
      }
      // Set in_ops, out_ops, inputs, outputs for transAddition op
      NPUPassUtils::UpdateOp(post_trans_op, {op}, {}, post_trans_op->inputs(), post_trans_op->outputs());
      trans_ops->push_back(post_trans_op);

      // for those non-insert post ops, update their in_tensor
      for (auto non_insert_op : post_non_insert_ops) {
        auto inputs = non_insert_op->inputs();
        std::replace(inputs.begin(), inputs.end(), cur_output, *nchw_tensor);
        non_insert_op->set_inputs(inputs);
      }
      // update origin op's out tensor and out op
      NPUPassUtils::UpdateNC2NHTransNodePreOp(op, *trans_ops, {});
    }
  }
  return RET_OK;
}

int NPUTransAdditionPass::Run(NPUGraph *subgraph) {
  all_ops_ = subgraph->GetOps();
  all_tensors_ = subgraph->GetInsertTensors();
  auto graph_outputs = subgraph->outputs();
  for (size_t i = 0; i < all_ops_->size();) {
    auto op = (*all_ops_)[i];
    if (op->type() != schema::PrimitiveType_Transpose && !op->out_ops().empty()) {
      // insert post trans for the graph output comes from the branch
      std::vector<NPUOp *> post_ops;
      auto ret = InsertTransNodes(op, &post_ops, graph_outputs);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert nchw2nhwc op after op " << op->name() << " failed.";
        return RET_ERROR;
      }
      if (!post_ops.empty()) {
        all_ops_->insert(all_ops_->begin() + i, post_ops.begin(), post_ops.end());
        i += post_ops.size();
        continue;
      }
    }
    i++;
  }
  return RET_OK;
}
}  // namespace mindspore
