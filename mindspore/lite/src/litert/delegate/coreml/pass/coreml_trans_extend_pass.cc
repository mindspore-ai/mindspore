/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "src/litert/delegate/coreml/pass/coreml_trans_extend_pass.h"
#include <algorithm>
#include <set>
#include <string>
#include "src/litert/delegate/coreml/pass/coreml_pass_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
std::set<mindspore::schema::PrimitiveType> format_depend_nodes = {
  schema::PrimitiveType_Conv2DFusion,  schema::PrimitiveType_Conv2dTransposeFusion,
  schema::PrimitiveType_MaxPoolFusion, schema::PrimitiveType_AvgPoolFusion,
  schema::PrimitiveType_CropAndResize, schema::PrimitiveType_InstanceNorm,
  schema::PrimitiveType_ArgMaxFusion,  schema::PrimitiveType_FullConnection,
  schema::PrimitiveType_ScaleFusion,   schema::PrimitiveType_ExpandDims,
  schema::PrimitiveType_Unsqueeze,     schema::PrimitiveType_SliceFusion,
  schema::PrimitiveType_BroadcastTo,   schema::PrimitiveType_TileFusion,
  schema::PrimitiveType_Resize,        schema::PrimitiveType_MatMulFusion,
  schema::PrimitiveType_Gather,        schema::PrimitiveType_Gather,
  schema::PrimitiveType_Squeeze,       schema::PrimitiveType_Reshape,
  schema::PrimitiveType_Unsqueeze,     schema::PrimitiveType_Transpose,
};

// this pass goal is to minimize subgraphs generated
// by inserting nchw2nhwc or nhwc2nchw before or after the operator (e.g. concat, add, etc..) together with
// fusion pass. If transpose inserted are more than half of input output, we will insert remaining input
// output with transpose and hopefully do a fusion pass. Otherwise, we don't insert anything.

// Typically concat accept output from nchw2nhwc, we fill other input with nh2nc and nc2nh so that inputs to concat are
// format same and then fusion all nchw2nhwc op.
// e.g.
// original     (conv->nchw2nhwc, add(format nhwc)) -> concat-> (nhwc2nchw->conv)
// current pass (conv->nchw2nhwc, add->nhwc2nchw->nchw2nhwc) -> concat -> (nhwc2nchw->conv)
// fusion pass  (conv, add->nhwc2nchw) -> concat -> conv
// original 2 cpusubgraph, after 2 pass, only 1 cpu subgraph

// Such ops require inputs all have same format, could be nchw or nhwc or other format.
// Their inputs outputs may not be 4d, or are already format ok,
// so we won't insert nc2nh or nh2nc when op's in ops and out ops contains no nc2nh or nh2nc.
// This pass should be run after npu_transform_pass, which insert transpose for nchw-input-limited op like conv2d.

InsertState CoreMLTransExtendPass::GetInsertState(CoreMLOp *op) {
  // filter out irrelevant op
  if (format_depend_nodes.find(op->type()) != format_depend_nodes.end()) {
    return InsertState::InsertNone;
  }
  // current op is target op
  // Use out ops to count the out lines from current op since a single tensor can be used by multiple out ops. Besides,
  // a tensor can be used by out ops and graph output at the same time, there will be one more line in this case.
  std::vector<mindspore::MSTensor> inputs = CoreMLPassUtils::GetNonConstInputs(op);
  size_t in_out_tensor_num =
    inputs.size() + std::max(std::max(op->out_ops().size(), static_cast<size_t>(1)), op->outputs().size());
  size_t transpose_input_num = 0;
  size_t transpose_output_num = 0;
  size_t graph_input_num = 0;
  size_t graph_output_num = 0;
  bool need_pre_insert = false;
  bool need_post_insert = false;
  // count number of input tensor from nc2nh and output tensor to nh2nc
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto in_op = CoreMLPassUtils::OpInputFromOp(op, inputs.at(i));
    if (CoreMLPassUtils::IsNchw2Nhwc(in_op)) {
      transpose_input_num++;
    } else {
      need_pre_insert = true;
    }
    if (in_op == nullptr) {
      graph_input_num++;
    }
  }
  auto graph_output = subgraph_->outputs();
  for (auto output : op->outputs()) {
    if (std::find(graph_output.begin(), graph_output.end(), output) != graph_output.end()) {
      graph_output_num++;
      need_post_insert = true;
    }
  }
  for (const auto out_op : op->out_ops()) {
    for (auto out_op_input : out_op->inputs()) {
      if (std::find(graph_output.begin(), graph_output.end(), out_op_input) != graph_output.end()) {
        in_out_tensor_num++;
      }
    }
    if (CoreMLPassUtils::IsNhwc2Nchw(out_op)) {
      transpose_output_num++;
    } else {
      need_post_insert = true;
    }
  }

  // won't insert any thing if num of transpose tensor is smaller than half of total op inputs and op outputs, unless
  // current op is the graph input or output op, since we should avoid to build a single op subgraph in this case.
  // won't insert if total input output are all transpose tensor, the fusion pass will handle this.
  size_t transpose_tensor_num = transpose_input_num + transpose_output_num;
  size_t connected_in_out_tensor_num = in_out_tensor_num - graph_output_num - graph_input_num;
  if (transpose_tensor_num == 0 || transpose_tensor_num * REPEAT_TIMES2 < connected_in_out_tensor_num ||
      transpose_tensor_num == in_out_tensor_num) {
    return InsertState::InsertNone;
  }
  InsertState ret = (need_pre_insert && need_post_insert)
                      ? InsertState::BothInsert
                      : (need_pre_insert ? InsertState::PreInsert
                                         : (need_post_insert ? InsertState::PostInsert : InsertState::InsertNone));

  return ret;
}

int CoreMLTransExtendPass::InsertTransNode(CoreMLOp *op, CoreMLOp *post_op, const mindspore::MSTensor &trans_in_tensor,
                                           std::vector<CoreMLOp *> *trans_ops) {
  MS_ASSERT(op != nullptr || post_op != nullptr);
  std::string op_name;
  std::vector<CoreMLOp *> in_ops;
  std::vector<CoreMLOp *> out_ops;
  if (op != nullptr) {
    op_name = op->name() + "_post";
    in_ops.emplace_back(op);
  }
  if (post_op != nullptr) {
    op_name = post_op->name() + "_pre";
    out_ops.emplace_back(post_op);
  }
  auto nhwc_shape = trans_in_tensor.Shape();
  std::vector<int64_t> nchw_shape = {nhwc_shape[kNHWC_N], nhwc_shape[kNHWC_C], nhwc_shape[kNHWC_H],
                                     nhwc_shape[kNHWC_W]};

  auto nh2nc_name = op_name + "_nh2nc_" + std::to_string(total++);
  auto nh2nc_tensor =
    mindspore::MSTensor::CreateTensor(nh2nc_name + "/output0", trans_in_tensor.DataType(), nchw_shape, nullptr, 0);
  if (nh2nc_tensor == nullptr) {
    MS_LOG(ERROR) << "New nchw tensor failed when inserting nchw2nhwc op.";
    return RET_ERROR;
  }
  nh2nc_tensor->SetFormat(Format::NCHW);
  std::vector<mindspore::MSTensor> nh2nc_tensors = {*nh2nc_tensor};
  all_tensors_->push_back(nh2nc_tensor);

  auto nc2nh_name = op_name + "_nc2nh_" + std::to_string(total++);
  auto nc2nh_tensor =
    mindspore::MSTensor::CreateTensor(nc2nh_name + "/output0", trans_in_tensor.DataType(), nhwc_shape, nullptr, 0);
  if (nc2nh_tensor == nullptr) {
    MS_LOG(ERROR) << "New nhwc tensor failed when inserting nhwc2nchw op.";
    return RET_ERROR;
  }
  nc2nh_tensor->SetFormat(Format::NHWC);
  std::vector<mindspore::MSTensor> nc2nh_tensors = {*nc2nh_tensor};
  all_tensors_->push_back(nc2nh_tensor);

  auto *nh2nc_op = CoreMLPassUtils::CreateNhwc2NchwOp({trans_in_tensor}, nh2nc_tensors, nh2nc_name);
  trans_ops->push_back(nh2nc_op);

  auto *nc2nh_op = CoreMLPassUtils::CreateNchw2NhwcOp(nh2nc_tensors, nc2nh_tensors, nc2nh_name);
  trans_ops->push_back(nc2nh_op);

  CoreMLPassUtils::UpdateOp(nh2nc_op, in_ops, {nc2nh_op}, {trans_in_tensor}, nh2nc_tensors);
  CoreMLPassUtils::UpdateOp(nc2nh_op, {nh2nc_op}, out_ops, {nh2nc_tensors[0]}, nc2nh_tensors);
  if (op != nullptr) {
    CoreMLPassUtils::UpdateNH2NCTransNodePreOp(op, nh2nc_op, post_op);
  }
  if (post_op != nullptr) {
    CoreMLPassUtils::UpdateNC2NHTransNodePostOp(op, nc2nh_op, post_op, trans_in_tensor);
  } else {
    // post_op nullptr mean output, we remain graph output tensor name unchanged
    auto graph_output_name = trans_in_tensor.Name();
    nc2nh_tensor->SetTensorName(graph_output_name + "_after_" + name_);
  }
  return RET_OK;
}

int CoreMLTransExtendPass::InsertPreNodes(CoreMLOp *op, std::vector<CoreMLOp *> *trans_ops) {
  int ret = RET_OK;
  auto inputs = CoreMLPassUtils::GetNonConstInputs(op);
  for (auto tensor : inputs) {
    if (tensor.Shape().size() < COMM_SHAPE_SIZE) {
      continue;
    }
    // the input tensor can only come from a single op
    auto pre_op = CoreMLPassUtils::OpInputFromOp(op, tensor);
    if (CoreMLPassUtils::IsNchw2Nhwc(pre_op)) {
      continue;
    }
    // if this tensor is input of graph, pre_op is nullptr.;
    ret = InsertTransNode(pre_op, op, tensor, trans_ops);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
      return ret;
    }
  }
  return ret;
}

int CoreMLTransExtendPass::InsertPostNodes(CoreMLOp *op, std::vector<CoreMLOp *> *trans_ops) {
  int ret = RET_OK;
  for (size_t idx = 0; idx < op->outputs().size(); idx++) {
    auto out_tensor = op->outputs().at(idx);
    if (out_tensor.Shape().size() < COMM_SHAPE_SIZE) {
      continue;
    }
    if (std::find(subgraph_->outputs().begin(), subgraph_->outputs().end(), out_tensor) != subgraph_->outputs().end()) {
      // the case that op's out tensor is graph output
      ret = InsertTransNode(op, nullptr, op->outputs().at(idx), trans_ops);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
        return RET_ERROR;
      }
      // use origin output as the last trans op's output in order to avoid the lost of the output tensor after transpose
      // fusion. The input of the cur_op's out_op will be updated in the loop below.
      auto last_trans = trans_ops->back();
      auto trans_output = last_trans->outputs();
      auto cur_outputs = op->outputs();
      cur_outputs[idx] = last_trans->outputs()[0];
      trans_output[0] = op->outputs()[idx];
      last_trans->set_outputs(trans_output);
      op->set_outputs(cur_outputs);
    }

    // besides of being as graph outputs, the output tensors also can connected with multiple ops.
    for (auto post_op : op->out_ops()) {
      auto post_op_input = post_op->inputs();
      auto it = std::find(post_op_input.begin(), post_op_input.end(), out_tensor);
      if (it == post_op_input.end()) {
        continue;
      }
      auto related_idx = it - post_op_input.begin();
      post_op_input[related_idx] = op->outputs().at(idx);
      post_op->set_inputs(post_op_input);

      if (CoreMLPassUtils::IsNhwc2Nchw(post_op)) {
        continue;
      }
      // the case that op's out tensor is one of post_op's input tensor
      ret = InsertTransNode(op, post_op, op->outputs().at(idx), trans_ops);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
        return ret;
      }
    }
  }
  return ret;
}

int CoreMLTransExtendPass::Run(CoreMLGraph *subgraph) {
  subgraph_ = subgraph;
  all_ops_ = subgraph_->GetOps();
  all_tensors_ = subgraph_->GetInsertTensors();
  std::vector<CoreMLOp *> insert_ops;
  for (int j = 0; j < REPEAT_TIMES2; ++j) {
    for (size_t i = 0; i < all_ops_->size(); i++) {
      auto op = (*all_ops_)[i];
      auto insert_state = GetInsertState(op);
      insert_ops.clear();
      // If the every output op is nhwc2nchw, insert
      // modify loop index add post_ops.size() to the next op in the origin vector
      switch (insert_state) {
        case InsertState::PreInsert: {
          auto ret = InsertPreNodes(op, &insert_ops);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops_->insert(all_ops_->begin() + i, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        case InsertState::PostInsert: {
          auto ret = InsertPostNodes(op, &insert_ops);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops_->insert(all_ops_->begin() + i + 1, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        case InsertState::BothInsert: {
          auto ret = InsertPreNodes(op, &insert_ops);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops_->insert(all_ops_->begin() + i, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();

          insert_ops.clear();
          ret = InsertPostNodes(op, &insert_ops);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops_->insert(all_ops_->begin() + i + 1, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        default:
          MS_LOG(DEBUG) << "Insert Nothing on op " << op->name();
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
