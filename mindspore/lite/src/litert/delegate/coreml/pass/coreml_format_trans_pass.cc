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
#include "src/litert/delegate/coreml/pass/coreml_format_trans_pass.h"
#include <vector>
#include "src/litert/delegate/coreml/pass/coreml_pass_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
std::set<mindspore::schema::PrimitiveType> nchw_nodes = {
  schema::PrimitiveType_Conv2DFusion,  schema::PrimitiveType_Conv2dTransposeFusion, schema::PrimitiveType_Resize,
  schema::PrimitiveType_MaxPoolFusion, schema::PrimitiveType_AvgPoolFusion,         schema::PrimitiveType_ScaleFusion,
  schema::PrimitiveType_CropAndResize, schema::PrimitiveType_InstanceNorm};

int CoreMLFormatTransPass::InsertPreNodes(CoreMLOp *op, std::vector<CoreMLOp *> *trans_ops) {
  bool is_input_op = op->in_ops().empty();
  // not always single input (like CropAndResize), but we care about the input with 4d.
  auto it = std::find_if(op->in_ops().begin(), op->in_ops().end(), [](CoreMLOp *k) {
    return k->outputs().size() > 0 && k->outputs()[0].Shape().size() == COMM_SHAPE_SIZE;
  });
  if (!is_input_op && it == op->in_ops().end()) {
    MS_LOG(ERROR) << "CoreML Transform pass does not find in op with 4d output";
    return RET_ERROR;
  }
  if (is_input_op || nchw_nodes.find((*it)->type()) == nchw_nodes.end()) {
    CoreMLOp *pre_op = nullptr;
    if (!is_input_op) {
      pre_op = *it;
    }

    // Create pre transform op's out tensor.
    auto name = op->name() + "_pre_trans" + "_Nhwc2Nchw_" + std::to_string(total++);
    auto nhwc_shape = op->inputs()[0].Shape();
    std::vector<int64_t> nchw_shape = {nhwc_shape[kNHWC_N], nhwc_shape[kNHWC_C], nhwc_shape[kNHWC_H],
                                       nhwc_shape[kNHWC_W]};
    auto tensor =
      mindspore::MSTensor::CreateTensor(name + "/output0", op->inputs()[0].DataType(), nchw_shape, nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "New nchw tensor failed when inserting pre nhwc2nchw op.";
      return RET_ERROR;
    }
    tensor->SetFormat(Format::NCHW);
    std::vector<mindspore::MSTensor> pre_trans_outputs = {*tensor};
    all_tensors_->push_back(tensor);

    // Create pre transform op: Nhwc2Nchw
    auto *trans_op = CoreMLPassUtils::CreateNhwc2NchwOp({op->inputs()[0]}, pre_trans_outputs, name);
    if (trans_op == nullptr) {
      MS_LOG(ERROR) << "Create Nhwc2Nchw transpose op failed.";
      return RET_ERROR;
    }
    trans_ops->push_back(trans_op);

    // Set in_ops, out_ops, inputs, outputs for transform op
    std::vector<CoreMLOp *> pre_trans_in_ops;
    if (!is_input_op) {
      pre_trans_in_ops = {pre_op};
    }
    CoreMLPassUtils::UpdateOp(trans_op, pre_trans_in_ops, {op}, trans_op->inputs(), pre_trans_outputs);

    if (pre_op != nullptr) {
      CoreMLPassUtils::UpdateNH2NCTransNodePreOp(pre_op, trans_op, op);
    }
    CoreMLPassUtils::UpdateNH2NCTransNodePostOp(trans_op, op);
  }
  return RET_OK;
}

int CoreMLFormatTransPass::InsertPostNodes(CoreMLOp *op, std::vector<CoreMLOp *> *trans_ops) {
  bool is_output_op = false;
  if (op->out_ops().empty() ||
      find(subgraph_->outputs().begin(), subgraph_->outputs().end(), op->outputs()[0]) != subgraph_->outputs().end()) {
    is_output_op = true;
  }
  // Get the post op that need insert trans op.
  // If no need for inserting trans op, the post op must be coreml and in trans_nodes.
  std::vector<CoreMLOp *> post_insert_ops;
  std::vector<CoreMLOp *> post_non_insert_ops;
  for (int i = 0; i < op->out_ops().size(); i++) {
    auto post_op = op->out_ops()[i];
    if (nchw_nodes.find(post_op->type()) == nchw_nodes.end()) {
      post_insert_ops.push_back(post_op);
    } else {
      post_non_insert_ops.push_back(post_op);
    }
  }
  if (!is_output_op && post_insert_ops.empty()) {
    return RET_OK;
  }
  // Create post transform op's in tensor.
  auto name = op->name() + "_post_trans" + "_Nchw2Nhwc" + std::to_string(total++);

  auto nhwc_shape = op->outputs()[0].Shape();
  std::vector<int64_t> nchw_shape = {nhwc_shape[kNHWC_N], nhwc_shape[kNHWC_C], nhwc_shape[kNHWC_H],
                                     nhwc_shape[kNHWC_W]};
  auto nc2nh_tensor =
    mindspore::MSTensor::CreateTensor(name + "/input0", op->outputs()[0].DataType(), nchw_shape, nullptr, 0);
  if (nc2nh_tensor == nullptr) {
    MS_LOG(ERROR) << "New nchw tensor failed when inserting post nchw2nhwc op.";
    return RET_ERROR;
  }
  nc2nh_tensor->SetFormat(Format::NCHW);
  all_tensors_->push_back(nc2nh_tensor);

  if (is_output_op) {
    std::vector<mindspore::MSTensor> nc2nh_outputs{op->outputs().at(0)};
    // Create post transform op: Nchw2Nhwc
    auto *post_trans_op = CoreMLPassUtils::CreateNchw2NhwcOp({*nc2nh_tensor}, nc2nh_outputs, name);
    if (post_trans_op == nullptr) {
      MS_LOG(ERROR) << "Create Nchw2Nhwc transpose op failed.";
      return RET_ERROR;
    }
    // Set in_ops, out_ops, inputs, outputs for transform op
    CoreMLPassUtils::UpdateOp(post_trans_op, {op}, {}, post_trans_op->inputs(), post_trans_op->outputs());
    trans_ops->push_back(post_trans_op);
  }
  // for each to-be-insert out op, create one transpose op, one perm tensor, one out tensor
  // but using same one in_tensor.
  for (auto i = 0; i < post_insert_ops.size(); ++i) {
    auto post_insert_op = post_insert_ops.at(i);
    // nc2nh op out tensor: abandon original out_tensor, all ops use newly created out tensor.
    std::vector<mindspore::MSTensor> nc2nh_outputs{};
    auto origin_out_tensor = op->outputs().at(0);
    auto out_tensor_name = op->name() + "_post_trans" + "_Nchw2Nhwc_" + std::to_string(i) + "_out_tensor";
    auto out_tensor = mindspore::MSTensor::CreateTensor(out_tensor_name, origin_out_tensor.DataType(),
                                                        origin_out_tensor.Shape(), nullptr, 0);
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "New nhwc tensor failed when inserting post nchw2nhwc op.";
      return RET_ERROR;
    }
    out_tensor->SetFormat(Format::NHWC);
    all_tensors_->push_back(out_tensor);
    nc2nh_outputs.push_back(*out_tensor);

    // Create post transform op: Nchw2Nhwc
    auto *post_trans_op =
      CoreMLPassUtils::CreateNchw2NhwcOp({*nc2nh_tensor}, nc2nh_outputs, name + "_" + std::to_string(i));
    if (post_trans_op == nullptr) {
      MS_LOG(ERROR) << "Create Nchw2Nhwc transpose op failed.";
      return RET_ERROR;
    }
    // Set in_ops, out_ops, inputs, outputs for transform op
    CoreMLPassUtils::UpdateOp(post_trans_op, {op}, {post_insert_op}, post_trans_op->inputs(), post_trans_op->outputs());
    trans_ops->push_back(post_trans_op);
    // update post op inputs in_ops
    CoreMLPassUtils::UpdateNC2NHTransNodePostOp(op, post_trans_op, post_insert_op, origin_out_tensor);
  }
  // for those non-insert post ops, update their in_tensor
  for (auto non_insert_op : post_non_insert_ops) {
    auto inputs = non_insert_op->inputs();
    std::replace(inputs.begin(), inputs.end(), op->outputs().at(0), *nc2nh_tensor);
    non_insert_op->set_inputs(inputs);
  }
  // update origin op's out tensor and out op
  CoreMLPassUtils::UpdateNC2NHTransNodePreOp(op, *trans_ops, post_insert_ops);
  return RET_OK;
}

int CoreMLFormatTransPass::Run(CoreMLGraph *subgraph) {
  subgraph_ = subgraph;
  all_ops_ = subgraph_->GetOps();
  all_tensors_ = subgraph_->GetInsertTensors();
  for (size_t i = 0; i < all_ops_->size();) {
    auto op = (*all_ops_)[i];
    if (nchw_nodes.find(op->type()) == nchw_nodes.end()) {
      i++;
      continue;
    }
    if (op->type() == schema::PrimitiveType_InstanceNorm && op->inputs().front().format() == mindspore::Format::NCHW) {
      i++;
      continue;
    }
    // insert pre_ops before op in vector
    // modify loop index add (pre_ops.size() + 1) to the post_ops insert location
    std::vector<CoreMLOp *> pre_ops;
    auto ret = InsertPreNodes(op, &pre_ops);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw op before op " << op->name() << " failed.";
      return RET_ERROR;
    }
    all_ops_->insert(all_ops_->begin() + i, pre_ops.begin(), pre_ops.end());
    i += (pre_ops.size() + 1);

    // insert post_ops after op in vector
    // modify loop index add post_ops.size() to the next op in the origin vector
    std::vector<CoreMLOp *> post_ops;
    ret = InsertPostNodes(op, &post_ops);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nchw2nhwc op after op " << op->name() << " failed.";
      return RET_ERROR;
    }
    all_ops_->insert(all_ops_->begin() + i, post_ops.begin(), post_ops.end());
    i += post_ops.size();
  }
  return RET_OK;
}
}  // namespace mindspore::lite
