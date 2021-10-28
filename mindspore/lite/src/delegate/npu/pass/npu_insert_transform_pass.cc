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
#include "src/delegate/npu/pass/npu_insert_transform_pass.h"
#include <algorithm>
#include <set>
#include <string>
#include "src/delegate/npu/pass/npu_pass_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
enum InsertState { InsertNone, PreInsert, PostInsert, BothInsert };
std::set<mindspore::schema::PrimitiveType> insert_nodes = {
  schema::PrimitiveType_Concat,       schema::PrimitiveType_AddFusion, schema::PrimitiveType_Eltwise,
  schema::PrimitiveType_Activation,   schema::PrimitiveType_Split,     schema::PrimitiveType_PadFusion,
  schema::PrimitiveType_StridedSlice, schema::PrimitiveType_MulFusion, schema::PrimitiveType_DivFusion};

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

int NPUInsertTransformPass::GetInsertState(NPUOp *op) {
  // filter out irrelevant op
  if (insert_nodes.find(op->type()) == insert_nodes.end()) {
    return InsertNone;
  }

  // current op is target op
  // use out ops to count how many out lines from current op
  std::vector<mindspore::MSTensor> inputs = NPUPassUtils::GetNonConstInputs(op);
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
    auto in_op = NPUPassUtils::OpInputFromOp(op, inputs.at(i));
    if (NPUPassUtils::IsNchw2Nhwc(in_op)) {
      transpose_input_num++;
    } else {
      need_pre_insert = true;
    }
    if (in_op == nullptr) {
      graph_input_num++;
    }
  }
  if (op->out_ops().empty()) {
    need_post_insert = true;
  }
  if (op->outputs().size() > op->out_ops().size()) {
    graph_output_num = op->outputs().size() - op->out_ops().size();
  }
  for (const auto out_op : op->out_ops()) {
    if (NPUPassUtils::IsNhwc2Nchw(out_op)) {
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
  if (transpose_tensor_num == 0 || transpose_tensor_num * 2 < connected_in_out_tensor_num ||
      transpose_tensor_num == in_out_tensor_num) {
    return InsertNone;
  }
  InsertState ret =
    (need_pre_insert && need_post_insert)
      ? BothInsert
      : ((need_pre_insert && !need_post_insert) ? PreInsert
                                                : ((!need_pre_insert && need_post_insert) ? PostInsert : InsertNone));

  return ret;
}

int NPUInsertTransformPass::InsertNode(NPUOp *op, NPUOp *post_op, size_t post_input_index,
                                       std::vector<NPUOp *> *trans_ops) {
  // Op and post_op can't be nullptr at the same time.
  std::string op_name;
  std::vector<mindspore::MSTensor> in_tensors;
  std::vector<NPUOp *> out_ops;
  // If post_op equals nullptr, op is the output of whole graph.
  if (post_op != nullptr) {
    out_ops.push_back(post_op);
    op_name = post_op->name() + "_pre";
    in_tensors.push_back(post_op->inputs().at(post_input_index));
  }
  std::vector<NPUOp *> in_ops;
  // If op equals nullptr, post_op is the input of whole graph.
  if (op != nullptr && !op->outputs().empty()) {
    in_ops.push_back(op);
    op_name = op->name() + "_post";
    in_tensors.resize(op->outputs().size());
    std::copy(op->outputs().begin(), op->outputs().end(), in_tensors.begin());
  }
  for (auto i = 0; i < in_tensors.size(); ++i) {
    auto in_tensor = in_tensors[i];
    auto nhwc_shape = in_tensor.Shape();
    if (nhwc_shape.size() == 0) {
      continue;
    } else if (nhwc_shape.size() < 4) {
      MS_LOG(ERROR) << "nhwc_shape size < " << 4;
      return RET_ERROR;
    }
    std::vector<int64_t> nchw_shape = {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};

    auto nh2nc_name = op_name + "_nh2nc_" + std::to_string(total++);
    auto nh2nc_tensor =
      mindspore::MSTensor::CreateTensor(nh2nc_name + "/output0", in_tensor.DataType(), nchw_shape, nullptr, 0);
    if (nh2nc_tensor == nullptr) {
      MS_LOG(ERROR) << "New nchw tensor failed when inserting nchw2nhwc op.";
      return RET_ERROR;
    }
    nh2nc_tensor->SetTensorName(nh2nc_name + "/output0");
    std::vector<mindspore::MSTensor> nh2nc_tensors = {*nh2nc_tensor};
    all_tensors_->push_back(nh2nc_tensor);

    auto nc2nh_name = op_name + "_nc2nh_" + std::to_string(total++);
    auto nc2nh_tensor =
      mindspore::MSTensor::CreateTensor(nc2nh_name + "/output0", in_tensor.DataType(), nhwc_shape, nullptr, 0);
    if (nc2nh_tensor == nullptr) {
      MS_LOG(ERROR) << "New nhwc tensor failed when inserting nhwc2nchw op.";
      return RET_ERROR;
    }
    std::vector<mindspore::MSTensor> nc2nh_tensors = {*nc2nh_tensor};
    all_tensors_->push_back(nc2nh_tensor);

    auto *nh2nc_op = NPUPassUtils::CreateNhwc2NchwOp({in_tensor}, nh2nc_tensors, nh2nc_name);
    trans_ops->push_back(nh2nc_op);

    auto *nc2nh_op = NPUPassUtils::CreateNchw2NhwcOp(nh2nc_tensors, nc2nh_tensors, nc2nh_name);
    trans_ops->push_back(nc2nh_op);

    NPUPassUtils::UpdateOp(nh2nc_op, in_ops, {nc2nh_op}, {in_tensor}, nh2nc_tensors);
    NPUPassUtils::UpdateOp(nc2nh_op, {nh2nc_op}, out_ops, {nh2nc_tensors[0]}, nc2nh_tensors);
    if (op != nullptr) {
      NPUPassUtils::UpdateNH2NCTransNodePreOp(op, nh2nc_op, post_op);
    }
    if (post_op != nullptr) {
      NPUPassUtils::UpdateNC2NHTransNodePostOp(op, nc2nh_op, post_op);
    } else {
      // post_op nullptr mean output, we remain graph output tensor name unchanged
      auto graph_output_name = in_tensor.Name();
      nc2nh_tensor->SetTensorName(graph_output_name + "_after_" + name_);
    }
  }
  return RET_OK;
}

int NPUInsertTransformPass::InsertForInputTensor(NPUOp *op, size_t in_tensor_index, NPUOp *pre_op,
                                                 std::vector<NPUOp *> *trans_ops) {
  // insert transpose nodes before target ops
  return InsertNode(pre_op, op, in_tensor_index, trans_ops);
}

int NPUInsertTransformPass::InsertForOutputTensor(NPUOp *op, NPUOp *post_op, size_t post_in_tensor_index,
                                                  std::vector<NPUOp *> *trans_ops) {
  // insert transpose nodes after target ops
  return InsertNode(op, post_op, post_in_tensor_index, trans_ops);
}

int NPUInsertTransformPass::InsertPreNodes(NPUOp *op, std::vector<NPUOp *> *trans_ops) {
  int ret = RET_OK;
  auto inputs = NPUPassUtils::GetNonConstInputs(op);
  for (auto tensor : inputs) {
    auto pre_op = NPUPassUtils::OpInputFromOp(op, tensor);
    if (NPUPassUtils::IsNchw2Nhwc(pre_op)) {
      continue;
    }
    // if this tensor is input of graph, pre_op is nullptr.
    auto it = find(op->inputs().begin(), op->inputs().end(), tensor);
    if (it == op->inputs().end()) {
      MS_LOG(ERROR) << "Find in tensor index error";
      return RET_ERROR;
    }
    size_t index = it - op->inputs().begin();
    ret = InsertForInputTensor(op, index, pre_op, trans_ops);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
      return ret;
    }
  }
  return ret;
}

int NPUInsertTransformPass::InsertPostNodes(NPUOp *op, std::vector<NPUOp *> *trans_ops) {
  int ret = RET_OK;

  for (const auto post_op : op->out_ops()) {
    if (NPUPassUtils::IsNhwc2Nchw(post_op)) {
      continue;
    }
    auto post_op_in_tensors = post_op->inputs();
    // op's out tensor is one of post_op's input tensor
    auto it = std::find(post_op_in_tensors.begin(), post_op_in_tensors.end(), op->outputs().at(0));
    if (it == post_op_in_tensors.end()) {
      return RET_ERROR;
    }
    size_t input_index = it - post_op_in_tensors.begin();
    ret = InsertForOutputTensor(op, post_op, input_index, trans_ops);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
      return ret;
    }
  }
  if (op->outputs().size() > op->out_ops().size()) {
    // op out is graph output
    ret = InsertForOutputTensor(op, nullptr, 0, trans_ops);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
      return ret;
    }
  }
  return ret;
}

int NPUInsertTransformPass::Run(NPUGraph *subgraph) {
  all_ops_ = subgraph->GetOps();
  all_tensors_ = subgraph->GetInsertTensors();
  std::vector<NPUOp *> insert_ops;
  for (int j = 0; j < 2; ++j) {
    for (size_t i = 0; i < all_ops_->size(); i++) {
      auto op = (*all_ops_)[i];
      auto insert_state = GetInsertState(op);
      insert_ops.clear();
      // If the every output op is nhwc2nchw, insert
      // modify loop index add post_ops.size() to the next op in the origin vector
      switch (insert_state) {
        case PreInsert: {
          auto ret = InsertPreNodes(op, &insert_ops);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops_->insert(all_ops_->begin() + i, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        case PostInsert: {
          auto ret = InsertPostNodes(op, &insert_ops);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops_->insert(all_ops_->begin() + i + 1, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        case BothInsert: {
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
}  // namespace mindspore
