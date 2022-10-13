/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/delegate/npu/pass/npu_fusion_pass.h"
#include <set>
#include <vector>
#include "src/runtime/delegate/npu/pass/npu_pass_utils.h"
#include "src/runtime/delegate/npu/npu_converter_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
bool CheckFusion(NPUOp *cur_op, const std::vector<mindspore::MSTensor> &graph_outputs) {
  if (cur_op->in_ops().empty() || cur_op->out_ops().empty()) {
    return false;
  }
  auto pre_flag = std::all_of(cur_op->in_ops().begin(), cur_op->in_ops().end(), [](NPUOp *in_op) {
    return NPUPassUtils::IsNchw2Nhwc(in_op) && in_op->out_ops().size() == 1;
  });
  if (!pre_flag) {
    return false;
  }
  auto post_flag = std::all_of(cur_op->out_ops().begin(), cur_op->out_ops().end(),
                               [](NPUOp *out_op) { return NPUPassUtils::IsNhwc2Nchw(out_op); });
  if (!post_flag) {
    return false;
  }
  for (auto out_op : cur_op->out_ops()) {
    // If the pattern is "nc2nh->cur_op->nh2nc" while the output tensors of "cur_op" and "nh2nc" are both graph output,
    // the trans ops can not be fused since it will cause the missing of graph output.
    if (out_op->out_ops().empty() &&
        std::find(graph_outputs.begin(), graph_outputs.end(), out_op->inputs().at(0)) != graph_outputs.end()) {
      return false;
    }
  }
  return true;
}

bool CheckFormatFusion(NPUOp *cur_op) {
  if (cur_op->out_ops().empty()) {
    return false;
  }
  if (NPUPassUtils::IsNhwc2Nchw(cur_op)) {
    return std::all_of(cur_op->out_ops().begin(), cur_op->out_ops().end(),
                       [](NPUOp *cur_op) { return NPUPassUtils::IsNchw2Nhwc(cur_op); });
  }
  if (NPUPassUtils::IsNchw2Nhwc(cur_op)) {
    return std::all_of(cur_op->out_ops().begin(), cur_op->out_ops().end(),
                       [](NPUOp *cur_op) { return NPUPassUtils::IsNhwc2Nchw(cur_op); });
  }
  return false;
}

void NPUFusionPass::RemoveAndFreeOp(NPUOp *cur_op) {
  auto itr = find(all_ops_->begin(), all_ops_->end(), cur_op);
  if (itr != all_ops_->end()) {
    all_ops_->erase(itr);
  }
  delete cur_op;
}

int NPUFusionPass::UpdatePreOps(NPUOp *cur_op) {
  auto cur_in_ops = cur_op->in_ops();
  std::set<NPUOp *> has_visited;
  for (auto in_op : cur_op->in_ops()) {
    if (has_visited.find(in_op) != has_visited.end()) {
      continue;
    }
    // graph in op
    if (in_op->in_ops().empty()) {
      cur_in_ops.erase(find(cur_in_ops.begin(), cur_in_ops.end(), in_op));
    } else {
      auto pre_op = in_op->in_ops()[0];
      auto pre_out_ops = pre_op->out_ops();
      for (size_t i = 0; i < pre_out_ops.size(); i++) {
        if (pre_out_ops[i] == in_op) {
          pre_out_ops[i] = cur_op;
        }
      }
      pre_op->set_out_ops(pre_out_ops);

      for (size_t i = 0; i < cur_in_ops.size(); i++) {
        if (cur_in_ops[i] == in_op) {
          cur_in_ops[i] = pre_op;
        }
      }
    }
    RemoveAndFreeOp(in_op);
    (void)has_visited.insert(in_op);
  }
  cur_op->set_in_ops(cur_in_ops);
  return RET_OK;
}

int NPUFusionPass::UpdatePostOps(NPUOp *cur_op) {
  auto cur_out_ops = cur_op->out_ops();
  std::set<NPUOp *> has_visited;
  for (auto out_op : cur_op->out_ops()) {
    if (has_visited.find(out_op) != has_visited.end()) {
      continue;
    }
    // graph out op
    if (out_op->out_ops().empty()) {
      cur_out_ops.erase(find(cur_out_ops.begin(), cur_out_ops.end(), out_op));
    } else {
      auto post_op = out_op->out_ops()[0];
      auto post_in_ops = post_op->in_ops();
      for (size_t i = 0; i < post_in_ops.size(); i++) {
        if (post_in_ops[i] == out_op) {
          post_in_ops[i] = cur_op;
        }
      }
      post_op->set_in_ops(post_in_ops);

      for (size_t i = 0; i < cur_out_ops.size(); i++) {
        if (cur_out_ops[i] == out_op) {
          cur_out_ops[i] = post_op;
        }
      }
    }
    RemoveAndFreeOp(out_op);
    (void)has_visited.insert(out_op);
  }
  cur_op->set_out_ops(cur_out_ops);
  return RET_OK;
}

int UpdatePreTensors(NPUOp *cur_op) {
  auto in_tensors_vec = cur_op->inputs();
  for (auto in_op : cur_op->in_ops()) {
    if (in_op->inputs().empty() || in_op->outputs().empty()) {
      MS_LOG(ERROR) << "in_tensors or out_tensors of input op is empty.";
      return RET_ERROR;
    }
    mindspore::MSTensor cur_tensor;
    auto in_tensor = in_op->inputs()[0];
    auto out_tensor = in_op->outputs()[0];
    if (!in_op->in_ops().empty()) {
      auto pre_op = in_op->in_ops()[0];
      for (size_t i = 0; i < pre_op->outputs().size(); i++) {
        if (pre_op->outputs()[i] == in_tensor) {
          cur_tensor = pre_op->outputs()[i];
          break;
        }
      }
    } else {
      // graph input
      cur_tensor = in_tensor;
    }

    for (size_t i = 0; i < in_tensors_vec.size(); i++) {
      if (in_tensors_vec[i] == out_tensor) {
        in_tensors_vec[i] = cur_tensor;
      }
    }
  }
  cur_op->set_inputs(in_tensors_vec);
  return RET_OK;
}

int UpdatePostTensors(NPUOp *cur_op) {
  mindspore::MSTensor new_post_input;
  for (auto out_op : cur_op->out_ops()) {
    auto in_tensor = out_op->inputs()[0];
    auto out_tensor = out_op->outputs()[0];
    auto nhwc_shape = in_tensor.Shape();
    if (in_tensor.format() == Format::NHWC) {
      MS_CHECK_TRUE_MSG(nhwc_shape.size() == NPU_SHAPE_SIZE, RET_ERROR, "Invalid transpose dim size!");
      in_tensor.SetShape({nhwc_shape[NHWC_N], nhwc_shape[NHWC_C], nhwc_shape[NHWC_H], nhwc_shape[NHWC_W]});
      in_tensor.SetFormat(Format::NCHW);
    }
    // out_op is a graph output op
    if (out_op->out_ops().empty()) {
      auto out_tensors_vec = cur_op->outputs();
      for (size_t i = 0; i < out_tensors_vec.size(); i++) {
        if (out_tensors_vec[i] == in_tensor) {
          out_tensors_vec[i] = out_op->outputs()[0];
        }
      }
      cur_op->set_outputs(out_tensors_vec);
      // exist other out_ops using the same tensor as the current out_op, note that the other out_op has likely been
      // updated, which mean it may be not a Transpose op anymore.
      for (auto other_out_op : cur_op->out_ops()) {
        auto other_in_tensors_vec = other_out_op->inputs();
        for (size_t i = 0; i < other_in_tensors_vec.size(); i++) {
          if (other_in_tensors_vec[i] == in_tensor) {
            other_in_tensors_vec[i] = out_op->outputs()[0];
          }
        }
        other_out_op->set_inputs(other_in_tensors_vec);
      }
    }
    // out_op is not a graph out op
    for (auto post_op : out_op->out_ops()) {
      auto in_tensors_vec = post_op->inputs();
      for (size_t i = 0; i < in_tensors_vec.size(); i++) {
        if (in_tensors_vec[i] == out_tensor) {
          in_tensors_vec[i] = in_tensor;
        }
      }
      post_op->set_inputs(in_tensors_vec);
    }
  }
  return RET_OK;
}

int NPUFusionPass::UpdateOp(NPUOp *cur_op) {
  if (cur_op == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return RET_ERROR;
  }
  auto ret = UpdatePreTensors(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePreTensors failed.";
    return RET_ERROR;
  }
  ret = UpdatePostTensors(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePostTensors failed.";
    return RET_ERROR;
  }
  ret = UpdatePreOps(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePreOps failed.";
    return RET_ERROR;
  }
  ret = UpdatePostOps(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePostOps failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int NPUFusionPass::CommonFusion(NPUOp *cur_op) {
  if (cur_op == nullptr) {
    return RET_ERROR;
  }
  auto ret = UpdateOp(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateOp failed.";
    return RET_ERROR;
  }
  auto all_tensors = subgraph_->GetInsertTensors();
  ret = cur_op->HandleAxisAndConstantInputs(all_tensors);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HandleAxisAndConstantInputs failed.";
    return ret;
  }
  return RET_OK;
}

void UpdateOutOpsOfPreOp(NPUOp *cur_op, bool found_graph_out_tensor, const mindspore::MSTensor &graph_out_tensor,
                         const std::vector<NPUOp *> &pre_insert_ops) {
  MS_ASSERT(cur_op != nullptr);
  auto is_graph_input = cur_op->in_ops().empty();
  auto cur_op_in_tensor = cur_op->inputs()[0];
  if (!is_graph_input) {
    auto pre_op = cur_op->in_ops()[0];
    auto pre_out_ops = pre_op->out_ops();
    size_t cur_op_index = 0;
    for (size_t index = 0; index < pre_out_ops.size(); index++) {
      if (pre_out_ops[index] == cur_op) {
        pre_out_ops.erase(pre_out_ops.begin() + index);
        cur_op_index = index;
        index--;
      } else if (found_graph_out_tensor) {
        // only in this case, the output of pre_op is specified to 2nd trans op's output and pre_out_ops need update.
        auto tensors_vec = pre_out_ops[index]->inputs();
        for (size_t i = 0; i < tensors_vec.size(); i++) {
          if (tensors_vec[i] == cur_op_in_tensor) {
            tensors_vec[i] = graph_out_tensor;
            break;
          }
        }
        pre_out_ops[index]->set_inputs(tensors_vec);
      }
    }
    pre_out_ops.insert(pre_out_ops.begin() + cur_op_index, pre_insert_ops.begin(), pre_insert_ops.end());
    pre_op->set_out_ops(pre_out_ops);
  }
  return;
}

int NPUFusionPass::FormatFusion(NPUOp *cur_op) {
  CHECK_NULL_RETURN(cur_op);
  auto is_graph_input = cur_op->in_ops().empty();
  auto cur_op_in_tensor = cur_op->inputs()[0];
  std::vector<NPUOp *> pre_insert_ops;
  NPUOp *pre_op = nullptr;
  if (!is_graph_input) {
    pre_op = cur_op->in_ops()[0];
  }
  mindspore::MSTensor graph_out_tensor;
  bool found_graph_out_tensor = false;
  auto graph_outputs = subgraph_->outputs();
  // if the output of second trans op(s) is graph output, find it out and use it as the pre-op's output.
  for (const auto &sec_op : cur_op->out_ops()) {
    if (std::find(graph_outputs.begin(), graph_outputs.end(), sec_op->outputs()[0]) != graph_outputs.end()) {
      graph_out_tensor = sec_op->outputs()[0];
      if (!is_graph_input) {
        found_graph_out_tensor = true;
        // cur_op is the first trans op, it's input op num and input tensor num must be 1
        pre_op->set_outputs({graph_out_tensor});
        // in fp16 mode, tensor data type fp16 need to be changed back.
        auto tensor = pre_op->outputs()[0];
        if (tensor.DataType() == DataType::kNumberTypeFloat16) {
          tensor.SetDataType(DataType::kNumberTypeFloat32);
        }
        break;
      } else {
        MS_LOG(WARNING) << "Existing graph output equivalent to graph input, which is unsupported now.";
        return RET_OK;
      }
    }
  }
  for (const auto &trans_op : cur_op->out_ops()) {
    for (const auto &post_op : trans_op->out_ops()) {
      // update tensor
      auto tensors_vec = post_op->inputs();
      for (size_t i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == trans_op->outputs()[0]) {
          tensors_vec[i] = found_graph_out_tensor ? graph_out_tensor : cur_op_in_tensor;
          break;
        }
      }
      post_op->set_inputs(tensors_vec);

      // update op
      auto post_in_ops = post_op->in_ops();
      for (size_t i = 0; i < post_in_ops.size(); i++) {
        if (post_in_ops[i] == trans_op) {
          if (is_graph_input) {
            post_in_ops.erase(post_in_ops.begin() + i);
          } else {
            post_in_ops[i] = pre_op;
          }
          break;
        }
      }
      post_op->set_in_ops(post_in_ops);
      pre_insert_ops.push_back(post_op);
    }
    RemoveAndFreeOp(trans_op);
  }
  UpdateOutOpsOfPreOp(cur_op, found_graph_out_tensor, graph_out_tensor, pre_insert_ops);
  RemoveAndFreeOp(cur_op);
  return RET_OK;
}

int NPUFusionPass::Run(NPUGraph *subgraph) {
  subgraph_ = subgraph;
  all_ops_ = subgraph->GetOps();
  for (size_t i = 0; i < all_ops_->size(); i++) {
    auto cur_op = (*all_ops_)[i];
    auto ret = RET_OK;
    if (CheckFusion(cur_op, subgraph->outputs())) {
      i -= cur_op->in_ops().size();
      ret = CommonFusion(cur_op);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fusion failed.";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < all_ops_->size(); ++i) {
    auto cur_op = (*all_ops_)[i];
    if (CheckFormatFusion(cur_op)) {
      i--;
      auto ret = FormatFusion(cur_op);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "FormatFusion failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore
