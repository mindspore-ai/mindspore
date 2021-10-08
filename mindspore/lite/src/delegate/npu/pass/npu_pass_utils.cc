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

#include "src/delegate/npu/pass/npu_pass_utils.h"
#include <algorithm>
#include "src/delegate/npu/op/scale_npu.h"
#include "src/delegate/npu/op/transpose_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
std::unordered_map<schema::PrimitiveType, std::set<int>> nodes2const_index{
  {schema::PrimitiveType_Split, {1}},
  {schema::PrimitiveType_PadFusion, {1}},
  {schema::PrimitiveType_StridedSlice, {1, 2, 3}}};

NPUOp *NPUPassUtils::CreateNchw2NhwcOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                       const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name) {
  std::vector<int> perm = {0, 2, 3, 1};
  auto npu_op = new (std::nothrow) TransposeNPUOp(in_tensors, out_tensors, perm, name);
  if (npu_op == nullptr) {
    MS_LOG(ERROR) << "New Nchw2Nhwc NPUOp failed.";
    return nullptr;
  }
  return npu_op;
}

NPUOp *NPUPassUtils::CreateNhwc2NchwOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                       const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name) {
  std::vector<int> perm = {0, 3, 1, 2};
  auto npu_op = new (std::nothrow) TransposeNPUOp(in_tensors, out_tensors, perm, name);
  if (npu_op == nullptr) {
    MS_LOG(ERROR) << "New Nhwc2Nchw NPUOp failed.";
    return nullptr;
  }
  return npu_op;
}

void NPUPassUtils::UpdateOp(NPUOp *op, const std::vector<NPUOp *> &in_ops, const std::vector<NPUOp *> &out_ops,
                            const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &outputs) {
  op->set_inputs(in_tensors);
  op->set_outputs(outputs);
  op->set_in_ops(in_ops);
  op->set_out_ops(out_ops);
}

void NPUPassUtils::UpdateNH2NCTransNodePreOp(NPUOp *pre_op, NPUOp *trans_op, NPUOp *op) {
  // For op before trans, update the out_ops; the output tensor of op is the input tensor of trans.
  std::vector<NPUOp *> out_ops = pre_op->out_ops();
  size_t i = 0;
  for (; i < out_ops.size(); i++) {
    if (out_ops[i] == op) {
      out_ops[i] = trans_op;
      break;
    }
  }
  if (i == out_ops.size()) {
    out_ops.push_back(trans_op);
  }
  pre_op->set_out_ops(out_ops);
}

void NPUPassUtils::UpdateNC2NHTransNodePreOp(NPUOp *pre_op, const std::vector<NPUOp *> &trans_ops,
                                             const std::vector<NPUOp *> &ops) {
  // For op before trans, there may be multiple outputs.
  auto cur_out_ops = pre_op->out_ops();
  for (size_t i = 0; i < ops.size(); i++) {
    auto itr = find(cur_out_ops.begin(), cur_out_ops.end(), ops[i]);
    if (itr != cur_out_ops.end()) {
      cur_out_ops.erase(itr);
    }
  }
  std::copy(trans_ops.begin(), trans_ops.end(), std::back_inserter(cur_out_ops));
  pre_op->set_out_ops(cur_out_ops);
  // For op before trans, the output tensor is used for output tensor of trans, so replace the output tensor
  // with the input tensor of trans.
  pre_op->set_outputs({trans_ops.at(0)->inputs().at(0)});
}

void NPUPassUtils::UpdateNH2NCTransNodePostOp(NPUOp *trans_op, NPUOp *post_op) {
  auto cur_in_tensors = post_op->inputs();
  cur_in_tensors[0] = trans_op->outputs()[0];
  post_op->set_inputs(cur_in_tensors);
  post_op->set_in_ops({trans_op});
}

void NPUPassUtils::UpdateNC2NHPostOpInTensors(NPUOp *op, NPUOp *trans_op, NPUOp *post_op) {
  // For post_op that doesn't require insert trans op, because the output tensor of op(input tensor of
  // trans_op) is updated, replace the input tensor of post_op.
  auto post_in_tensors = post_op->inputs();
  for (size_t i = 0; i < post_in_tensors.size(); i++) {
    if (post_in_tensors[i] == op->outputs()[0]) {
      post_in_tensors[i] = trans_op->inputs()[0];
      break;
    }
  }
  post_op->set_inputs(post_in_tensors);
}

void NPUPassUtils::UpdateNC2NHTransNodePostOp(NPUOp *op, NPUOp *trans_op, NPUOp *post_op) {
  // The input tensor should be replaced with the output tensor of trans_op.
  auto post_in_tensors = post_op->inputs();
  mindspore::MSTensor old_in_tensor;
  // find out which input tensor of post_op should be updated
  for (size_t i = 0; i < post_in_tensors.size(); ++i) {
    if (OpInputFromOp(post_op, post_in_tensors.at(i)) == op) {
      old_in_tensor = post_in_tensors.at(i);
      break;
    }
  }
  if (old_in_tensor == nullptr) {
    MS_LOG(WARNING) << "Could not find in tensor index";
    return;
  }
  std::replace(post_in_tensors.begin(), post_in_tensors.end(), old_in_tensor, trans_op->outputs().at(0));
  post_op->set_inputs(post_in_tensors);

  // For post_op after trans, op in in_ops should be replaced with trans_op.
  auto post_in_ops = post_op->in_ops();
  if (op == nullptr) {
    post_in_ops.push_back(trans_op);
  } else {
    std::replace(post_in_ops.begin(), post_in_ops.end(), op, trans_op);
  }
  post_op->set_in_ops(post_in_ops);
}

bool NPUPassUtils::IsNhwc2Nchw(NPUOp *op) {
  if (op == nullptr) {
    return false;
  }
  if (op->type() != schema::PrimitiveType_Transpose) {
    return false;
  }
  auto transpose_op = static_cast<TransposeNPUOp *>(op);
  std::vector<int> perm = transpose_op->GetPerm();
  std::vector<int> nh2nc_perm = {0, 3, 1, 2};
  if (perm != nh2nc_perm) {
    return false;
  }
  return true;
}

bool NPUPassUtils::IsNchw2Nhwc(NPUOp *op) {
  if (op == nullptr) {
    return false;
  }
  if (op->type() != schema::PrimitiveType_Transpose) {
    return false;
  }
  auto transpose_op = static_cast<TransposeNPUOp *>(op);
  std::vector<int> perm = transpose_op->GetPerm();
  std::vector<int> nc2nh_perm = {0, 2, 3, 1};
  if (perm != nc2nh_perm) {
    return false;
  }
  return true;
}

NPUOp *NPUPassUtils::OpInputFromOp(NPUOp *op, mindspore::MSTensor in_tensor) {
  // given op and input tensor index, get which op output this tensor.
  // If input tensor is graph input, return nullptr.
  if (op == nullptr) {
    return nullptr;
  }
  auto in_ops = op->in_ops();
  auto output_contain = [in_tensor](NPUOp *op) {
    auto outputs = op->outputs();
    return std::find(outputs.begin(), outputs.end(), in_tensor) != outputs.end();
  };
  auto it = std::find_if(in_ops.begin(), in_ops.end(), output_contain);
  if (it == in_ops.end()) {
    return nullptr;
  }
  return *it;
}

std::vector<mindspore::MSTensor> NPUPassUtils::GetNonConstInputs(NPUOp *op) {
  if (op == nullptr) {
    return std::vector<mindspore::MSTensor>{};
  }
  auto type = op->type();
  auto it = nodes2const_index.find(type);
  if (it != nodes2const_index.end()) {
    auto const_input_indices = it->second;
    std::vector<mindspore::MSTensor> non_const_in_tensors;
    auto in_tensors = op->inputs();
    for (auto i = 0; i < in_tensors.size(); ++i) {
      if (const_input_indices.find(i) == const_input_indices.end()) {
        non_const_in_tensors.push_back(in_tensors[i]);
      }
    }
    return non_const_in_tensors;
  }
  return op->inputs();
}

bool NPUPassUtils::Scale4dCase(NPUOp *op) {
  if (op == nullptr) {
    return false;
  }
  if (op->type() != schema::PrimitiveType_ScaleFusion) {
    return false;
  }
  auto scale_op = static_cast<ScaleNPUOp *>(op);
  auto axis = scale_op->GetAxis();
  auto in_tensor = op->inputs().at(0);
  auto scale_tensor = op->inputs().at(1);
  return in_tensor.Shape().size() == NPU_SHAPE_SIZE && scale_tensor.Shape().size() == 1 &&
         (axis == NHWC_C || axis == -1);
}

void NPUPassUtils::AssistDataNHWC2NCHW(int *data, size_t unit_size) {
  MS_ASSERT(data != nullptr);
  for (size_t i = 0; i < unit_size; ++i) {
    int c = data[3 * unit_size + i];
    // n h w c
    // n c h w
    data[3 * unit_size + i] = data[2 * unit_size + i];
    data[2 * unit_size + i] = data[unit_size + i];
    data[unit_size + i] = c;
  }
}

int NPUPassUtils::MaskDataNHWC2NCHW(int mask) {
  int mask_vec[4];
  for (int i = 0; i < 4; ++i) {
    mask_vec[i] = (uint32_t)(mask) & (1 << i);
  }
  AssistDataNHWC2NCHW(mask_vec, 1);
  int ret = 0;
  for (int i = 0; i < 4; ++i) {
    if (mask_vec[i]) {
      ret += 1 << i;
    }
  }
  return ret;
}
}  // namespace mindspore
