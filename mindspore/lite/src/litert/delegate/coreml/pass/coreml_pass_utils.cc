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

#include "src/litert/delegate/coreml/pass/coreml_pass_utils.h"
#include <algorithm>
#include "src/litert/delegate/coreml/op/transpose_coreml.h"

namespace mindspore::lite {
CoreMLOp *CoreMLPassUtils::CreateNchw2NhwcOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                             const std::vector<mindspore::MSTensor> &out_tensors,
                                             const std::string &name) {
  auto trans_op = new (std::nothrow) TransposeCoreMLOp(in_tensors, out_tensors, NCHW2NHWC_PERM, name);
  if (trans_op == nullptr) {
    MS_LOG(ERROR) << "New Nchw2Nhwc CoreMLOp failed.";
    return nullptr;
  }
  auto ret = trans_op->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Nchw2Nhwc transpose op init failed.";
    return nullptr;
  }
  return trans_op;
}

CoreMLOp *CoreMLPassUtils::CreateNhwc2NchwOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                             const std::vector<mindspore::MSTensor> &out_tensors,
                                             const std::string &name) {
  auto trans_op = new (std::nothrow) TransposeCoreMLOp(in_tensors, out_tensors, NHWC2NCHW_PERM, name);
  if (trans_op == nullptr) {
    MS_LOG(ERROR) << "New Nhwc2Nchw CoreMLOp failed.";
    return nullptr;
  }
  auto ret = trans_op->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Nhwc2Nchw transpose op init failed.";
    return nullptr;
  }
  return trans_op;
}

void CoreMLPassUtils::UpdateOp(CoreMLOp *op, const std::vector<CoreMLOp *> &in_ops,
                               const std::vector<CoreMLOp *> &out_ops,
                               const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &outputs) {
  op->set_inputs(in_tensors);
  op->set_outputs(outputs);
  op->set_in_ops(in_ops);
  op->set_out_ops(out_ops);
}

void CoreMLPassUtils::UpdateNH2NCTransNodePreOp(CoreMLOp *pre_op, CoreMLOp *trans_op, CoreMLOp *op) {
  // For op before trans, update the out_ops; the output tensor of op is the input tensor of trans, no need to update.
  std::vector<CoreMLOp *> out_ops = pre_op->out_ops();
  if (op == nullptr) {
    out_ops.emplace_back(trans_op);
  } else {
    for (size_t i = 0; i < out_ops.size(); i++) {
      if (out_ops[i] == op) {
        out_ops[i] = trans_op;
        break;
      }
    }
  }
  pre_op->set_out_ops(out_ops);
}

void CoreMLPassUtils::UpdateNC2NHTransNodePreOp(CoreMLOp *pre_op, const std::vector<CoreMLOp *> &trans_ops,
                                                const std::vector<CoreMLOp *> &ops) {
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

void CoreMLPassUtils::UpdateNH2NCTransNodePostOp(CoreMLOp *trans_op, CoreMLOp *post_op) {
  auto cur_in_tensors = post_op->inputs();
  cur_in_tensors[0] = trans_op->outputs()[0];
  post_op->set_inputs(cur_in_tensors);
  post_op->set_in_ops({trans_op});
}

void CoreMLPassUtils::UpdateNC2NHTransNodePostOp(CoreMLOp *op, CoreMLOp *trans_op, CoreMLOp *post_op,
                                                 const mindspore::MSTensor &org_in_tensor) {
  // The input tensor should be replaced with the output tensor of trans_op.
  auto post_in_tensors = post_op->inputs();
  std::replace(post_in_tensors.begin(), post_in_tensors.end(), org_in_tensor, trans_op->outputs().at(0));
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

bool CoreMLPassUtils::IsNhwc2Nchw(CoreMLOp *op) {
  if (op == nullptr) {
    return false;
  }
  if (op->type() != schema::PrimitiveType_Transpose) {
    return false;
  }
  auto transpose_op = static_cast<TransposeCoreMLOp *>(op);
  std::vector<int> perm = transpose_op->GetPerm();
  std::vector<int> nh2nc_perm = {0, 3, 1, 2};
  if (perm != nh2nc_perm) {
    return false;
  }
  return true;
}

bool CoreMLPassUtils::IsNchw2Nhwc(CoreMLOp *op) {
  if (op == nullptr) {
    return false;
  }
  if (op->type() != schema::PrimitiveType_Transpose) {
    return false;
  }
  auto transpose_op = static_cast<TransposeCoreMLOp *>(op);
  std::vector<int> perm = transpose_op->GetPerm();
  std::vector<int> nc2nh_perm = {0, 2, 3, 1};
  if (perm != nc2nh_perm) {
    return false;
  }
  return true;
}

CoreMLOp *CoreMLPassUtils::OpInputFromOp(CoreMLOp *op, mindspore::MSTensor in_tensor) {
  // given op and input tensor index, get which op output this tensor.
  // If input tensor is graph input, return nullptr.
  if (op == nullptr) {
    return nullptr;
  }
  auto in_ops = op->in_ops();
  auto output_contain = [in_tensor](CoreMLOp *in_op) {
    auto outputs = in_op->outputs();
    return std::find(outputs.begin(), outputs.end(), in_tensor) != outputs.end();
  };
  auto it = std::find_if(in_ops.begin(), in_ops.end(), output_contain);
  if (it == in_ops.end()) {
    return nullptr;
  }
  return *it;
}

std::vector<mindspore::MSTensor> CoreMLPassUtils::GetNonConstInputs(CoreMLOp *op) {
  MS_CHECK_TRUE_MSG(op != nullptr, {}, "Input op is null!");
  std::vector<mindspore::MSTensor> non_const_in_tensors;
  std::copy_if(op->inputs().begin(), op->inputs().end(), std::back_inserter(non_const_in_tensors),
               [](const auto &tensor) { return !tensor.IsConst(); });
  return non_const_in_tensors;
}
}  // namespace mindspore::lite
