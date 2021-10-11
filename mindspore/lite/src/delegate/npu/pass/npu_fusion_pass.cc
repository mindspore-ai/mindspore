/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/pass/npu_fusion_pass.h"
#include <vector>
#include "src/delegate/npu/pass/npu_pass_utils.h"
#include "src/delegate/npu/npu_converter_utils.h"
#include "src/delegate/npu/op/concat_npu.h"
#include "src/delegate/npu/op/split_npu.h"
#include "src/delegate/npu/op/pad_npu.h"
#include "src/delegate/npu/op/strided_slice_npu.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace {
constexpr int kNumDims = 4;
constexpr int kNumInputSize = 4;
}  // namespace

namespace mindspore {
bool CheckFusion(NPUOp *cur_op) {
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
  return post_flag;
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
  for (auto in_op : cur_op->in_ops()) {
    // graph in op
    if (in_op->in_ops().empty()) {
      continue;
    }
    auto pre_op = in_op->in_ops()[0];

    auto pre_out_ops = pre_op->out_ops();
    for (size_t i = 0; i < pre_out_ops.size(); i++) {
      if (pre_out_ops[i] == in_op) {
        pre_out_ops[i] = cur_op;
        break;
      }
    }
    pre_op->set_out_ops(pre_out_ops);

    auto cur_in_ops = cur_op->in_ops();
    for (size_t i = 0; i < cur_in_ops.size(); i++) {
      if (cur_in_ops[i] == in_op) {
        cur_in_ops[i] = pre_op;
        break;
      }
    }
    cur_op->set_in_ops(cur_in_ops);
    RemoveAndFreeOp(in_op);
  }
  return RET_OK;
}

int NPUFusionPass::UpdatePostOps(NPUOp *cur_op) {
  auto cur_out_ops = cur_op->out_ops();
  for (auto out_op : cur_op->out_ops()) {
    // graph out op
    if (out_op->out_ops().empty()) {
      cur_out_ops.erase(find(cur_out_ops.begin(), cur_out_ops.end(), out_op));
    } else {
      auto post_op = out_op->out_ops()[0];
      auto post_in_ops = post_op->in_ops();
      for (size_t i = 0; i < post_in_ops.size(); i++) {
        if (post_in_ops[i] == out_op) {
          post_in_ops[i] = cur_op;
          break;
        }
      }
      post_op->set_in_ops(post_in_ops);

      for (size_t i = 0; i < cur_out_ops.size(); i++) {
        if (cur_out_ops[i] == out_op) {
          cur_out_ops[i] = post_op;
          break;
        }
      }
    }
    RemoveAndFreeOp(out_op);
  }
  cur_op->set_out_ops(cur_out_ops);
  return RET_OK;
}

int UpdatePreTensors(NPUOp *cur_op) {
  auto tensors_vec = NPUPassUtils::GetNonConstInputs(cur_op);
  for (auto in_op : cur_op->in_ops()) {
    if (in_op->inputs().empty() || in_op->outputs().empty() || in_op->in_ops().empty()) {
      MS_LOG(ERROR) << "in_tensors/out_tensors/in_ops is empty.";
      return RET_ERROR;
    }
    mindspore::MSTensor cur_tensor;
    auto in_tensor = in_op->inputs()[0];
    auto out_tensor = in_op->outputs()[0];
    auto pre_op = in_op->in_ops()[0];
    for (size_t i = 0; i < pre_op->outputs().size(); i++) {
      if (pre_op->outputs()[i] == in_tensor) {
        cur_tensor = pre_op->outputs()[i];
      }
    }
    for (size_t i = 0; i < tensors_vec.size(); i++) {
      if (tensors_vec[i] == out_tensor) {
        tensors_vec[i] = cur_tensor;
      }
    }
  }
  // add constant inputs back
  if (nodes2const_index.find(cur_op->type()) != nodes2const_index.end()) {
    tensors_vec.resize(cur_op->inputs().size());
    auto const_index = nodes2const_index[cur_op->type()];
    for (auto index : const_index) {
      if (index >= cur_op->inputs().size()) {
        continue;
      }
      tensors_vec[index] = cur_op->inputs()[index];
    }
  }
  cur_op->set_inputs(tensors_vec);
  return RET_OK;
}

bool NodeWithNhwc2nchw2nhwcOutput(NPUOp *cur_op) {
  auto out_ops = cur_op->out_ops();
  if (out_ops.empty()) {
    return false;
  }
  bool all_out_ops_transpose = std::all_of(out_ops.begin(), out_ops.end(), [](NPUOp *op) {
    return op->type() == schema::PrimitiveType_Transpose && op->out_ops().size() == 1 &&
           op->out_ops()[0]->type() == schema::PrimitiveType_Transpose && op->out_ops()[0]->out_ops().empty();
  });
  return all_out_ops_transpose;
}

int UpdatePostTensors(NPUOp *cur_op) {
  auto tensor = cur_op->outputs()[0];

  // in case: node->nh2nc->nc2nh(graph output) --->>> node->nc2nh, node out_tensor should be put to nc2nh out tensors
  auto out_ops = cur_op->out_ops();
  if (NodeWithNhwc2nchw2nhwcOutput(cur_op)) {
    std::vector<MSTensor> outputs;
    for (auto i = 0; i < out_ops.size(); ++i) {
      auto ori_out_tensor = cur_op->outputs()[i];
      auto nc_tensor = out_ops[i]->outputs()[0];
      outputs.push_back(nc_tensor);
      auto post_post_op = out_ops[i]->out_ops()[0];
      post_post_op->set_inputs({nc_tensor});
      post_post_op->set_outputs({ori_out_tensor});
    }
    cur_op->set_outputs(outputs);
    return RET_OK;
  }

  auto nhwc_shape = tensor.Shape();
  if (nhwc_shape.size() < kNumDims) {
    MS_LOG(ERROR) << "nhwc_shape < " << kNumDims;
    return RET_ERROR;
  }
  tensor.SetShape({nhwc_shape[NHWC_N], nhwc_shape[NHWC_C], nhwc_shape[NHWC_H], nhwc_shape[NHWC_W]});
  for (auto out_op : cur_op->out_ops()) {
    auto out_tensor = out_op->outputs()[0];
    if (out_op->out_ops().empty()) {
      cur_op->set_outputs({out_op->outputs()[0]});
    }
    for (auto post_op : out_op->out_ops()) {
      auto tensors_vec = post_op->inputs();
      for (int i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == out_tensor) {
          tensors_vec[i] = tensor;
        }
      }
      post_op->set_inputs(tensors_vec);
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
  return RET_OK;
}

int NPUFusionPass::ConcatFusion(NPUOp *cur_op) {
  if (cur_op == nullptr) {
    return RET_ERROR;
  }
  int ret = UpdateOp(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateOp failed.";
    return ret;
  }
  if (cur_op->type() != schema::PrimitiveType_Concat) {
    return RET_ERROR;
  }
  auto concat_op = static_cast<ConcatNPUOp *>(cur_op);
  ret = concat_op->HandleAxis();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HandleAxis failed.";
    return ret;
  }
  return RET_OK;
}

int NPUFusionPass::SplitFusion(NPUOp *cur_op) {
  if (cur_op == nullptr) {
    return RET_ERROR;
  }
  int ret = UpdateOp(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateOp failed.";
    return RET_ERROR;
  }
  if (cur_op->type() != schema::PrimitiveType_Split) {
    return RET_ERROR;
  }
  auto split_op = static_cast<SplitNPUOp *>(cur_op);
  ret = split_op->HandleAxis();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HandleAxis failed.";
    return ret;
  }
  return RET_OK;
}

int NPUFusionPass::PadFusion(NPUOp *cur_op) {
  if (cur_op == nullptr) {
    return RET_ERROR;
  }
  int ret = UpdateOp(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateOp failed.";
    return ret;
  }
  if (cur_op->type() != schema::PrimitiveType_PadFusion) {
    return RET_ERROR;
  }
  auto pad_op = static_cast<PadNPUOp *>(cur_op);
  ret = pad_op->HandleAxis();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HandleAxis failed.";
    return ret;
  }
  return RET_OK;
}

int NPUFusionPass::StridedSliceFusion(NPUOp *cur_op) {
  // basic requirement: input is nhwc 4d
  if (cur_op == nullptr) {
    return RET_ERROR;
  }
  int ret = UpdateOp(cur_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdateOp failed.";
    return ret;
  }
  if (cur_op->inputs().size() < kNumInputSize) {
    MS_LOG(ERROR) << "in tensors size < " << kNumInputSize;
    return RET_ERROR;
  }
  if (cur_op->type() != schema::PrimitiveType_StridedSlice) {
    return RET_ERROR;
  }
  auto begin_tensor = cur_op->inputs().at(BEGIN_INDEX);
  int *begin = reinterpret_cast<int *>(begin_tensor.MutableData());
  MS_ASSERT(begin);
  (void)NPUPassUtils::AssistDataNHWC2NCHW(begin, 1);
  auto end_tensor = cur_op->inputs().at(END_INDEX);
  int *end = reinterpret_cast<int *>(end_tensor.MutableData());
  MS_ASSERT(end);
  NPUPassUtils::AssistDataNHWC2NCHW(end, 1);
  auto stride_tensor = cur_op->inputs().at(STRIDE_INDEX);
  if (cur_op->inputs().size() == ONNX_INPUT_SIZE) {
    stride_tensor = cur_op->inputs().at(ONNX_STRIDE_INDEX);
  }
  int *stride = reinterpret_cast<int *>(stride_tensor.MutableData());
  MS_ASSERT(stride);
  NPUPassUtils::AssistDataNHWC2NCHW(stride, 1);

  auto stride_slice_op = static_cast<StridedSliceNPUOp *>(cur_op);
  ret = stride_slice_op->HandleAxis();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HandleAxis failed.";
    return ret;
  }
  return RET_OK;
}

int NPUFusionPass::FormatFusion(NPUOp *cur_op) {
  if (cur_op == nullptr) {
    return RET_ERROR;
  }
  auto is_input_op = cur_op->in_ops().empty();
  NPUOp *pre_op = nullptr;
  if (!is_input_op) {
    pre_op = cur_op->in_ops()[0];
  }
  auto in_tensor = cur_op->inputs()[0];
  std::vector<NPUOp *> pre_insert_ops;
  for (const auto &trans_op : cur_op->out_ops()) {
    if (trans_op->out_ops().empty() && !is_input_op) {
      // cur_op is a trans cur_op, it's input cur_op num and input tensor num must be 1
      cur_op->in_ops()[0]->set_outputs({trans_op->outputs()[0]});
      // in fp16 mode, tensor data type fp16 need to be changed back.
      auto tensor = cur_op->in_ops()[0]->outputs()[0];
      if (tensor.DataType() == DataType::kNumberTypeFloat16) {
        tensor.SetDataType(DataType::kNumberTypeFloat32);
      }
    }
    for (const auto &post_op : trans_op->out_ops()) {
      // update tensor
      auto tensors_vec = post_op->inputs();
      for (size_t i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == trans_op->outputs()[0]) {
          tensors_vec[i] = in_tensor;
          break;
        }
      }
      post_op->set_inputs(tensors_vec);

      // update op
      auto post_in_ops = post_op->in_ops();
      for (size_t i = 0; i < post_in_ops.size(); i++) {
        if (post_in_ops[i] == trans_op) {
          if (is_input_op) {
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
  if (!is_input_op) {
    auto pre_out_ops = pre_op->out_ops();
    size_t cur_op_index = 0;
    for (size_t index = 0; index < pre_out_ops.size(); index++) {
      if (pre_out_ops[index] == cur_op) {
        pre_out_ops.erase(pre_out_ops.begin() + index);
        cur_op_index = index;
      } else {
        auto tensors_vec = pre_out_ops[index]->inputs();
        for (size_t i = 0; i < tensors_vec.size(); i++) {
          if (tensors_vec[i] == in_tensor) {
            tensors_vec[i] = pre_op->outputs()[0];
            break;
          }
        }
        pre_out_ops[index]->set_inputs(tensors_vec);
      }
    }
    pre_out_ops.insert(pre_out_ops.begin() + cur_op_index, pre_insert_ops.begin(), pre_insert_ops.end());
    pre_op->set_out_ops(pre_out_ops);
  }
  RemoveAndFreeOp(cur_op);
  return RET_OK;
}

int NPUFusionPass::Run(NPUGraph *subgraph) {
  all_ops_ = subgraph->GetOps();
  for (size_t i = 0; i < all_ops_->size(); i++) {
    auto cur_op = (*all_ops_)[i];
    auto ret = RET_OK;
    if (CheckFusion(cur_op)) {
      switch (cur_op->type()) {
        case schema::PrimitiveType_Split:
          i -= cur_op->in_ops().size();
          ret = SplitFusion(cur_op);
          continue;
        case schema::PrimitiveType_Concat:
          i -= cur_op->in_ops().size();
          ret = ConcatFusion(cur_op);
          continue;
        case schema::PrimitiveType_PadFusion:
          i -= cur_op->in_ops().size();
          ret = PadFusion(cur_op);
          continue;
        case schema::PrimitiveType_StridedSlice:
          i -= cur_op->in_ops().size();
          ret = StridedSliceFusion(cur_op);
          continue;
        case schema::PrimitiveType_AddFusion:
        case schema::PrimitiveType_MulFusion:
        case schema::PrimitiveType_DivFusion:
        case schema::PrimitiveType_Activation:
        case schema::PrimitiveType_Eltwise:
          i -= cur_op->in_ops().size();
          ret = CommonFusion(cur_op);
          continue;
        default:
          continue;
      }
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
