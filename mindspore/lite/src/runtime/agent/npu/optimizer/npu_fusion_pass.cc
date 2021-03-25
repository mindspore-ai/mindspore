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
#include "src/runtime/agent/npu/optimizer/npu_fusion_pass.h"
#include <vector>
#include "src/runtime/agent/npu/optimizer/npu_pass_utils.h"
#include "src/lite_kernel.h"
#include "nnacl/concat_parameter.h"
#include "nnacl/split_parameter.h"
#include "nnacl/pad_parameter.h"
#include "nnacl/strided_slice_parameter.h"

namespace mindspore::lite {
bool CheckFusion(kernel::LiteKernel *kernel) {
  if (kernel->in_kernels().empty() || kernel->out_kernels().empty()) {
    return false;
  }
  auto pre_flag =
    std::all_of(kernel->in_kernels().begin(), kernel->in_kernels().end(), [](const kernel::LiteKernel *in_kernel) {
      return NPUPassUtils::IsNchw2Nhwc(in_kernel) && in_kernel->out_kernels().size() == 1;
    });
  if (!pre_flag) {
    return false;
  }
  auto post_flag =
    std::all_of(kernel->out_kernels().begin(), kernel->out_kernels().end(),
                [](const kernel::LiteKernel *out_kernel) { return NPUPassUtils::IsNhwc2Nchw(out_kernel); });
  return post_flag;
}

bool CheckFormatFusion(kernel::LiteKernel *kernel) {
  if (kernel->out_kernels().empty()) {
    return false;
  }
  if (NPUPassUtils::IsNhwc2Nchw(kernel)) {
    return std::all_of(kernel->out_kernels().begin(), kernel->out_kernels().end(),
                       [](const kernel::LiteKernel *kernel) { return NPUPassUtils::IsNchw2Nhwc(kernel); });
  }
  if (NPUPassUtils::IsNchw2Nhwc(kernel)) {
    return std::all_of(kernel->out_kernels().begin(), kernel->out_kernels().end(),
                       [](const kernel::LiteKernel *kernel) { return NPUPassUtils::IsNhwc2Nchw(kernel); });
  }
  return false;
}

void NPUFusionPass::RemoveAndFreeKernel(kernel::LiteKernel *cur_kernel) {
  auto itr = find(kernels->begin(), kernels->end(), cur_kernel);
  if (itr != kernels->end()) {
    kernels->erase(itr);
  }
  delete cur_kernel;
}

void NPUFusionPass::UpdatePreKernels(kernel::LiteKernel *cur_kernel) {
  for (auto in_kernel : cur_kernel->in_kernels()) {
    // graph in kernel
    if (in_kernel->in_kernels().empty()) {
      continue;
    }
    auto pre_kernel = in_kernel->in_kernels()[0];

    auto pre_out_kernels = pre_kernel->out_kernels();
    for (size_t i = 0; i < pre_out_kernels.size(); i++) {
      if (pre_out_kernels[i] == in_kernel) {
        pre_out_kernels[i] = cur_kernel;
        break;
      }
    }
    pre_kernel->set_out_kernels(pre_out_kernels);

    auto cur_in_kernels = cur_kernel->in_kernels();
    for (size_t i = 0; i < cur_in_kernels.size(); i++) {
      if (cur_in_kernels[i] == in_kernel) {
        cur_in_kernels[i] = pre_kernel;
        break;
      }
    }
    cur_kernel->set_in_kernels(cur_in_kernels);
    RemoveAndFreeKernel(in_kernel);
  }
}

void NPUFusionPass::UpdatePostKernels(kernel::LiteKernel *cur_kernel) {
  auto cur_out_kernels = cur_kernel->out_kernels();
  for (auto out_kernel : cur_kernel->out_kernels()) {
    // graph out kernel
    if (out_kernel->out_kernels().empty()) {
      cur_out_kernels.erase(find(cur_out_kernels.begin(), cur_out_kernels.end(), out_kernel));
    } else {
      auto post_kernel = out_kernel->out_kernels()[0];
      auto post_in_kernels = post_kernel->in_kernels();
      for (size_t i = 0; i < post_in_kernels.size(); i++) {
        if (post_in_kernels[i] == out_kernel) {
          post_in_kernels[i] = cur_kernel;
          break;
        }
      }
      post_kernel->set_in_kernels(post_in_kernels);

      for (size_t i = 0; i < cur_out_kernels.size(); i++) {
        if (cur_out_kernels[i] == out_kernel) {
          cur_out_kernels[i] = post_kernel;
          break;
        }
      }
    }
    RemoveAndFreeKernel(out_kernel);
  }
  cur_kernel->set_out_kernels(cur_out_kernels);
}

void UpdatePreTensors(kernel::LiteKernel *cur_kernel) {
  auto tensors_vec = NPUPassUtils::GetNonConstInputs(cur_kernel);
  for (auto in_kernel : cur_kernel->in_kernels()) {
    lite::Tensor *cur_tensor = nullptr;
    auto in_tensor = in_kernel->in_tensors()[0];
    auto out_tensor = in_kernel->out_tensors()[0];
    auto pre_kernel = in_kernel->in_kernels()[0];
    for (size_t i = 0; i < pre_kernel->out_tensors().size(); i++) {
      if (pre_kernel->out_tensors()[i] == in_tensor) {
        cur_tensor = pre_kernel->out_tensors()[i];
      }
    }
    for (size_t i = 0; i < tensors_vec.size(); i++) {
      if (tensors_vec[i] == out_tensor) {
        tensors_vec[i] = cur_tensor;
      }
    }
  }
  // add constant inputs back
  if (nodes2const_index.find(static_cast<schema::PrimitiveType>(cur_kernel->op_parameter()->type_)) !=
      nodes2const_index.end()) {
    tensors_vec.resize(cur_kernel->in_tensors().size());
    auto const_index = nodes2const_index[static_cast<schema::PrimitiveType>(cur_kernel->op_parameter()->type_)];
    for (auto index : const_index) {
      tensors_vec[index] = cur_kernel->in_tensors()[index];
    }
  }
  cur_kernel->set_in_tensors(tensors_vec);
}

void UpdatePostTensors(kernel::LiteKernel *cur_kernel) {
  auto tensor = cur_kernel->out_tensors()[0];

  // in case: node->nh2nc->nc2nh(graph output) --->>> node->nc2nh, node out_tensor should be put to nnc2nh out tensors
  auto out_kernels = cur_kernel->out_kernels();
  if (out_kernels.size() == 1 && out_kernels[0]->out_kernels().size() == 1 &&
      out_kernels[0]->out_kernels()[0]->out_kernels().empty() &&
      out_kernels[0]->out_kernels()[0]->type_str() == "Transpose") {
    auto nc_tensor = out_kernels[0]->out_tensors()[0];  // nh2nc's out tensor
    cur_kernel->set_out_tensors({nc_tensor});
    auto post_post_kernel = out_kernels[0]->out_kernels()[0];
    // nc2nh kernel set in_tensor out_tensor
    auto post_post_k_in_tensors = post_post_kernel->in_tensors();
    post_post_k_in_tensors[0] = nc_tensor;
    post_post_kernel->set_in_tensors(post_post_k_in_tensors);
    post_post_kernel->set_out_tensors({tensor});
    return;
  }

  tensor->set_format(schema::Format_NCHW);
  auto nhwc_shape = tensor->shape();
  tensor->set_shape({nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]});
  for (auto out_kernel : cur_kernel->out_kernels()) {
    auto out_tensor = out_kernel->out_tensors()[0];
    if (out_kernel->out_kernels().empty()) {
      cur_kernel->set_out_tensors({out_kernel->out_tensors()[0]});
    }
    for (auto post_kernel : out_kernel->out_kernels()) {
      auto tensors_vec = post_kernel->in_tensors();
      for (int i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == out_tensor) {
          tensors_vec[i] = tensor;
        }
      }
      post_kernel->set_in_tensors(tensors_vec);
    }
  }
}

int TransFormAxis(int axis) {
  switch (axis) {
    case 0:
      return 0;
    case 1:
      return 2;
    case 2:
      return 3;
    case 3:
    case -1:
      return 1;
    default:
      return -2;
  }
}

void NPUFusionPass::UpdateKernel(kernel::LiteKernel *kernel) {
  UpdatePreTensors(kernel);
  UpdatePostTensors(kernel);
  UpdatePreKernels(kernel);
  UpdatePostKernels(kernel);
}

int NPUFusionPass::CommonFusion(kernel::LiteKernel *kernel) {
  UpdateKernel(kernel);
  return RET_OK;
}

int NPUFusionPass::ConcatFusion(kernel::LiteKernel *kernel) {
  UpdateKernel(kernel);
  auto concat_param = reinterpret_cast<ConcatParameter *>(kernel->op_parameter());
  concat_param->axis_ = TransFormAxis(concat_param->axis_);
  return RET_OK;
}

int NPUFusionPass::FormatFusion(kernel::LiteKernel *kernel) {
  auto is_input_kernel = kernel->in_kernels().empty();
  kernel::LiteKernel *pre_kernel = nullptr;
  if (!is_input_kernel) {
    pre_kernel = kernel->in_kernels()[0];
  }
  auto in_tensor = kernel->in_tensors()[0];
  std::vector<kernel::LiteKernel *> pre_insert_kernels;
  for (const auto &trans_kernel : kernel->out_kernels()) {
    if (trans_kernel->out_kernels().empty()) {
      // kernel is a trans kernel, it's input kernel num and input tensor num must be 1
      kernel->in_kernels()[0]->set_out_tensors({trans_kernel->out_tensors()[0]});
      // in fp16 mode, tensor data type fp16 need to be changed back.
      auto tensor = kernel->in_kernels()[0]->out_tensors()[0];
      if (tensor->data_type() == kNumberTypeFloat16) {
        tensor->set_data_type(kNumberTypeFloat32);
      }
    }
    for (const auto &post_kernel : trans_kernel->out_kernels()) {
      // update tensor
      auto tensors_vec = post_kernel->in_tensors();
      for (size_t i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == trans_kernel->out_tensors()[0]) {
          tensors_vec[i] = in_tensor;
          break;
        }
      }
      post_kernel->set_in_tensors(tensors_vec);

      // update kernel
      auto post_in_kernels = post_kernel->in_kernels();
      for (size_t i = 0; i < post_in_kernels.size(); i++) {
        if (post_in_kernels[i] == trans_kernel) {
          if (is_input_kernel) {
            post_in_kernels.erase(post_in_kernels.begin() + i);
          } else {
            post_in_kernels[i] = pre_kernel;
          }
          break;
        }
      }
      post_kernel->set_in_kernels(post_in_kernels);
      pre_insert_kernels.push_back(post_kernel);
    }
    RemoveAndFreeKernel(trans_kernel);
  }
  if (!is_input_kernel) {
    auto pre_out_kernels = pre_kernel->out_kernels();
    size_t index = 0;
    for (; index < pre_out_kernels.size(); index++) {
      if (pre_out_kernels[index] == kernel) {
        pre_out_kernels.erase(pre_out_kernels.begin() + index);
        break;
      }
    }
    pre_out_kernels.insert(pre_out_kernels.begin() + index, pre_insert_kernels.begin(), pre_insert_kernels.end());
    pre_kernel->set_out_kernels(pre_out_kernels);
  }
  RemoveAndFreeKernel(kernel);
  return RET_OK;
}

int NPUFusionPass::SplitFusion(kernel::LiteKernel *kernel) {
  UpdateKernel(kernel);
  auto split_param = reinterpret_cast<SplitParameter *>(kernel->op_parameter());
  split_param->split_dim_ = TransFormAxis(split_param->split_dim_);
  return RET_OK;
}

int NPUFusionPass::PadFusion(kernel::LiteKernel *kernel) {
  UpdateKernel(kernel);
  auto pad_param = reinterpret_cast<PadParameter *>(kernel->op_parameter());
  int c1 = pad_param->paddings_[6];
  int c2 = pad_param->paddings_[7];
  // 0 1 2 3 4 5 6 7
  // n n h h w w c c
  // n n c c h h w w
  pad_param->paddings_[6] = pad_param->paddings_[4];
  pad_param->paddings_[7] = pad_param->paddings_[5];
  pad_param->paddings_[4] = pad_param->paddings_[2];
  pad_param->paddings_[5] = pad_param->paddings_[3];
  pad_param->paddings_[2] = c1;
  pad_param->paddings_[3] = c2;
  return RET_OK;
}

int NPUFusionPass::StridedSliceFusion(kernel::LiteKernel *kernel) {
  // basic requirement: input is nhwc 4d
  UpdateKernel(kernel);
  auto param = reinterpret_cast<StridedSliceParameter *>(kernel->op_parameter());
  auto begin_tensor = kernel->in_tensors().at(1);
  int *begin = reinterpret_cast<int *>(begin_tensor->data_c());
  (void)NPUPassUtils::AssistDataNHWC2NCHW(begin, 1);
  auto end_tensor = kernel->in_tensors().at(2);
  int *end = reinterpret_cast<int *>(end_tensor->data_c());
  NPUPassUtils::AssistDataNHWC2NCHW(end, 1);
  auto stride_tensor = kernel->in_tensors().at(3);
  if (kernel->in_tensors().size() == 5) {
    stride_tensor = kernel->in_tensors().at(4);
  }
  int *stride = reinterpret_cast<int *>(stride_tensor->data_c());
  NPUPassUtils::AssistDataNHWC2NCHW(stride, 1);
  param->begins_mask_ = NPUPassUtils::MaskDataNHWC2NCHW(param->begins_mask_);
  param->ends_mask_ = NPUPassUtils::MaskDataNHWC2NCHW(param->ends_mask_);
  param->ellipsisMask_ = NPUPassUtils::MaskDataNHWC2NCHW(param->ellipsisMask_);
  param->newAxisMask_ = NPUPassUtils::MaskDataNHWC2NCHW(param->newAxisMask_);
  param->shrinkAxisMask_ = NPUPassUtils::MaskDataNHWC2NCHW(param->shrinkAxisMask_);
  return RET_OK;
}

int NPUFusionPass::Run() {
  for (size_t i = 0; i < kernels->size(); i++) {
    auto kernel = (*kernels)[i];
    if (CheckFusion(kernel)) {
      switch (kernel->Type()) {
        case schema::PrimitiveType_Split:
          i -= kernel->in_kernels().size();
          SplitFusion(kernel);
          continue;
        case schema::PrimitiveType_Concat:
          i -= kernel->in_kernels().size();
          ConcatFusion(kernel);
          continue;
        case schema::PrimitiveType_PadFusion:
          i -= kernel->in_kernels().size();
          PadFusion(kernel);
          continue;
        case schema::PrimitiveType_StridedSlice:
          i -= kernel->in_kernels().size();
          StridedSliceFusion(kernel);
          continue;
        case schema::PrimitiveType_AddFusion:
        case schema::PrimitiveType_Activation:
        case schema::PrimitiveType_Eltwise:
          i -= kernel->in_kernels().size();
          CommonFusion(kernel);
          continue;
        default:
          continue;
      }
    }
  }
  for (size_t i = 0; i < kernels->size(); ++i) {
    auto kernel = (*kernels)[i];
    if (CheckFormatFusion(kernel)) {
      i--;
      FormatFusion(kernel);
    }
  }

  return RET_OK;
}
}  // namespace mindspore::lite
