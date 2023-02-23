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

#include "src/litert/pass/format_pass/transpose_strategy.h"
#include "nnacl/op_base.h"
#include "nnacl/arg_min_max_parameter.h"
#include "nnacl/concat_parameter.h"
#include "nnacl/crop_parameter.h"

namespace mindspore::lite::pass {
static const std::set<schema::PrimitiveType> arithmetic_kernel_lists = {
  schema::PrimitiveType_AddFusion,    schema::PrimitiveType_AddN,
  schema::PrimitiveType_DivFusion,    schema::PrimitiveType_Eltwise,
  schema::PrimitiveType_Equal,        schema::PrimitiveType_FloorDiv,
  schema::PrimitiveType_FloorMod,     schema::PrimitiveType_Greater,
  schema::PrimitiveType_GreaterEqual, schema::PrimitiveType_Less,
  schema::PrimitiveType_LessEqual,    schema::PrimitiveType_LogicalAnd,
  schema::PrimitiveType_LogicalOr,    schema::PrimitiveType_Maximum,
  schema::PrimitiveType_Minimum,      schema::PrimitiveType_Mod,
  schema::PrimitiveType_MulFusion,    schema::PrimitiveType_NotEqual,
  schema::PrimitiveType_RealDiv,      schema::PrimitiveType_SquaredDifference,
  schema::PrimitiveType_SubFusion,
};

size_t TransposeStrategy::GetTransCount(const std::vector<kernel::KernelExec *> &kernels, TransInfoPair *trans_info) {
  size_t count = 0;
  for (const auto &in_kernel : kernels) {
    TransInfoPair tmp_trans;
    if (GetTransposeInfo(in_kernel, &tmp_trans) != RET_OK) {
      continue;
    }
    if (IsNoneTranspose(*trans_info)) {
      *trans_info = tmp_trans;
      count++;
    } else if (IsSameTranspose(*trans_info, tmp_trans)) {
      count++;
    } else {
      continue;
    }
  }
  return count;
}

bool CheckInTensorsShape(const kernel::KernelExec *kernel, const Format &runtime_format) {
  // If teh fusion is valid, kernel will be executed in runtime_format.
  // Only check arithmetic (two input) kernel input tensors.
  // If broadcast for various formats is supported, this function can be deleted.
  // eg: tensor 1 shape(1, 128, 24, 24), tensor 2 shape(1, 128, 1, 1), the NC4HW4 format is not supported now.
  if (arithmetic_kernel_lists.find(kernel->type()) == arithmetic_kernel_lists.end()) {
    return true;
  }
  for (const auto &in_tensor : kernel->in_tensors()) {
    const auto &in_shape = in_tensor->shape();
    if (std::any_of(in_shape.begin(), in_shape.end(), [](const int &dim) { return dim == -1; })) {
      return false;
    }
  }
  const auto &in0_shape = kernel->in_tensors().at(0)->shape();
  if (runtime_format == NHWC || runtime_format == NCHW) {
    // For NCHW or NHWC format, the shape.size must be equal.
    if (std::any_of(kernel->in_tensors().begin(), kernel->in_tensors().end(),
                    [&in0_shape](const Tensor *in_tensor) { return in_tensor->shape().size() != in0_shape.size(); })) {
      return false;
    }
  } else {
    // For other format(NCXHWX), the shape must be equal.
    if (std::any_of(kernel->in_tensors().begin(), kernel->in_tensors().end(),
                    [&in0_shape](const Tensor *in_tensor) { return in_tensor->shape() != in0_shape; })) {
      return false;
    }
  }
  return true;
}

bool TransposeStrategy::CheckFusion(const kernel::KernelExec *kernel, TransInfoPair *pre_trans,
                                    TransInfoPair *post_trans) {
  if (dynamic_format_kernel_lists.find(kernel->type()) == dynamic_format_kernel_lists.end()) {
    return false;
  }
  auto input_count = GetTransCount(kernel->in_kernels(), pre_trans);
  auto output_count = GetTransCount(kernel->out_kernels(), post_trans);
  if (IsSameTranspose(*pre_trans, *post_trans)) {
    return false;
  }
  if (!IsOppositiveTranspose(*pre_trans, *post_trans)) {
    return false;
  }
  auto in_and_out_size = kernel->in_tensors().size() + kernel->out_kernels().size();
  if ((input_count + output_count) <= in_and_out_size / C2NUM) {
    MS_LOG(INFO) << "The fusion can't decrease transpose op number.";
    return false;
  }
  if (IsNoneTranspose(*pre_trans)) {
    pre_trans->src_format_ = post_trans->dst_format_;
    pre_trans->dst_format_ = post_trans->src_format_;
  }
  if (IsNoneTranspose(*post_trans)) {
    post_trans->src_format_ = pre_trans->dst_format_;
    post_trans->dst_format_ = pre_trans->src_format_;
  }

  if (((!IsSameTranspose(*post_trans, NCHW2NHWCTrans)) && (!IsSameTranspose(*post_trans, NHWC2NCHWTrans))) &&
      dynamic_format_kernel_lists.at(kernel->type()) == true) {
    return false;
  }
  if (CheckInTensorsShape(kernel, (Format)(post_trans->dst_format_)) == false) {
    return false;
  }
  return true;
}

int TransFormAxis(int axis, const TransInfoPair &trans) {
  if (IsSameTranspose(trans, NHWC2NCHWTrans)) {
    switch (axis) {
      case kNHWC_N:
        return kNCHW_N;
      case kNHWC_H:
        return kNCHW_H;
      case kNHWC_W:
        return kNCHW_W;
      case kNHWC_C:
        return kNCHW_C;
      default:
        return axis;
    }
  }
  if (IsSameTranspose(trans, NCHW2NHWCTrans)) {
    switch (axis) {
      case kNCHW_N:
        return kNHWC_N;
      case kNCHW_H:
        return kNHWC_H;
      case kNCHW_W:
        return kNHWC_W;
      case kNCHW_C:
        return kNHWC_C;
      default:
        return axis;
    }
  }
  return axis;
}

int HandleArgMinMaxKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto arg_min_max_param = reinterpret_cast<ArgMinMaxParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(arg_min_max_param);
  arg_min_max_param->axis_ = TransFormAxis(arg_min_max_param->axis_, trans);
  return RET_OK;
}

int HandleConcatKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto concat_param = reinterpret_cast<ConcatParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(concat_param);
  concat_param->axis_ = TransFormAxis(concat_param->axis_, trans);
  return RET_OK;
}

int HandleCropKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto crop_param = reinterpret_cast<CropParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(crop_param);
  crop_param->axis_ = TransFormAxis(static_cast<int>(crop_param->axis_), trans);
  return RET_OK;
}

// Nullptr: the change axis function for corresponding kernel is to be implemented.
static const std::map<schema::PrimitiveType, std::function<int(kernel::KernelExec *kernel, const TransInfoPair &trans)>>
  process_funcs = {
    {schema::PrimitiveType_ArgMinFusion, HandleArgMinMaxKernel},
    {schema::PrimitiveType_ArgMaxFusion, HandleArgMinMaxKernel},
    {schema::PrimitiveType_Concat, HandleConcatKernel},
    {schema::PrimitiveType_Crop, HandleCropKernel},
    {schema::PrimitiveType_SliceFusion, nullptr},
    {schema::PrimitiveType_Softmax, nullptr},
    {schema::PrimitiveType_Split, nullptr},
    {schema::PrimitiveType_Squeeze, nullptr},
    {schema::PrimitiveType_Stack, nullptr},
    {schema::PrimitiveType_StridedSlice, nullptr},
    {schema::PrimitiveType_Unsqueeze, nullptr},
    {schema::PrimitiveType_Unstack, nullptr},
    {schema::PrimitiveType_LogSoftmax, nullptr},
};

int TransposeStrategy::ChangeKernelAxis(kernel::KernelExec *kernel, const TransInfoPair &trans) {
  if (dynamic_format_kernel_lists.find(kernel->type()) == dynamic_format_kernel_lists.end()) {
    MS_LOG(ERROR) << "Can't find the axis change function for " << kernel->name();
    return RET_ERROR;
  }
  if (dynamic_format_kernel_lists.at(kernel->type()) == false) {
    MS_LOG(INFO) << "No need to change axis for " << kernel->name();
    return RET_OK;
  }
  auto process_iter = process_funcs.find(kernel->type());
  if (process_iter != process_funcs.end()) {
    CHECK_NULL_RETURN(process_iter->second);
    return process_iter->second(kernel, trans);
  } else {
    MS_LOG(ERROR) << "Can't find the axis change function for " << kernel->name();
    return RET_ERROR;
  }
}
}  // namespace mindspore::lite::pass
