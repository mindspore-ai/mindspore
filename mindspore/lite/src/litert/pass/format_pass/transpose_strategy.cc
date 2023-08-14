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
#include "nnacl/softmax_parameter.h"
#include "nnacl/split_parameter.h"
#include "nnacl/squeeze_parameter.h"
#include "nnacl/stack_parameter.h"
#include "nnacl/unsqueeze_parameter.h"
#include "nnacl/unstack_parameter.h"
#include "nnacl/slice_parameter.h"
#include "nnacl/strided_slice_parameter.h"

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
  if (arithmetic_kernel_lists.find(kernel::SchemaType(kernel->type())) == arithmetic_kernel_lists.end()) {
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

namespace {
using TransAxisFunc = std::function<int(kernel::KernelExec *, const TransInfoPair &)>;
TransAxisFunc kNoNeedTransAxisFunc = [](kernel::KernelExec *, const TransInfoPair &) { return RET_OK; };
TransAxisFunc kNotImplementedTransAxisFunc = [](kernel::KernelExec *, const TransInfoPair &) { return RET_ERROR; };
int HandleArgMinMaxKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto arg_min_max_param = reinterpret_cast<ArgMinMaxParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(arg_min_max_param);
  arg_min_max_param->axis_ = TransFormAxis(arg_min_max_param->axis_, trans);
  return RET_OK;
}

int HandleSoftMaxKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  // nnacl need transpose op_parameter but BaseOperator beed transpose Primitive
  auto param = reinterpret_cast<SoftmaxParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(param);
  param->axis_ = TransFormAxis(param->axis_, trans);
  return RET_OK;
}

int HandleSplitKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto param = reinterpret_cast<SplitParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(param);
  param->split_dim_ = TransFormAxis(param->split_dim_, trans);
  return RET_OK;
}

int HandleSqueezeKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto param = reinterpret_cast<SqueezeParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(param);
  for (size_t i = 0; i < param->axis_size_; i++) {
    param->axis_[i] = TransFormAxis(param->axis_[i], trans);
  }
  return RET_OK;
}

int HandleUnSqueezeKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto param = reinterpret_cast<UnSqueezeParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(param);
  param->axis_ = TransFormAxis(param->axis_, trans);
  return RET_OK;
}

int HandleStackKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto param = reinterpret_cast<StackParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(param);
  param->axis_ = TransFormAxis(param->axis_, trans);
  return RET_OK;
}

int HandleUnStackKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto param = reinterpret_cast<UnstackParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(param);
  param->axis_ = TransFormAxis(param->axis_, trans);
  return RET_OK;
}

int HandleConcatKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto concat_param = reinterpret_cast<ConcatParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(concat_param);
  concat_param->axis_ = TransFormAxis(concat_param->axis_, trans);
  return RET_OK;
}

namespace {
int Handle0AxisCrop(const TransInfoPair &trans, CropParameter *crop_param) {
  auto offset = crop_param->offset_;
  if (IsSameTranspose(trans, kNCHW2NHWCTrans)) {
    crop_param->offset_[kNHWC_N] = offset[kNCHW_N];
    crop_param->offset_[kNHWC_H] = offset[kNCHW_H];
    crop_param->offset_[kNHWC_W] = offset[kNCHW_W];
    crop_param->offset_[kNHWC_C] = offset[kNCHW_C];
    return RET_OK;
  }
  if (IsSameTranspose(trans, kNHWC2NCHWTrans)) {
    crop_param->offset_[kNCHW_N] = offset[kNHWC_N];
    crop_param->offset_[kNCHW_H] = offset[kNHWC_H];
    crop_param->offset_[kNCHW_W] = offset[kNHWC_W];
    crop_param->offset_[kNCHW_C] = offset[kNHWC_C];
    return RET_OK;
  }
  MS_LOG(ERROR) << "Unknown transpose info: from " << trans.src_format_ << " to " << trans.dst_format_;
  return RET_ERROR;
}

int Handle1AxisCrop(const TransInfoPair &trans, CropParameter *crop_param) {
  auto offset = crop_param->offset_;
  if (IsSameTranspose(trans, kNCHW2NHWCTrans)) {
    crop_param->offset_[kNHWC_H - 1] = offset[kNCHW_H - 1];
    crop_param->offset_[kNHWC_W - 1] = offset[kNCHW_W - 1];
    crop_param->offset_[kNHWC_C - 1] = offset[kNCHW_C - 1];
    return RET_OK;
  }
  if (IsSameTranspose(trans, kNHWC2NCHWTrans)) {
    crop_param->offset_[kNCHW_H - 1] = offset[kNHWC_H - 1];
    crop_param->offset_[kNCHW_W - 1] = offset[kNHWC_W - 1];
    crop_param->offset_[kNCHW_C - 1] = offset[kNHWC_C - 1];
    return RET_OK;
  }
  MS_LOG(ERROR) << "Unknown transpose info: from " << trans.src_format_ << " to " << trans.dst_format_;
  return RET_ERROR;
}

int Handle2AxisCrop(const TransInfoPair &trans, CropParameter *crop_param, const lite::Tensor &input_tensor,
                    const lite::Tensor &shape_tensor) {
  if (IsSameTranspose(trans, kNCHW2NHWCTrans)) {
    if (input_tensor.Channel() != shape_tensor.Channel()) {
      return RET_NO_CHANGE;
    }
    crop_param->offset_[4 - crop_param->axis_] = 0;
    crop_param->axis_ = crop_param->axis_ - 1;
    crop_param->offset_size_ = crop_param->offset_size_ + 1;
    return RET_OK;
  }
  auto offset1 = crop_param->offset_[1];
  auto offset0 = crop_param->offset_[0];
  if (IsSameTranspose(trans, kNHWC2NCHWTrans)) {
    if (input_tensor.Height() != shape_tensor.Height()) {
      return RET_NO_CHANGE;
    }
    crop_param->axis_ = 1;
    crop_param->offset_size_ = 3;
    crop_param->offset_[0] = offset1;
    crop_param->offset_[1] = 0;
    crop_param->offset_[2] = offset0;
    return RET_OK;
  }
  MS_LOG(ERROR) << "Unknown transpose info: from " << trans.src_format_ << " to " << trans.dst_format_;
  return RET_ERROR;
}

int Handle3AxisCrop(const TransInfoPair &trans, CropParameter *crop_param, const lite::Tensor &input_tensor,
                    const lite::Tensor &shape_tensor) {
  if (IsSameTranspose(trans, kNCHW2NHWCTrans)) {
    if (input_tensor.Channel() != shape_tensor.Channel()) {
      return RET_NO_CHANGE;
    }
    crop_param->offset_[4 - crop_param->axis_] = 0;
    crop_param->axis_ = crop_param->axis_ - 1;
    crop_param->offset_size_ = crop_param->offset_size_ + 1;
    return RET_OK;
  }
  auto offset0 = crop_param->offset_[0];
  if (IsSameTranspose(trans, kNHWC2NCHWTrans)) {
    if (input_tensor.Height() != shape_tensor.Height() || input_tensor.Width() != shape_tensor.Width()) {
      return RET_NO_CHANGE;
    }
    crop_param->axis_ = 1;
    crop_param->offset_size_ = 3;
    crop_param->offset_[0] = offset0;
    crop_param->offset_[1] = 0;
    crop_param->offset_[2] = 0;
    return RET_OK;
  }
  MS_LOG(ERROR) << "Unknown transpose info: from " << trans.src_format_ << " to " << trans.dst_format_;
  return RET_ERROR;
}
}  // namespace

int HandleCropKernel(const kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto crop_param = reinterpret_cast<CropParameter *>(kernel->op_parameter());
  CHECK_NULL_RETURN(crop_param);
  auto inputs = kernel->in_tensors();
  for (const auto &input : inputs) {
    auto shape = input->shape();
    if (shape.size() != DIMENSION_4D) {
      return RET_NO_CHANGE;
    }
    if (std::any_of(shape.begin(), shape.end(), [](const int &dim) { return dim < 0; })) {
      return RET_NO_CHANGE;
    }
  }
  if (crop_param->axis_ == 0) {
    return Handle0AxisCrop(trans, crop_param);
  }
  if (crop_param->axis_ == 1) {
    return Handle1AxisCrop(trans, crop_param);
  }
  if (crop_param->axis_ == 2) {
    return Handle2AxisCrop(trans, crop_param, *inputs[0], *inputs[1]);
  }
  if (crop_param->axis_ == 3) {
    return Handle3AxisCrop(trans, crop_param, *inputs[0], *inputs[1]);
  }
  MS_LOG(ERROR) << "axis of parameter of Crop out of range, input dimension: 4, axis: " << crop_param->axis_;
  return RET_ERROR;
}

// bool value determines whether the kernel has axis attribute or not.
// If bool value is true, the single kernel can be processd only for NHWC2NCHW or NCHW2NHWC.
static const std::unordered_map<schema::PrimitiveType, TransAxisFunc> kTransAxisFuncs = {
  {schema::PrimitiveType_Abs, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Activation, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_AddFusion, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_AddN, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_ArgMaxFusion, HandleArgMinMaxKernel},
  {schema::PrimitiveType_ArgMinFusion, HandleArgMinMaxKernel},
  {schema::PrimitiveType_Cast, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Ceil, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Clip, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Concat, HandleConcatKernel},
  {schema::PrimitiveType_Cos, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Crop, HandleCropKernel},
  {schema::PrimitiveType_DivFusion, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Elu, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Eltwise, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Equal, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_ExpFusion, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Floor, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_FloorDiv, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_FloorMod, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Greater, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_GreaterEqual, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Less, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_LessEqual, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Log, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_LogicalAnd, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_LogicalNot, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_LogicalOr, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Maximum, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Minimum, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Mod, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_MulFusion, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Neg, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_NotEqual, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_PowFusion, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_QuantDTypeCast, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_RealDiv, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Round, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Rsqrt, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Sin, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_SliceFusion, kNotImplementedTransAxisFunc},
  {schema::PrimitiveType_Softmax, HandleSoftMaxKernel},
  {schema::PrimitiveType_Split, HandleSplitKernel},
  {schema::PrimitiveType_Sqrt, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Squeeze, HandleSqueezeKernel},
  {schema::PrimitiveType_Square, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_SquaredDifference, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Stack, HandleStackKernel},
  {schema::PrimitiveType_StridedSlice, kNotImplementedTransAxisFunc},
  {schema::PrimitiveType_SubFusion, kNoNeedTransAxisFunc},
  {schema::PrimitiveType_Unsqueeze, HandleUnSqueezeKernel},
  {schema::PrimitiveType_Unstack, HandleUnStackKernel},
  {schema::PrimitiveType_LogSoftmax, HandleSoftMaxKernel},
  {schema::PrimitiveType_Erf, kNoNeedTransAxisFunc},
};
}  // namespace

bool TransposeStrategy::CrossKernelFusionPreCheck(const kernel::KernelExec *kernel, TransInfoPair *pre_trans,
                                                  TransInfoPair *post_trans) {
  if (kTransAxisFuncs.find(kernel::SchemaType(kernel->type())) == kTransAxisFuncs.end()) {
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
    MS_LOG(DEBUG) << "The fusion can't decrease transpose op number.";
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

  if (((!IsSameTranspose(*post_trans, kNCHW2NHWCTrans)) && (!IsSameTranspose(*post_trans, kNHWC2NCHWTrans))) &&
      kTransAxisFuncs.at(kernel::SchemaType(kernel->type()))) {
    return false;
  }
  if (!CheckInTensorsShape(kernel, (Format)(post_trans->dst_format_))) {
    return false;
  }
  return true;
}

int TransposeStrategy::TryTransKernelAxis(kernel::KernelExec *kernel, const TransInfoPair &trans) {
  auto trans_axis_func = kTransAxisFuncs.find(kernel::SchemaType(kernel->type()));
  if (trans_axis_func == kTransAxisFuncs.end() || trans_axis_func->second == nullptr) {
    MS_LOG(ERROR) << "Can't find the axis change function for " << kernel->name();
    return RET_ERROR;
  }
  return trans_axis_func->second(kernel, trans);
}
}  // namespace mindspore::lite::pass
