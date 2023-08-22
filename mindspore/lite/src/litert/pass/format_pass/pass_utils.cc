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

#include "src/litert/pass/format_pass/pass_utils.h"
#include <string>
#include <vector>
#include "nnacl/format_transpose_parameter.h"
#include "nnacl/arg_min_max_parameter.h"

namespace mindspore::lite::pass {
bool IsNoneTranspose(const TransInfoPair &trans) {
  return trans.src_format_ == Format::DEFAULT_FORMAT && trans.dst_format_ == Format::DEFAULT_FORMAT;
}

bool IsSameTranspose(const TransInfoPair &trans0, const TransInfoPair &trans1) {
  if (!IsNoneTranspose(trans0) && !IsNoneTranspose(trans1)) {
    return trans0.src_format_ == trans1.src_format_ && trans0.dst_format_ == trans1.dst_format_;
  }
  return false;
}

bool IsOppositiveTranspose(const TransInfoPair &trans0, const TransInfoPair &trans1) {
  if (!IsNoneTranspose(trans0) && IsNoneTranspose(trans1)) {
    return true;
  } else if (IsNoneTranspose(trans0) && !IsNoneTranspose(trans1)) {
    return true;
  } else if (!IsNoneTranspose(trans0) && !IsNoneTranspose(trans1)) {
    return trans0.src_format_ == trans1.dst_format_ && trans0.dst_format_ == trans1.src_format_;
  } else {
    return false;
  }
}

bool SetShape(const Tensor *src_tensor, Tensor *dst_tensor) {
  auto shape = src_tensor->shape();
  if (shape.size() != DIMENSION_4D) {
    dst_tensor->set_shape(shape);
    return true;
  }
  if (std::any_of(shape.begin(), shape.end(), [](int dim) { return dim == -1; })) {
    dst_tensor->set_shape({-1});
    return true;
  }
  bool ret;
  auto new_shape = TransShape(src_tensor->shape(), {src_tensor->format(), dst_tensor->format()}, &ret);
  if (!ret) {
    MS_LOG(ERROR) << "Transpose shape of tensor failed";
    return false;
  }
  dst_tensor->set_shape(new_shape);
  return true;
}

bool SetShape4D(const Tensor *src_tensor, Tensor *dst_tensor) {
  auto shape = src_tensor->shape();
  auto invalid_shape = {-1};
  if (shape.size() != DIMENSION_4D) {
    dst_tensor->set_shape(invalid_shape);
    return true;
  }
  return SetShape(src_tensor, dst_tensor);
}

bool TransTensorShapeAndFormat(Tensor *tensor, Format dst_format) {
  auto shape = tensor->shape();
  if (shape.size() != DIMENSION_4D) {
    tensor->set_shape(shape);
    return true;
  }
  bool ret;
  auto new_shape = TransShape(tensor->shape(), {tensor->format(), dst_format}, &ret);
  if (!ret) {
    MS_LOG(ERROR) << "Transpose shape of tensor failed";
    return false;
  }
  tensor->set_shape(new_shape);
  tensor->set_format(dst_format);
  return true;
}

int InsertPreTranspose(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel, std::vector<Tensor *> *all_tensors,
                       const TransInfoPair &trans_info, const size_t &index, const CreateFormatTransposeFunc &func) {
  if (func == nullptr) {
    MS_LOG(ERROR) << "CreateFormatTransposeFunc is nullptr.";
    return RET_INPUT_PARAM_INVALID;
  }
  auto trans_name = kernel->name() + "_pre_" + std::to_string(index);
  auto in_tensor = kernel->in_tensors().at(index);
  auto in_tensor_shape = in_tensor->shape();
  if (std::all_of(in_tensor_shape.begin(), in_tensor_shape.end(), [](const int &dim) { return dim >= 0; }) &&
      in_tensor_shape.size() != DIMENSION_4D) {
    MS_LOG(INFO) << index << "th input tensor of kernel " << kernel->name()
                 << " is infershaped and do not have 4 dimensions, skip inserting transpose kernel.";
    return RET_OK;
  }
  auto out_tensor = new (std::nothrow) Tensor(in_tensor->data_type(), {}, (Format)trans_info.dst_format_);
  CHECK_NULL_RETURN(out_tensor);
  out_tensor->set_tensor_name(trans_name + "_output");
  if (!SetShape4D(in_tensor, out_tensor)) {
    MS_LOG(ERROR) << "Sync shape from in_tensor to out_tensor failed.";
    delete out_tensor;
    return RET_ERROR;
  }

  auto trans_kernel = func(in_tensor, out_tensor, trans_info, trans_name, kernel->Context(), kernel->desc());
  if (trans_kernel == nullptr) {
    delete out_tensor;
    return RET_NULL_PTR;
  }

  all_tensors->push_back(out_tensor);
  subgraph->InsertInEdge(kernel, trans_kernel, index);
  return RET_OK;
}

int InsertPostTranspose(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel,
                        std::vector<Tensor *> *all_tensors, const TransInfoPair &trans_info, const size_t &index,
                        const CreateFormatTransposeFunc &func) {
  if (func == nullptr) {
    MS_LOG(ERROR) << "CreateFormatTransposeFunc is nullptr.";
    return RET_INPUT_PARAM_INVALID;
  }
  auto trans_name = kernel->name() + "_post_" + std::to_string(index);

  auto out_tensor = kernel->out_tensors().at(index);
  auto out_tensor_shape = out_tensor->shape();
  if (std::all_of(out_tensor_shape.begin(), out_tensor_shape.end(), [](const int &dim) { return dim >= 0; }) &&
      out_tensor_shape.size() != DIMENSION_4D) {
    MS_LOG(INFO) << index << "th output tensor of kernel " << kernel->name()
                 << " is infershaped and do not have 4 dimensions, skip inserting transpose kernel.";
    return RET_OK;
  }
  auto in_tensor = new (std::nothrow) Tensor(out_tensor->data_type(), {}, (Format)trans_info.src_format_);
  CHECK_NULL_RETURN(in_tensor);
  in_tensor->set_tensor_name(trans_name + "_input");
  if (!SetShape4D(out_tensor, in_tensor)) {
    MS_LOG(ERROR) << "Sync shape from in_tensor to out_tensor failed.";
    delete out_tensor;
    return RET_ERROR;
  }

  auto trans_kernel = func(in_tensor, out_tensor, trans_info, trans_name, kernel->Context(), kernel->desc());
  if (trans_kernel == nullptr) {
    delete out_tensor;
    return RET_NULL_PTR;
  }

  all_tensors->push_back(in_tensor);
  subgraph->InsertOutEdge(kernel, trans_kernel, index);
  return RET_OK;
}

int GetTransposeInfo(const kernel::KernelExec *kernel, TransInfoPair *trans_info) {
  CHECK_NULL_RETURN(kernel);
  if (kernel->type() != schema::PrimitiveType_Transpose && kernel->type() != schema::PrimitiveType_FormatTranspose) {
    return RET_INVALID_OP_ATTR;
  }
  if (kernel->type() == schema::PrimitiveType_Transpose) {
    CHECK_LESS_RETURN(kernel->in_tensors().size(), FormatTransposeInput);
    auto perm_tensor = kernel->in_tensors().at(1);
    CHECK_NULL_RETURN(perm_tensor);
    if (perm_tensor->ElementsNum() != DIMENSION_4D || perm_tensor->data_type() != kNumberTypeInt32) {
      return RET_INVALID_OP_ATTR;
    }
    auto perm_data = reinterpret_cast<int *>(perm_tensor->data());
    CHECK_NULL_RETURN(perm_data);
    std::vector<int> perm;
    for (int i = 0; i < perm_tensor->ElementsNum(); i++) {
      perm.push_back(perm_data[i]);
    }
    if (perm == nc2nh_perm) {
      trans_info->src_format_ = NCHW;
      trans_info->dst_format_ = NHWC;
    } else if (perm == nh2nc_perm) {
      trans_info->src_format_ = NHWC;
      trans_info->dst_format_ = NCHW;
    } else {
      return RET_INVALID_OP_ATTR;
    }
  }
  if (kernel->type() == schema::PrimitiveType_FormatTranspose) {
    auto param = reinterpret_cast<FormatTransposeParameter *>(kernel->op_parameter());
    CHECK_NULL_RETURN(param);
    trans_info->src_format_ = static_cast<Format>((param->src_format_));
    trans_info->dst_format_ = static_cast<Format>((param->dst_format_));
  }
  return RET_OK;
}
}  // namespace mindspore::lite::pass
