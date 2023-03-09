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
#include <map>
#include <vector>
#include <memory>
#include "src/litert/kernel_registry.h"
#include "nnacl/format_transpose_parameter.h"
#include "nnacl/arg_min_max_parameter.h"

namespace mindspore::lite::pass {
static const std::map<FormatC, std::string> format_str = {
  {Format_NCHW, "NCHW"}, {Format_NHWC, "NHWC"}, {Format_NC4HW4, "NC4HW4"}, {Format_NCHW, "NC8HW8"}};

std::string GetFormatStr(const FormatC &format) {
  return format_str.find(format) != format_str.end() ? format_str.at(format) : "";
}

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

kernel::KernelExec *CreateFormatTranspose(Tensor *input, Tensor *output, const TransInfoPair &trans_info,
                                          const std::string &name, const lite::InnerContext *ctx,
                                          const kernel::KernelKey &desc) {
  auto param = reinterpret_cast<FormatTransposeParameter *>(malloc(sizeof(FormatTransposeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Malloc FormatTransposeParameter failed.";
    return nullptr;
  }
  (void)memset(param, 0, sizeof(FormatTransposeParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_FormatTranspose);
  param->src_format_ = static_cast<FormatC>((trans_info.src_format_));
  param->dst_format_ = static_cast<FormatC>((trans_info.dst_format_));
  kernel::KernelKey format_transpose_key = desc;
  format_transpose_key.type = schema::PrimitiveType_FormatTranspose;
  format_transpose_key.format = NHWC;

  kernel::KernelExec *kernel = nullptr;
  auto ret = KernelRegistry::GetInstance()->GetKernelExec({input}, {output}, ctx, nullptr, format_transpose_key,
                                                          reinterpret_cast<OpParameter *>(param), &kernel);
  if (ret != RET_OK || kernel == nullptr) {
    free(param);
    return nullptr;
  }
  kernel->set_name(name);
  return kernel;
}

void SetShape(const Tensor *src_tensor, Tensor *dst_tensor) {
  auto shape = src_tensor->shape();
  auto invalid_shape = {-1};
  if (shape.size() != DIMENSION_4D) {
    dst_tensor->set_shape(invalid_shape);
    return;
  }
  if (std::any_of(shape.begin(), shape.end(), [](int dim) { return dim == -1; })) {
    dst_tensor->set_shape(invalid_shape);
    return;
  }
  auto batch = src_tensor->Batch();
  auto height = src_tensor->Height();
  auto width = src_tensor->Width();
  auto channel = src_tensor->Channel();
  if (dst_tensor->format() == NHWC) {
    dst_tensor->set_shape({batch, height, width, channel});
  }
  if (dst_tensor->format() == NCHW || dst_tensor->format() == NC4HW4 || dst_tensor->format() == NC8HW8) {
    dst_tensor->set_shape({batch, channel, height, width});
  }
  return;
}

int InsertPreTranspose(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel, std::vector<Tensor *> *all_tensors,
                       const TransInfoPair &trans_info, const size_t &index) {
  auto trans_name = kernel->name() + "_pre_" + std::to_string(index);
  auto in_tensor = kernel->in_tensors().at(index);
  auto out_tensor = new (std::nothrow) Tensor(in_tensor->data_type(), {}, (Format)trans_info.dst_format_);
  CHECK_NULL_RETURN(out_tensor);
  out_tensor->set_tensor_name(trans_name + "_output");
  SetShape(in_tensor, out_tensor);

  auto trans_kernel =
    CreateFormatTranspose(in_tensor, out_tensor, trans_info, trans_name, kernel->Context(), kernel->desc());
  if (trans_kernel == nullptr) {
    delete out_tensor;
    return RET_NULL_PTR;
  }

  all_tensors->push_back(out_tensor);
  subgraph->InsertInEdge(kernel, trans_kernel, index);
  return RET_OK;
}

int InsertPostTranspose(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel,
                        std::vector<Tensor *> *all_tensors, const TransInfoPair &trans_info, const size_t &index) {
  auto trans_name = kernel->name() + "_post_" + std::to_string(index);

  auto out_tensor = kernel->out_tensors().at(index);
  auto in_tensor = new (std::nothrow) Tensor(out_tensor->data_type(), {}, (Format)trans_info.src_format_);
  CHECK_NULL_RETURN(in_tensor);
  in_tensor->set_tensor_name(trans_name + "_input");
  SetShape(out_tensor, in_tensor);

  auto trans_kernel =
    CreateFormatTranspose(in_tensor, out_tensor, trans_info, trans_name, kernel->Context(), kernel->desc());
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
