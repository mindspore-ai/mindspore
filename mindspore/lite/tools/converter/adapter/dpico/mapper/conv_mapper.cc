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

#include "mapper/conv_mapper.h"
#include <memory>
#include <vector>
#include <utility>
#include "common/op_enum.h"
#include "ops/fusion/conv2d_fusion.h"
#include "op/conv_operator.h"
#include "op/depthwiseconv_operator.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kconvKernelSize = 2;
constexpr int kconvStrideSize = 2;
constexpr int kconvDilationSize = 2;
constexpr int kconvPadListSize = 4;
struct ConvAttr {
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t group{1};
  int64_t out_channel{1};
  PadMode pad_mode{PadMode::PAD};
};
STATUS GetConvAttrFromPrimitive(ConvAttr *conv_attr, const api::SharedPtr<ops::Conv2DFusion> &conv_prim) {
  if (conv_attr == nullptr) {
    MS_LOG(ERROR) << "conv_attr is nullptr.";
    return RET_ERROR;
  }
  if (conv_prim->GetAttr(ops::kKernelSize) != nullptr) {
    conv_attr->kernel_size = conv_prim->get_kernel_size();
  } else {
    MS_LOG(ERROR) << "kernel_size attr doesn't exist.";
    return RET_ERROR;
  }
  if (conv_prim->GetAttr(ops::kStride) != nullptr) {
    conv_attr->stride = conv_prim->get_stride();
  } else {
    MS_LOG(ERROR) << "stride attr doesn't exist.";
    return RET_ERROR;
  }
  if (conv_prim->GetAttr(ops::kDilation) != nullptr) {
    conv_attr->dilation = conv_prim->get_dilation();
  } else {
    MS_LOG(ERROR) << "dilation attr doesn't exist.";
    return RET_ERROR;
  }
  if (conv_prim->GetAttr(ops::kGroup) != nullptr) {
    conv_attr->group = conv_prim->get_group();
  } else {
    MS_LOG(ERROR) << "group attr doesn't exist.";
    return RET_ERROR;
  }
  if (conv_prim->GetAttr(ops::kPadMode) != nullptr) {
    conv_attr->pad_mode = conv_prim->get_pad_mode();
  } else {
    MS_LOG(ERROR) << "pad_mode attr doesn't exist.";
    return RET_ERROR;
  }
  if (conv_prim->GetAttr(ops::kOutChannel) != nullptr) {
    conv_attr->out_channel = conv_prim->get_out_channel();
  } else {
    MS_LOG(ERROR) << "out_channel attr doesn't exist.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace
STATUS ConvMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto conv_prim = api::utils::cast<api::SharedPtr<ops::Conv2DFusion>>(prim);
  MS_ASSERT(conv_prim != nullptr);
  ConvAttr conv_attr;
  if (GetConvAttrFromPrimitive(&conv_attr, conv_prim) != RET_OK) {
    MS_LOG(ERROR) << "get conv attr from primitive failed.";
    return RET_ERROR;
  }

  if (conv_attr.kernel_size.size() != kconvKernelSize) {
    MS_LOG(ERROR) << "kernel_size should be 2 dims, which is " << conv_attr.kernel_size.size();
    return RET_ERROR;
  }
  if (conv_attr.stride.size() != kconvStrideSize) {
    MS_LOG(ERROR) << "stride should be 2 dims, which is " << conv_attr.stride.size();
    return RET_ERROR;
  }
  if (conv_attr.dilation.size() != kconvDilationSize) {
    MS_LOG(ERROR) << "dilation should be 2 dims, which is " << conv_attr.dilation.size();
    return RET_ERROR;
  }

  std::unique_ptr<mapper::ConvOperator> conv_operator;
  if (conv_prim->GetAttr(ops::kIsDepthWise) != nullptr && api::GetValue<bool>(conv_prim->GetAttr(ops::kIsDepthWise))) {
    conv_operator = std::make_unique<mapper::DepthwiseconvOperator>();
    if (conv_operator == nullptr) {
      MS_LOG(ERROR) << "conv_operator is nullptr.";
      return RET_ERROR;
    }
    conv_operator->SetGroup(static_cast<uint32_t>(conv_attr.group));
  } else {
    conv_operator = std::make_unique<mapper::ConvOperator>();
    if (conv_operator == nullptr) {
      MS_LOG(ERROR) << "conv_operator is nullptr.";
      return RET_ERROR;
    }
    conv_operator->SetGroup(static_cast<uint32_t>(conv_attr.group));
  }

  if (SetCommonAttr(cnode, conv_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  conv_operator->SetOpType(mapper::OpType::CONVOLUTION);

  conv_operator->SetOutputChannel(static_cast<uint32_t>(conv_attr.out_channel));
  conv_operator->SetKernelHeight(static_cast<uint32_t>(conv_attr.kernel_size[0]));
  conv_operator->SetKernelWidth(static_cast<uint32_t>(conv_attr.kernel_size[1]));
  conv_operator->SetStrideHeight(static_cast<uint32_t>(conv_attr.stride[0]));
  conv_operator->SetStrideWidth(static_cast<uint32_t>(conv_attr.stride[1]));
  conv_operator->SetDilationHeight(static_cast<uint32_t>(conv_attr.dilation[0]));
  conv_operator->SetDilationWidth(static_cast<uint32_t>(conv_attr.dilation[1]));
  if (conv_attr.pad_mode == PadMode::PAD) {
    auto pad_list = conv_prim->get_pad_list();
    if (pad_list.size() != kconvPadListSize) {
      MS_LOG(ERROR) << "pad_list size is invalid. " << pad_list.size();
      return RET_ERROR;
    }
    conv_operator->SetPadUp(static_cast<int>(pad_list[0]));
    conv_operator->SetPadDown(static_cast<int>(pad_list[1]));
    conv_operator->SetPadLeft(static_cast<int>(pad_list[kAxis2]));
    conv_operator->SetPadRight(static_cast<int>(pad_list[kAxis3]));
  } else if (conv_attr.pad_mode == PadMode::SAME) {
    conv_operator->SetAutoPadType(mapper::AutoPadType::PAD_SAME_UPPER);
  } else if (conv_attr.pad_mode == PadMode::VALID) {
    conv_operator->SetAutoPadType(mapper::AutoPadType::PAD_VALID);
  } else {
    MS_LOG(ERROR) << "Non supported pad mode. " << conv_attr.pad_mode;
    return RET_ERROR;
  }

  if (SetConvFcDataInfo(cnode, conv_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set conv data info failed.";
    return RET_ERROR;
  }
  if (PushOfflineArgs(cnode, conv_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(conv_operator));
  return RET_OK;
}
REG_MAPPER(Conv2DFusion, ConvMapper)
}  // namespace dpico
}  // namespace mindspore
