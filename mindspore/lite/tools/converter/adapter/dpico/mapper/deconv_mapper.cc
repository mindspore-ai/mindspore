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

#include "mapper/deconv_mapper.h"
#include <memory>
#include <vector>
#include <utility>
#include "common/op_enum.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "op/deconv_operator.h"

namespace mindspore {
namespace dpico {
constexpr int kDeconvKernelSize = 2;
constexpr int kDeconvStrideSize = 2;
constexpr int kDeconvDilationSize = 2;
constexpr int kDeconvPadListSize = 4;
STATUS DeconvMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto deconv_prim = api::utils::cast<api::SharedPtr<ops::Conv2dTransposeFusion>>(prim);
  MS_ASSERT(deconv_prim != nullptr);
  auto kernel_size = deconv_prim->get_kernel_size();
  auto stride = deconv_prim->get_stride();
  auto dilation = deconv_prim->get_dilation();
  auto pad_mode = deconv_prim->get_pad_mode();

  if (kernel_size.size() != kDeconvKernelSize) {
    MS_LOG(ERROR) << "kernel_size should be 2 dims, which is " << kernel_size.size();
    return RET_ERROR;
  }
  if (stride.size() != kDeconvStrideSize) {
    MS_LOG(ERROR) << "stride should be 2 dims, which is " << stride.size();
    return RET_ERROR;
  }
  if (dilation.size() != kDeconvDilationSize) {
    MS_LOG(ERROR) << "dilation should be 2 dims, which is " << dilation.size();
    return RET_ERROR;
  }

  auto deconv_operator = std::make_unique<mapper::DeconvOperator>();
  if (deconv_operator == nullptr) {
    MS_LOG(ERROR) << "deconv_operator is nullptr.";
    return RET_ERROR;
  }
  deconv_operator->SetGroup(static_cast<uint32_t>(deconv_prim->get_group()));

  if (SetCommonAttr(cnode, deconv_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  deconv_operator->SetOpType(mapper::OpType::DECONVOLUTION);
  deconv_operator->SetOutputChannel(static_cast<uint32_t>(deconv_prim->get_out_channel()));
  deconv_operator->SetKernelHeight(static_cast<uint32_t>(kernel_size[0]));
  deconv_operator->SetKernelWidth(static_cast<uint32_t>(kernel_size[1]));
  deconv_operator->SetStrideHeight(static_cast<uint32_t>(stride[0]));
  deconv_operator->SetStrideWidth(static_cast<uint32_t>(stride[1]));
  deconv_operator->SetDilationHeight(static_cast<uint32_t>(dilation[0]));
  deconv_operator->SetDilationWidth(static_cast<uint32_t>(dilation[1]));
  if (pad_mode == PadMode::PAD) {
    auto pad_list = deconv_prim->get_pad_list();
    if (pad_list.size() != kDeconvPadListSize) {
      MS_LOG(ERROR) << "pad_list size is invalid. " << pad_list.size();
      return RET_ERROR;
    }
    deconv_operator->SetPadUp(static_cast<int>(pad_list[0]));
    deconv_operator->SetPadDown(static_cast<int>(pad_list[1]));
    deconv_operator->SetPadLeft(static_cast<int>(pad_list[kAxis2]));
    deconv_operator->SetPadRight(static_cast<int>(pad_list[kAxis3]));
  } else if (pad_mode == PadMode::SAME) {
    deconv_operator->SetAutoPadType(mapper::AutoPadType::PAD_SAME_UPPER);
  } else if (pad_mode == PadMode::VALID) {
    deconv_operator->SetAutoPadType(mapper::AutoPadType::PAD_VALID);
  } else {
    MS_LOG(ERROR) << "Non supported pad mode. " << pad_mode;
    return RET_ERROR;
  }

  if (SetConvFcDataInfo(cnode, deconv_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set deconv data info failed.";
    return RET_ERROR;
  }

  base_operators->push_back(std::move(deconv_operator));
  return RET_OK;
}
REG_MAPPER(Conv2dTransposeFusion, DeconvMapper)
}  // namespace dpico
}  // namespace mindspore
