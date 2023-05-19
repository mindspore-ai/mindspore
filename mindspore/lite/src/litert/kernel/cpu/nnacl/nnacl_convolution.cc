/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/nnacl_convolution.h"
#include "nnacl/cxx_utils.h"
#include "src/litert/pack_weight_manager.h"
#include "nnacl/nnacl_manager.h"
#include "nnacl/kernel/convolution_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::nnacl {
int ConvolutionKernel::Prepare() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }

  ConvolutionBaseStruct *conv = reinterpret_cast<ConvolutionBaseStruct *>(kernel_);
  conv->pack_weight_manager_ = lite::PackWeightManager::GetInstance();
  conv->get_pack_data_by_sharing_weight_ = nnacl::DefaultGetSharingPackData;
  conv->free_by_sharing_weight_ = nnacl::DefaultFreeSharingPackData;
  conv->is_sharing_pack_ = true;
  conv->infershape_done_ = InferShapeDone();

  int ret = kernel_->prepare(kernel_);
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

NNACL_KERNEL(PrimitiveType_Conv2DFusion, kNumberTypeFloat32, NNACLOpt<ConvolutionKernel>)
}  // namespace mindspore::nnacl
