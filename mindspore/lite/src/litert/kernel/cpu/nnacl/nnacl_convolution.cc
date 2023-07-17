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
#include "nnacl/conv_parameter.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::nnacl {
int ConvolutionKernel::Prepare() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }

  ConvolutionBaseStruct *conv = reinterpret_cast<ConvolutionBaseStruct *>(kernel_);
  conv->shaing_manager_ = lite::PackWeightManager::GetInstance();
  conv->get_sharing_weight_ = nnacl::DefaultGetSharingPackData;
  conv->free_sharing_weight_ = nnacl::DefaultFreeSharingPackData;
  conv->infershape_done_ = InferShapeDone();
  conv->is_sharing_pack_ = true;

  return NNACLKernel::Prepare();
}

int ConvolutionKernel::ReSize() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }

  ConvolutionBaseStruct *conv = reinterpret_cast<ConvolutionBaseStruct *>(kernel_);
  conv->infershape_done_ = InferShapeDone();

  return NNACLKernel::ReSize();
}

NNACLKernel *NNACLConvolutionOpt(OpParameter *parameter, const std::vector<lite::Tensor *> &in,
                                 const std::vector<lite::Tensor *> &out, const lite::InnerContext *ctx) {
  reinterpret_cast<ConvParameter *>(parameter)->thread_num_ = ctx->thread_num_;
  auto shape = out.front()->shape();
  reinterpret_cast<ConvParameter *>(parameter)->dynamic_shape_ =
    std::find(shape.begin(), shape.end(), -1) != shape.end();

  auto *kernel = new (std::nothrow) ConvolutionKernel(parameter, in, out, ctx);
  return kernel;
}

NNACL_KERNEL(PrimitiveType_Conv2DFusion, kNumberTypeFloat32, NNACLConvolutionOpt)
}  // namespace mindspore::nnacl
