/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/adder_fp32.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "nnacl/fp32/adder_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::Format;
using mindspore::schema::PrimitiveType_AdderFusion;

namespace mindspore::kernel {
int AdderCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int AdderCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  return RET_OK;
}

int AdderCPUKernel::InitWeightBias() {
  CHECK_NULL_RETURN(conv_param_);
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int kernel_h = filter_tensor->Height();
  int kernel_w = filter_tensor->Width();
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(kernel_h, kernel_w), RET_ERROR);
  int kernel_hw = kernel_h * kernel_w;
  const int oc_block = C4NUM;
  int oc_block_num = UP_DIV(out_channel, C4NUM);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(in_channel, kernel_hw, RET_ERROR);
  int kernel_chw = in_channel * kernel_hw;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(oc_block_num * oc_block, kernel_chw, RET_ERROR);
  int pack_weight_size = oc_block_num * oc_block * kernel_chw;

  auto origin_weight = reinterpret_cast<float *>(filter_tensor->MutableData());
  CHECK_NULL_RETURN(origin_weight);
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
  packed_weight_ = GetConvPackWeightData(pack_weight_size * sizeof(float));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_ERROR;
  }
  RowMajor2Col4Major(origin_weight, reinterpret_cast<float *>(packed_weight_), out_channel, in_channel * kernel_hw);
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, oc_block_num * oc_block * sizeof(float));
  bias_data_ = reinterpret_cast<float *>(malloc(oc_block_num * oc_block * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * oc_block * sizeof(float));

  if (in_tensors_.size() == kInputSize2) {
    CHECK_NULL_RETURN(in_tensors_.at(kBiasIndex));
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->MutableData());
    CHECK_NULL_RETURN(ori_bias);
    MS_CHECK_TRUE_MSG(in_tensors_.at(kBiasIndex)->Size() == static_cast<size_t>(out_channel) * sizeof(float), RET_ERROR,
                      "bias is invalid.");
    memcpy(bias_data_, ori_bias, out_channel * sizeof(float));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int AdderCPUKernel::RunImpl(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  MS_ASSERT(input_tensor != nullptr);
  auto ori_input_data = reinterpret_cast<float *>(input_tensor->MutableData());
  CHECK_NULL_RETURN(ori_input_data);
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  CHECK_NULL_RETURN(output_addr);
  AdderFp32(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
            reinterpret_cast<float *>(bias_data_), col_major_input_, output_addr, task_id, conv_param_);
  return RET_OK;
}

int AdderImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto adder = reinterpret_cast<AdderCPUKernel *>(cdata);
  auto error_code = adder->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Adder Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AdderCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }

  int error_code = ParallelLaunch(this->ms_context_, AdderImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "adder error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  FreeTmpBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AdderFusion, LiteKernelCreator<AdderCPUKernel>)
}  // namespace mindspore::kernel
