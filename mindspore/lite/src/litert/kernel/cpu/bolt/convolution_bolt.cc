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

#include "bolt/convolution_bolt.h"
#include "bolt/bolt_kernel_manager.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/pack.h"
#include "bolt/compute/tensor/include/tensor_computing.h"
#include "bolt/common/memory/include/tensor_desc.h"
#include "bolt/bolt_tensor_utils.h"
#include "src/litert/pack_weight_manager.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::kernel::bolt {
ConvolutionBoltCPUKernel::~ConvolutionBoltCPUKernel() {
  lite::PackWeightManager::GetInstance()->Free(tile_weight_);
  lite::PackWeightManager::GetInstance()->Free(tile_bias_);
  lite::PackWeightManager::GetInstance()->Free(tmp_weight_);
  lite::PackWeightManager::GetInstance()->Free(pw_weight_);
}

int ConvolutionBoltCPUKernel::InitWeightBiasDesc() {
  dt_ = bolt_in_tensors_[kWeightIndex].get_desc().dt;
  auto in_channel = in_tensors_[kWeightIndex]->Channel();
  if (conv_param_spec_.convolution_type == CONVOLUTION_DILATION ||
      conv_param_spec_.convolution_type == CONVOLUTION_POINTWISE) {
    in_channel /= conv_param_spec_.group;
  }
  // to be delete, the input format will be NC8HW8 as default.
  if (in_tensors_[0]->format() == NC8HW8) {
    in_channel = UP_ROUND(in_channel, C8NUM);
  }

  std::vector<TensorDesc> filter_desc, bias_desc;
  size_t channel_axis = 3;
  filter_desc.push_back(
    tensor4d(dt_, conv_param_spec_.num_outputs, in_channel, conv_param_spec_.kernel_h, conv_param_spec_.kernel_w));
  if (CONVOLUTION_DEPTHWISE_POINTWISE == conv_param_spec_.convolution_type) {
    filter_desc.push_back(tensor4d(dt_, conv_param_spec_.num_outputs, in_channel, 1, 1));
  }

  std::vector<Tensor> filter_tensor(filter_desc.size());
  for (size_t i = 0; i < filter_desc.size(); i++) {
    filter_tensor[i].resize(filter_desc[i]);
  }
  switch (conv_param_spec_.convolution_type) {
    case CONVOLUTION_DILATION:
    case CONVOLUTION_POINTWISE: {
      bias_desc.push_back(tensor1d(dt_, conv_param_spec_.num_outputs));
      break;
    }
    case CONVOLUTION_DEPTHWISE: {
      filter_desc[0].dims[channel_axis] = 1;
      filter_tensor[0].resize(filter_desc[0]);
      bias_desc.push_back(tensor1d(dt_, conv_param_spec_.num_outputs));
      break;
    }
    case CONVOLUTION_DEPTHWISE_POINTWISE: {
      filter_desc[0].dims[channel_axis] = 1;
      filter_tensor[0].resize(filter_desc[0]);
      bias_desc.push_back(tensor1d(dt_, in_channel));
      bias_desc.push_back(tensor1d(dt_, conv_param_spec_.num_outputs));
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported convolution type for bolt convolution kernel.";
      return RET_NOT_SUPPORT;
  }
  if (weight_tensors_.empty()) {
    weight_tensors_ = filter_tensor;
  }
  if (bias_tensors_.empty()) {
    bias_tensors_ = std::vector<Tensor>(bias_desc.size());
    for (size_t i = 0; i < bias_desc.size(); i++) {
      bias_tensors_[i].resize(bias_desc[i]);
    }
  }
  return RET_OK;
}

int ConvolutionBoltCPUKernel::InitWeightBiasTensor() {
  CHECK_LESS_RETURN(in_tensors_.size(), Num3);
  auto weight_tensor = in_tensors_[kWeightIndex];
  CHECK_NULL_RETURN(weight_tensor);
  auto bias_tensor = in_tensors_[kBiasIndex];
  CHECK_NULL_RETURN(bias_tensor);
  if (!weight_tensor->IsConst() || !bias_tensor->IsConst()) {
    MS_LOG(ERROR) << "Bolt convolution kernel doesn't support dynamic weight right now.";
    return RET_NOT_SUPPORT;
  }
  auto ret = InitWeightBiasDesc();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight tensor and bias tensor TensorDesc failed.";
    return RET_ERROR;
  }
  auto out_channel = weight_tensor->Batch();
  auto in_channel = weight_tensor->Channel();
  auto kh = weight_tensor->Height();
  auto kw = weight_tensor->Width();

  // 1. weight NHWC -> NCHW, tmp buffer
  auto nchw_weight =
    lite::PackWeightManager::GetInstance()->GetPackData(nullptr, weight_tensor->Size(), &weight_is_packed_);
  if (nchw_weight == nullptr) {
    MS_LOG(ERROR) << "Malloc NCHW data for pack weight failed.";
    return RET_NULL_PTR;
  }
  PackNHWCToNCHWFp32(weight_tensor->data(), nchw_weight, out_channel, kh * kw, in_channel, 0, 0);

  // 2. weight NCHW -> tile 8, tmp buffer
  auto channel8 = UP_ROUND(out_channel, C8NUM);
  auto weight_tile8_size = channel8 * in_channel * kh * kw * lite::DataTypeSize(weight_tensor->data_type());
  tile_weight_ = lite::PackWeightManager::GetInstance()->GetPackData(nullptr, weight_tile8_size, &weight_is_packed_);
  if (tile_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc tile_8 data for pack weight failed.";
    return RET_NULL_PTR;
  }
  memset(tile_weight_, 0, weight_tile8_size);
  memcpy(tile_weight_, nchw_weight, weight_tensor->Size());
  std::shared_ptr<U8> weight_data(reinterpret_cast<U8 *>(tile_weight_), [](U8 *ptr) {});
  reinterpret_cast<CpuMemory *>(weight_tensors_[0].get_memory())->set_shared_ptr(weight_data);
  lite::PackWeightManager::GetInstance()->Free(nchw_weight);

  // bias -> tile 8, tmp buffer
  auto bias_tile8_size = channel8 * lite::DataTypeSize(bias_tensor->data_type());
  tile_bias_ = lite::PackWeightManager::GetInstance()->GetPackData(nullptr, bias_tile8_size, &weight_is_packed_);
  if (tile_bias_ == nullptr) {
    MS_LOG(ERROR) << "Malloc tile_8 data for pack bias failed.";
    return RET_NULL_PTR;
  }
  memset(tile_bias_, 0, bias_tile8_size);
  memcpy(tile_bias_, bias_tensor->data(), bias_tensor->Size());
  std::shared_ptr<U8> bias_data(reinterpret_cast<U8 *>(tile_bias_), [](U8 *ptr) {});
  reinterpret_cast<CpuMemory *>(bias_tensors_[0].get_memory())->set_shared_ptr(bias_data);
  return RET_OK;
}

int ConvolutionBoltCPUKernel::InferForwardAlgorithm() {
  auto input_tensor = bolt_in_tensors_[0];
  auto filter_tensor = weight_tensors_[0];
  auto output_tensor = bolt_out_tensors_[0];
  TensorDesc ori_input_desc = input_tensor.get_desc();
  TensorDesc ori_output_desc = output_tensor.get_desc();
  TensorDesc input_desc = transformDescTo4d(ori_input_desc);
  input_tensor.resize(input_desc);
  TensorDesc output_desc = transformDescTo4d(ori_output_desc);
  output_tensor.resize(output_desc);
  TensorDesc filter_desc = filter_tensor.get_desc();

  ConvolutionPolicy policy = CONVOLUTION_FASTEST;
  switch (conv_param_spec_.convolution_type) {
    case CONVOLUTION_DILATION:
    case CONVOLUTION_POINTWISE: {
      auto b_ret = convolution_infer_forward_algorithm(input_tensor, filter_tensor, output_tensor, conv_param_spec_,
                                                       policy, &(pw_alg_), filter_desc.dt, pw_act_param_, &arch_info_);
      if (b_ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt convolution infer forward algorithm failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE: {
      auto b_ret =
        depthwise_convolution_infer_forward_algorithm(input_tensor, filter_tensor, output_tensor, conv_param_spec_,
                                                      policy, &(dw_alg_), filter_desc.dt, dw_act_param_, &arch_info_);
      if (b_ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise convolution infer forward algorithm failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE_POINTWISE: {
      auto b_ret = depthwise_pointwise_convolution_infer_forward_algorithm(
        input_tensor, filter_tensor, weight_tensors_[1], output_tensor, conv_param_spec_, policy, &(dw_alg_),
        filter_desc.dt, dw_act_param_, pw_act_param_, &arch_info_);
      if (b_ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise pointwise convolution infer forward algorithm failed";
        return RET_ERROR;
      }
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported convolution type for bolt convolution kernel.";
      return RET_NOT_SUPPORT;
  }
  input_tensor.resize(ori_input_desc);
  output_tensor.resize(ori_output_desc);
  return RET_OK;
}

int ConvolutionBoltCPUKernel::InferFilterTransformBytes(int *bytes, int *bytes_extra) {
  auto filter_tensor = weight_tensors_[0];
  switch (conv_param_spec_.convolution_type) {
    case CONVOLUTION_DILATION:
    case CONVOLUTION_POINTWISE: {
      auto ret = convolution_transform_filter_bytes(filter_tensor, conv_param_spec_, pw_alg_, bytes, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt convolution calculate transform filter bytes failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE: {
      auto ret =
        depthwise_convolution_transform_filter_bytes(filter_tensor, conv_param_spec_, dw_alg_, bytes, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise convolution calculate transform filter bytes failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE_POINTWISE: {
      auto ret = depthwise_pointwise_convolution_transform_filter_bytes(
        filter_tensor, weight_tensors_[1], conv_param_spec_, dw_alg_, bytes, bytes_extra, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise pointwise convolution calculate transform filter bytes failed";
        return RET_ERROR;
      }
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported convolution type for bolt convolution kernel.";
      return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ConvolutionBoltCPUKernel::TransformFilter() {
  auto filter_tensor = weight_tensors_[0];
  weight_tmp_tensor_.reset();
  weight_tmp_tensor_ = std::make_shared<BoltTensor>();

  TensorDesc wtm_desc;
  // int8 winograd

  int wtm_bytes = 0;
  int bytes_extra = 0;
  auto ret = InferFilterTransformBytes(&wtm_bytes, &bytes_extra);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Bolt convolution infer transform filter bytes failed";
    return RET_ERROR;
  }
  tmp_weight_ = lite::PackWeightManager::GetInstance()->GetPackData(nullptr, wtm_bytes, &weight_is_packed_);
  if (tmp_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc tmp_weight_ data failed.";
    return RET_NULL_PTR;
  }
  weight_tmp_tensor_->resize(tensor1d(DT_U8, wtm_bytes));
  std::shared_ptr<U8> weight_data(reinterpret_cast<U8 *>(tmp_weight_), [](U8 *ptr) {});
  reinterpret_cast<CpuMemory *>(weight_tmp_tensor_->get_memory())->set_shared_ptr(weight_data);

  switch (conv_param_spec_.convolution_type) {
    case CONVOLUTION_DILATION:
    case CONVOLUTION_POINTWISE: {
      ret = convolution_transform_filter(filter_tensor, conv_param_spec_, pw_alg_, tmp_tensor_,
                                         weight_tmp_tensor_.get(), &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt convolution transform filter failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE: {
      ret = depthwise_convolution_transform_filter(filter_tensor, conv_param_spec_, dw_alg_, weight_tmp_tensor_.get(),
                                                   &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise convolution transform filter failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE_POINTWISE: {
      BoltTensor pw_tensor;
      pw_tensor.resize(tensor1d(DT_U8, bytes_extra));
      pw_weight_ = lite::PackWeightManager::GetInstance()->GetPackData(nullptr, bytes_extra, &weight_is_packed_);
      if (pw_weight_ == nullptr) {
        MS_LOG(ERROR) << "Malloc pw_weight_ data failed.";
        return RET_NULL_PTR;
      }
      std::shared_ptr<U8> pw_weight_data(reinterpret_cast<U8 *>(pw_weight_), [](U8 *ptr) {});
      reinterpret_cast<CpuMemory *>(pw_tensor.get_memory())->set_shared_ptr(pw_weight_data);
      ret =
        depthwise_pointwise_convolution_transform_filter(filter_tensor, weight_tensors_[1], conv_param_spec_, dw_alg_,
                                                         weight_tmp_tensor_.get(), &pw_tensor, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise pointwise convolution transform filter failed";
        return RET_ERROR;
      }
      weight_tensors_[1] = pw_tensor;
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported convolution type for bolt convolution kernel.";
      return RET_NOT_SUPPORT;
  }

  weight_tensors_[0] = *(weight_tmp_tensor_.get());
  return RET_OK;
}

int ConvolutionBoltCPUKernel::MallocTmpTensor() {
  Tensor input_tensor = this->bolt_in_tensors_[0];
  TensorDesc ori_input_desc = input_tensor.get_desc();
  TensorDesc input_desc = transformDescTo4d(ori_input_desc);
  input_tensor.resize(input_desc);
  Tensor filter_tensor = this->weight_tensors_[0];
  TensorDesc filter_desc = filter_tensor.get_desc();
  if (DT_F16_8Q == filter_desc.dt || DT_F32_8Q == filter_desc.dt) {
    filter_desc.dt = DT_I8;
    filter_tensor.resize(filter_desc);
  }
  Tensor output_tensor = this->bolt_out_tensors_[0];
  TensorDesc ori_output_desc = output_tensor.get_desc();
  TensorDesc output_desc = transformDescTo4d(ori_output_desc);
  output_tensor.resize(output_desc);

  U32 bytes = 0;
  switch (conv_param_spec_.convolution_type) {
    case CONVOLUTION_DILATION:
    case CONVOLUTION_POINTWISE: {
      auto ret = convolution_infer_forward_tmp_bytes(input_tensor, filter_tensor, output_tensor, conv_param_spec_,
                                                     pw_alg_, &bytes, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt convolution infer forward temp bytes failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE: {
      auto ret = depthwise_convolution_infer_forward_tmp_bytes(input_tensor, filter_tensor, output_tensor,
                                                               conv_param_spec_, dw_alg_, &bytes, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise convolution infer forward temp bytes failed";
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE_POINTWISE: {
      auto ret = depthwise_pointwise_convolution_infer_forward_tmp_bytes(
        input_tensor, filter_tensor, this->weight_tensors_[1], output_tensor, conv_param_spec_, dw_alg_, &bytes,
        &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Bolt depthwise pointwise convolution infer forward temp bytes failed";
        return RET_ERROR;
      }
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported convolution type for bolt convolution kernel.";
      return RET_NOT_SUPPORT;
  }
  input_tensor.resize(ori_input_desc);
  output_tensor.resize(ori_output_desc);
  tmp_tensor_.resize(tensor1d(DT_U8, bytes));

  run_buffer_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(bytes));
  if (run_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc run temp buffer data failed.";
    return RET_NULL_PTR;
  }
  std::shared_ptr<U8> run_data(reinterpret_cast<U8 *>(run_buffer_), [](U8 *ptr) {});
  reinterpret_cast<CpuMemory *>(tmp_tensor_.get_memory())->set_shared_ptr(run_data);
  return RET_OK;
}

int ConvolutionBoltCPUKernel::Prepare() {
  auto ret = BoltKernel::Prepare();
  if (ret != RET_OK) {
    return RET_ERROR;
  }
  ret = InitWeightBiasTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR)
      << "Preprocess lite weight tensor to Bolt weight tensor, including: nhwc->nchw, tile channel to 8 failed";
    return RET_ERROR;
  }
  ret = InferForwardAlgorithm();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Bolt convolution infer forward algorithm failed";
    return RET_ERROR;
  }
  ret = TransformFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Bolt convolution transform filter failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBoltCPUKernel::Run() {
  // compute
  auto ret = BoltKernel::Run();
  if (ret != RET_OK) {
    return RET_ERROR;
  }
  auto input_tensor = bolt_in_tensors_[0];
  std::vector<BoltTensor> input_tensors(1, input_tensor);
  auto filter_tensor = weight_tensors_[0];
  auto bias_tensor = bias_tensors_[0];
  auto output_tensor = bolt_out_tensors_[0];

  TensorDesc input_desc = input_tensor.get_desc();
  TensorDesc output_desc = output_tensor.get_desc();
  if (input_desc.df != DF_NCHW && input_desc.df != DF_NCHWC8) {
    MS_LOG(ERROR) << "Bolt convolution only supports NCHW or NC8HW8 input right now.";
    return RET_ERROR;
  }
  if (output_desc.df != DF_NCHWC8) {
    MS_LOG(ERROR) << "Bolt convolution only supports NC8HW8 output right now.";
    return RET_ERROR;
  }

  F32 *scale_ptr = nullptr;
  ret = MallocTmpTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Bolt convolution malloc tmp memory tensor failed.";
    return RET_ERROR;
  }

  switch (conv_param_spec_.convolution_type) {
    case CONVOLUTION_DILATION:
    case CONVOLUTION_POINTWISE: {
      std::vector<Tensor> tmp_tensors(1, tmp_tensor_);
      ret = convolution(input_tensors, filter_tensor, conv_param_spec_, pw_alg_, scale_ptr, bias_tensor, tmp_tensors,
                        output_tensor, pw_act_param_, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Run bolt convolution calculate function failed";
        ms_context_->allocator->Free(run_buffer_);
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE: {
      ret = depthwise_convolution(bolt_in_tensors_[0], filter_tensor, conv_param_spec_, dw_alg_, scale_ptr, bias_tensor,
                                  tmp_tensor_, output_tensor, dw_act_param_, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Run bolt depthwise convolution calculate function failed";
        ms_context_->allocator->Free(run_buffer_);
        return RET_ERROR;
      }
      break;
    }
    case CONVOLUTION_DEPTHWISE_POINTWISE: {
      std::vector<Tensor> tmp_tensors(1, tmp_tensor_);
      ret = depthwise_pointwise_convolution(input_tensors, filter_tensor, weight_tensors_[1], conv_param_spec_, dw_alg_,
                                            scale_ptr, bias_tensor, bias_tensors_[1], tmp_tensors, output_tensor,
                                            dw_act_param_, pw_act_param_, &arch_info_);
      if (ret != SUCCESS) {
        MS_LOG(ERROR) << "Run bolt depthwise pointwise convolution calculate function failed";
        ms_context_->allocator->Free(run_buffer_);
        return RET_ERROR;
      }
      break;
    }
    default: {
      MS_LOG(ERROR) << "Unsupported convolution type: " << conv_param_spec_.convolution_type;
      ms_context_->allocator->Free(run_buffer_);
      return RET_NOT_SUPPORT;
    }
  }
  ms_context_->allocator->Free(run_buffer_);
  return RET_OK;
}

BLOT_REG_KERNEL(PrimitiveType_Conv2DFusion, kNumberTypeFloat32, NC8HW8, BoltOpt<ConvolutionBoltCPUKernel>)
BLOT_REG_KERNEL(PrimitiveType_Conv2DFusion, kNumberTypeFloat16, NC8HW8, BoltOpt<ConvolutionBoltCPUKernel>)
}  // namespace mindspore::kernel::bolt
