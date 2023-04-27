/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/base/group_convolution_creator.h"

namespace mindspore::kernel {
void CopyTensorQuantParam(lite::Tensor *dst, const lite::Tensor *src) {
  for (size_t i = 0; i < src->quant_params().size(); i++) {
    dst->AddQuantParam(src->quant_params().at(i));
  }
}

ConvParameter *CreateNewConvParameter(const ConvParameter *parameter) {
  auto conv_parameter = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_parameter == nullptr) {
    MS_LOG(ERROR) << "Malloc new conv parameter failed.";
    return nullptr;
  }
  memcpy(conv_parameter, parameter, sizeof(ConvParameter));
  return conv_parameter;
}

void FreeCurrentConv(ConvParameter *conv_param, std::vector<lite::Tensor *> *new_inputs,
                     std::vector<lite::Tensor *> *new_outputs) {
  if (conv_param != nullptr) {
    free(conv_param);
    conv_param = nullptr;
  }
  if (new_inputs != nullptr) {
    for (auto &in_tensor : *new_inputs) {
      delete in_tensor;
      in_tensor = nullptr;
    }
  }
  if (new_outputs != nullptr) {
    for (auto &out_tensor : *new_outputs) {
      delete out_tensor;
      out_tensor = nullptr;
    }
  }
}

lite::Tensor *CreateConstTensor(const lite::Tensor *tensor, const std::vector<int> &shape, const int index) {
  auto new_tensor =
    new (std::nothrow) lite::Tensor(tensor->data_type(), shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  if (new_tensor == nullptr) {
    MS_LOG(ERROR) << "Create new_tensor failed.";
    return nullptr;
  }
  auto ret = new_tensor->MallocData();
  if (ret != lite::RET_OK) {
    delete new_tensor;
    MS_LOG(ERROR) << "Malloc new_tensor failed.";
    return nullptr;
  }

  if (new_tensor->Size() == 0) {
    delete new_tensor;
    MS_LOG(ERROR) << "Tensor data size should not be 0.";
    return nullptr;
  }
  auto size = new_tensor->Size();
  if (SIZE_MUL_OVERFLOW(static_cast<size_t>(index), size)) {
    delete new_tensor;
    MS_LOG(ERROR) << "Mul overflow.";
    return nullptr;
  }
  MS_CHECK_TRUE_MSG(tensor->data() != nullptr, nullptr, "tensor data is nullptr.");
  void *new_tensor_data =
    reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(tensor->data()) + index * new_tensor->Size());
  if (new_tensor_data == nullptr) {
    delete new_tensor;
    return nullptr;
  }
  (void)memcpy(new_tensor->data(), reinterpret_cast<void *>(new_tensor_data), new_tensor->Size());
  return new_tensor;
}

lite::Tensor *CreateVarTensor(const TensorInfo &tensor_info, bool inferred) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new tensor failed.";
    return nullptr;
  }
  tensor->set_data_type(tensor_info.data_type_);
  tensor->set_format(tensor_info.format_);
  tensor->set_category(tensor_info.tensor_type_);
  tensor->set_shape(tensor_info.shape_);
  tensor->set_allocator(tensor_info.allocator_);
  return tensor;
}

/* Class GroupConv Creator Implement Part */
void GroupConvCreator::CopyQuantParam(const std::vector<lite::Tensor *> *tensors) {
  for (size_t j = 0; j < origin_inputs_.size(); ++j) {
    CopyTensorQuantParam(tensors->at(j), origin_inputs_.at(j));
  }
}

void GroupConvCreator::FreeGroupConvs() {
  for (auto &sub_conv : group_convs_) {
    for (auto in_tensor : sub_conv->in_tensors()) {
      delete in_tensor;
      in_tensor = nullptr;
    }
    for (auto out_tensor : sub_conv->out_tensors()) {
      delete out_tensor;
      out_tensor = nullptr;
    }
    delete sub_conv;
    sub_conv = nullptr;
  }
  group_convs_.clear();
}

int GroupConvCreator::NewInputTensor(std::vector<lite::Tensor *> *tensors) {
  auto allocator = ms_context_ != nullptr ? ms_context_->allocator : nullptr;
  auto in_tensor =
    CreateVarTensor({input_shape_, allocator, mindspore::NHWC, data_type_, lite::Category::VAR, true}, infered_);
  if (in_tensor == nullptr) {
    return lite::RET_ERROR;
  }
  tensors->emplace_back(in_tensor);
  return lite::RET_OK;
}

int GroupConvCreator::NewOutputTensor(std::vector<lite::Tensor *> *tensors, const lite::Tensor *output) const {
  auto allocator = ms_context_ != nullptr ? ms_context_->allocator : nullptr;
  auto out_tensor =
    CreateVarTensor({output_shape_, allocator, output->format(), data_type_, output->category(), false}, infered_);
  if (out_tensor == nullptr) {
    return lite::RET_ERROR;
  }
  if (is_quant_) {
    CopyTensorQuantParam(out_tensor, output);
  }
  tensors->emplace_back(out_tensor);
  return lite::RET_OK;
}

int GroupConvCreator::NewConstTensor(std::vector<lite::Tensor *> *tensors, int group_id) {
  std::vector<std::pair<int, std::vector<int>>> const_tensor_list{std::make_pair(kWeightIndex, filter_shape_)};
  if (origin_inputs_.size() == kInputSize2) {
    const_tensor_list.emplace_back(std::make_pair(kBiasIndex, bias_shape_));
  }
  for (auto &info : const_tensor_list) {
    auto const_tensor = CreateConstTensor(origin_inputs_.at(info.first), info.second, group_id);
    if (const_tensor == nullptr) {
      MS_LOG(ERROR) << "const tensor is nullptr.";
      return lite::RET_ERROR;
    }
    tensors->emplace_back(const_tensor);
  }
  return lite::RET_OK;
}

int GroupConvCreator::SetShapeOfTensors() {
  auto weight_tensor = origin_inputs_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  int new_in_channel = weight_tensor->Channel();
  int new_out_channel = 0;
  if (conv_param_ == nullptr) {
    return lite::RET_ERROR;
  }
  if (conv_param_->group_ == 0) {
    MS_LOG(ERROR) << "Divisor 'group' cannot be 0.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(weight_tensor->shape().size() == DIMENSION_4D, lite::RET_ERROR,
                    "weight tensor shape size should be 4.");
  if (conv_param_->kernel_h_ != weight_tensor->shape()[SECOND_INPUT] ||
      conv_param_->kernel_w_ != weight_tensor->shape()[THIRD_INPUT]) {
    MS_LOG(ERROR) << "kernel_h, kernel_w should be equal to " << weight_tensor->shape()[SECOND_INPUT] << ", "
                  << weight_tensor->shape()[SECOND_INPUT] << " but got " << conv_param_->kernel_h_ << ", "
                  << conv_param_->kernel_w_;
    return lite::RET_ERROR;
  }
  new_out_channel = origin_inputs_.at(kWeightIndex)->Batch() / conv_param_->group_;
  /* set shape */
  set_filter_shape({new_out_channel, conv_param_->kernel_h_, conv_param_->kernel_w_, new_in_channel});
  set_bias_shape({new_out_channel});
  conv_param_->input_channel_ = new_in_channel;
  conv_param_->output_channel_ = new_out_channel;
  if (infered_) {
    set_input_shape({origin_inputs_.front()->Batch(), origin_inputs_.front()->Height(), origin_inputs_.front()->Width(),
                     new_in_channel});
    set_output_shape({origin_inputs_.front()->Batch(), origin_outputs_.front()->Height(),
                      origin_outputs_.front()->Width(), new_out_channel});
  } else {
    set_input_shape({-1});
    set_output_shape({-1});
  }
  return lite::RET_OK;
}

int GroupConvCreator::GetSingleConvParam(ConvParameter *conv_param, std::vector<lite::Tensor *> *new_inputs,
                                         std::vector<lite::Tensor *> *new_outputs, int group_id) {
  if (conv_param == nullptr) {
    FreeGroupConvs();
    return lite::RET_ERROR;
  }
  // create new input for each group
  if (NewInputTensor(new_inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "new input tensor failed.";
    FreeGroupConvs();
    FreeCurrentConv(conv_param, new_inputs, {});
    return lite::RET_ERROR;
  }
  // const tensor
  if (NewConstTensor(new_inputs, group_id) != lite::RET_OK) {
    MS_LOG(ERROR) << "new const tensor failed.";
    FreeGroupConvs();
    FreeCurrentConv(conv_param, new_inputs, {});
    return lite::RET_ERROR;
  }
  // create new output tensor
  for (auto &output : origin_outputs_) {
    if (NewOutputTensor(new_outputs, output) != lite::RET_OK) {
      MS_LOG(ERROR) << "new output tensor failed.";
      FreeGroupConvs();
      FreeCurrentConv(conv_param, new_inputs, new_outputs);
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}
}  // namespace mindspore::kernel
