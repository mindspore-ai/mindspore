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

#include "src/runtime/kernel/arm/base/group_convolution_creator.h"

namespace mindspore::kernel {
void CopyTensorQuantParam(lite::Tensor *dst, lite::Tensor *src) {
  for (size_t i = 0; i < src->quant_params().size(); i++) {
    dst->AddQuantParam(src->quant_params().at(i));
  }
}

ConvParameter *CreateNewConvParameter(ConvParameter *parameter) {
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

static inline lite::Tensor *TensorMalloc(lite::Tensor *tensor) {
  if (tensor->MallocData() != lite::RET_OK) {
    delete tensor;
    MS_LOG(ERROR) << "malloc tensor data failed.";
    return nullptr;
  }
  return tensor;
}

lite::Tensor *CreateConstTensor(lite::Tensor *tensor, const std::vector<int> &shape, const int index) {
  auto new_tensor = new (std::nothrow)
    lite::Tensor(tensor->data_type(), shape, schema::Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
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
  memcpy(new_tensor->data_c(), reinterpret_cast<char *>(tensor->data_c()) + index * new_tensor->Size(),
         new_tensor->Size());
  return new_tensor;
}

lite::Tensor *CreateVarTensor(const TensorInfo &tensor_info, bool inferred) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (!tensor) {
    MS_LOG(ERROR) << "new tensor failed.";
    return nullptr;
  }
  tensor->set_data_type(tensor_info.data_type_);
  tensor->set_format(tensor_info.format_);
  tensor->set_category(tensor_info.tensor_type_);
  if (tensor_info.is_in_) {
    tensor->set_shape(tensor_info.shape_);
  }

  if (inferred) {
    // set shape of out tensor
    if (!tensor_info.is_in_) {
      tensor->set_shape(tensor_info.shape_);
    }
    return TensorMalloc(tensor);
  }
  return tensor;
}

/* Class GroupConv Creator Implement Part*/
void GroupConvCreator::CopyQuantParam(std::vector<lite::Tensor *> *tensors) {
  for (size_t j = 0; j < origin_inputs_.size(); ++j) {
    CopyTensorQuantParam(tensors->at(j), origin_inputs_.at(j));
  }
}

void GroupConvCreator::FreeGroupConvs() {
  for (auto &sub_conv : group_convs_) {
    for (auto &in_tensor : sub_conv->in_tensors()) {
      delete in_tensor;
    }
    for (auto &out_tensor : sub_conv->out_tensors()) {
      delete out_tensor;
    }
    delete sub_conv;
    sub_conv = nullptr;
  }
  group_convs_.clear();
}

int GroupConvCreator::NewInputTensor(std::vector<lite::Tensor *> *tensors) {
  auto in_tensor =
    CreateVarTensor({input_shape_, schema::Format_NHWC, data_type_, lite::Tensor::Category::VAR, true}, infered_);
  if (in_tensor == nullptr) {
    return lite::RET_ERROR;
  }
  tensors->emplace_back(in_tensor);
  return lite::RET_OK;
}

int GroupConvCreator::NewOutputTensor(std::vector<lite::Tensor *> *tensors, lite::Tensor *output) {
  auto out_tensor = CreateVarTensor({output_shape_, output->format(), data_type_, output->category(), false}, infered_);
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
  if (origin_inputs_.size() == 3) {
    const_tensor_list.emplace_back(std::make_pair(kBiasIndex, bias_shape_));
  }
  for (auto &info : const_tensor_list) {
    auto const_tensor = CreateConstTensor(origin_inputs_.at(info.first), info.second, group_id);
    if (const_tensor == nullptr) {
      return lite::RET_ERROR;
    }
    tensors->emplace_back(const_tensor);
  }
  return lite::RET_OK;
}

void GroupConvCreator::SetShapeOfTensors() {
  int new_in_channel = origin_inputs_.at(kWeightIndex)->Channel();
  int new_out_channel;
  if (conv_param_->group_ == 0) {
    MS_LOG(ERROR) << "Divisor 'group' cannot be 0.";
    return;
  } else {
    new_out_channel = origin_inputs_.at(kWeightIndex)->Batch() / conv_param_->group_;
  }

  /* set shape */
  set_filter_shape({new_out_channel, conv_param_->kernel_h_, conv_param_->kernel_w_, new_in_channel});
  set_bias_shape({new_out_channel});
  if (infered_) {
    conv_param_->input_channel_ = new_in_channel;
    conv_param_->output_channel_ = new_out_channel;
    set_input_shape({origin_inputs_.front()->Batch(), origin_inputs_.front()->Height(), origin_inputs_.front()->Width(),
                     new_in_channel});
    set_output_shape({origin_inputs_.front()->Batch(), origin_outputs_.front()->Height(),
                      origin_outputs_.front()->Width(), new_out_channel});
  }
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
