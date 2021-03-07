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

#include "src/common/tensor_util.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
int InputTensor2TensorC(const std::vector<lite::Tensor *> &tensors_in, std::vector<TensorC *> *tensors_out) {
  for (size_t i = 0; i < tensors_in.size(); ++i) {
    size_t shape_size = tensors_in[i]->shape().size();
    if (shape_size >= MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "shape size " << shape_size << " unsupported!";
      return RET_ERROR;
    }
    auto *tensor_c = static_cast<TensorC *>(malloc(sizeof(TensorC)));
    if (tensor_c == nullptr) {
      MS_LOG(ERROR) << "malloc tensor fail!";
      return RET_ERROR;
    }
    tensor_c->format_ = tensors_in[i]->format();
    tensor_c->data_type_ = tensors_in[i]->data_type();
    tensor_c->shape_size_ = shape_size;
    tensor_c->data_ = tensors_in[i]->data_c();
    for (size_t j = 0; j < shape_size; ++j) {
      tensor_c->shape_[j] = tensors_in[i]->shape()[j];
    }
    tensors_out->push_back(tensor_c);
  }
  return RET_OK;
}

int OutputTensor2TensorC(const std::vector<lite::Tensor *> &tensors, std::vector<TensorC *> *tensors_c) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto *tensor_c = static_cast<TensorC *>(malloc(sizeof(TensorC)));
    if (tensor_c == nullptr) {
      MS_LOG(ERROR) << "malloc tensor fail!";
      return RET_ERROR;
    }
    tensor_c->data_type_ = kNumberTypeFloat32;
    tensor_c->format_ = schema::Format::Format_NCHW;
    tensor_c->data_ = nullptr;
    tensor_c->shape_size_ = 0;
    tensors_c->push_back(tensor_c);
  }
  return RET_OK;
}

void TensorC2LiteTensor(const std::vector<TensorC *> &tensors_in, std::vector<lite::Tensor *> *tensors_out) {
  for (size_t i = 0; i < tensors_in.size(); ++i) {
    tensors_out->at(i)->set_format(static_cast<schema::Format>(tensors_in[i]->format_));
    tensors_out->at(i)->set_data_type(static_cast<TypeId>(tensors_in[i]->data_type_));
    tensors_out->at(i)->set_shape({tensors_in[i]->shape_, tensors_in[i]->shape_ + tensors_in[i]->shape_size_});
  }
}

void FreeAllTensorC(std::vector<TensorC *> *tensors_in) {
  for (auto &i : *tensors_in) {
    if (i == nullptr) {
      continue;
    }
    if (i->data_type_ == kObjectTypeTensorType) {
      TensorListC *tensorListC = reinterpret_cast<TensorListC *>(i);
      FreeTensorListC(tensorListC);
      tensorListC = nullptr;
    } else {
      free(i);
      i = nullptr;
    }
  }
  tensors_in->clear();
}

void FreeTensorListC(TensorListC *tensorlist_c) {
  for (size_t i = 0; i < tensorlist_c->element_num_; i++) {
    free(tensorlist_c->tensors_[i]);
    tensorlist_c->tensors_[i] = nullptr;
  }
  if (tensorlist_c->tensors_ != nullptr) {
    free(tensorlist_c->tensors_);
    tensorlist_c->tensors_ = nullptr;
  }
  free(tensorlist_c);
}

TensorC *NewTensorC() {
  auto *tensor_c = static_cast<TensorC *>(malloc(sizeof(TensorC)));
  if (tensor_c == nullptr) {
    MS_LOG(ERROR) << "malloc tensor fail!";
    return nullptr;
  }
  tensor_c->data_type_ = kNumberTypeFloat32;
  tensor_c->format_ = schema::Format::Format_NCHW;
  tensor_c->data_ = nullptr;
  tensor_c->shape_size_ = 0;
  return tensor_c;
}

void Tensor2TensorC(Tensor *src, TensorC *dst) {
  dst->format_ = src->format();
  dst->data_ = src->data_c();
  dst->data_type_ = src->data_type();
  dst->shape_size_ = src->shape().size();
  for (size_t i = 0; i < dst->shape_size_; i++) {
    dst->shape_[i] = src->shape().at(i);
  }
}

void TensorC2Tensor(TensorC *src, Tensor *dst) {
  dst->set_format(static_cast<schema::Format>(src->format_));
  dst->set_data_type(static_cast<TypeId>(src->data_type_));  // get data during the runtime period
  dst->set_shape(std::vector<int>(src->shape_, src->shape_ + src->shape_size_));
}

int TensorList2TensorListC(TensorList *src, TensorListC *dst) {
  dst->data_type_ = static_cast<TypeIdC>(src->data_type());
  dst->format_ = src->format();
  dst->element_num_ = src->shape().empty() ? 0 : src->tensors().size();

  dst->tensors_ = reinterpret_cast<TensorC **>(malloc(dst->element_num_ * sizeof(TensorC *)));
  if (dst->tensors_ == nullptr) {
    return RET_ERROR;
  }
  memset(dst->tensors_, 0, dst->element_num_ * sizeof(TensorC *));
  for (size_t i = 0; i < dst->element_num_; i++) {
    dst->tensors_[i] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
    if (dst->tensors_[i] == nullptr) {
      return NNACL_ERR;
    }
    memset(dst->tensors_[i], 0, sizeof(TensorC));
    Tensor2TensorC(src->tensors().at(i), dst->tensors_[i]);
  }

  dst->tensors_data_type_ = src->tensors_data_type();
  dst->element_shape_size_ = src->element_shape().size();
  for (size_t i = 0; i < dst->element_shape_size_; i++) {
    dst->element_shape_[i] = src->element_shape().at(i);
  }
  dst->max_elements_num_ = src->max_elements_num();
  return NNACL_OK;
}

void TensorListC2TensorList(TensorListC *src, TensorList *dst) {
  dst->set_data_type(static_cast<TypeId>(src->data_type_));
  dst->set_format(static_cast<schema::Format>(src->format_));
  dst->set_shape(std::vector<int>(1, src->element_num_));
  dst->set_tensors_data_type(static_cast<TypeId>(src->tensors_data_type_));

  // Set Tensors
  for (size_t i = 0; i < src->element_num_; i++) {
    TensorC2Tensor(src->tensors_[i], dst->GetTensor(i));
  }

  dst->set_element_shape(std::vector<int>(src->element_shape_, src->element_shape_ + src->element_shape_size_));
  dst->set_max_elements_num(src->max_elements_num_);
}

int GenerateMergeOutTensorC(const std::vector<lite::Tensor *> &inputs, std::vector<lite::Tensor *> *outputs,
                            std::vector<TensorC *> *out_tensor_c) {
  int ret = RET_OK;
  for (size_t i = 0; i < outputs->size(); i++) {
    if (inputs.at(i)->data_type() == kObjectTypeTensorType) {
      auto *output_tensorlist = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
      if (output_tensorlist == nullptr) {
        return RET_ERROR;
      }
      memset(output_tensorlist, 0, sizeof(TensorListC));
      output_tensorlist->element_num_ = inputs[i]->shape().empty() ? 0 : inputs[i]->shape().at(0);
      if (output_tensorlist->element_num_ != 0) {
        output_tensorlist->tensors_ =
          reinterpret_cast<TensorC **>(malloc(output_tensorlist->element_num_ * sizeof(TensorC *)));
        if (output_tensorlist->tensors_ == nullptr) {
          free(output_tensorlist);
          output_tensorlist = nullptr;
          return RET_ERROR;
        }
        memset(output_tensorlist->tensors_, 0, output_tensorlist->element_num_ * sizeof(TensorC *));
        for (size_t j = 0; j < output_tensorlist->element_num_; j++) {
          output_tensorlist->tensors_[j] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
          if (output_tensorlist->tensors_[j] == nullptr) {
            for (size_t k = 0; k < j; k++) {
              free(output_tensorlist->tensors_[k]);
              output_tensorlist->tensors_[k] = nullptr;
            }
            free(output_tensorlist->tensors_);
            output_tensorlist->tensors_ = nullptr;
            free(output_tensorlist);
            output_tensorlist = nullptr;
            return RET_ERROR;
          }
          memset(output_tensorlist->tensors_[j], 0, sizeof(TensorC));
        }
      }

      out_tensor_c->push_back(reinterpret_cast<TensorC *const>(output_tensorlist));
    } else {
      auto *output_tensor = NewTensorC();
      if (output_tensor == nullptr) {
        MS_LOG(ERROR) << "malloc tensor_c failed";
        ret = RET_ERROR;
        break;
      }
      out_tensor_c->push_back(reinterpret_cast<TensorC *const>(output_tensor));
    }
  }
  return ret;
}

int GenerateSwitchOutTensorC(const std::vector<lite::Tensor *> &inputs, std::vector<lite::Tensor *> *outputs,
                             std::vector<TensorC *> *out_tensor_c) {
  int ret = RET_OK;
  MS_ASSERT(inputs.size() == outputs->size() / 2 + 1);
  out_tensor_c->resize(outputs->size());
  for (size_t i = 0; i < outputs->size() / 2; i++) {
    if (inputs.at(i + 1)->data_type() == kObjectTypeTensorType) {
      auto *output_tensorlist1 = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
      if (output_tensorlist1 == nullptr) {
        MS_LOG(ERROR) << "malloc tensorlist_c failed";
        ret = RET_ERROR;
        break;
      }

      memset(output_tensorlist1, 0, sizeof(TensorListC));
      output_tensorlist1->element_num_ = inputs[i + 1]->shape().empty() ? 0 : inputs[i + 1]->shape().at(0);
      if (output_tensorlist1->element_num_ != 0) {
        output_tensorlist1->tensors_ =
          reinterpret_cast<TensorC **>(malloc(output_tensorlist1->element_num_ * sizeof(TensorC *)));
        if (output_tensorlist1->tensors_ == nullptr) {
          free(output_tensorlist1);
          output_tensorlist1 = nullptr;
          return RET_ERROR;
        }
        memset(output_tensorlist1->tensors_, 0, output_tensorlist1->element_num_ * sizeof(TensorC *));
        for (size_t j = 0; j < output_tensorlist1->element_num_; j++) {
          output_tensorlist1->tensors_[j] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
          if (output_tensorlist1->tensors_[j] == nullptr) {
            for (size_t k = 0; k < j; k++) {
              free(output_tensorlist1->tensors_[k]);
              output_tensorlist1->tensors_[k] = nullptr;
            }
            free(output_tensorlist1->tensors_);
            output_tensorlist1->tensors_ = nullptr;
            return RET_ERROR;
          }
          memset(output_tensorlist1->tensors_[j], 0, sizeof(TensorC));
        }
      }

      out_tensor_c->at(i) = reinterpret_cast<TensorC *const>(output_tensorlist1);

      auto *output_tensorlist2 = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
      if (output_tensorlist2 == nullptr) {
        return RET_ERROR;
      }
      memset(output_tensorlist2, 0, sizeof(TensorListC));
      output_tensorlist2->element_num_ = inputs[i + 1]->shape().empty() ? 0 : inputs[i + 1]->shape().at(0);
      if (output_tensorlist2->element_num_ != 0) {
        output_tensorlist2->tensors_ =
          reinterpret_cast<TensorC **>(malloc(output_tensorlist2->element_num_ * sizeof(TensorC *)));
        if (output_tensorlist2->tensors_ == nullptr) {
          free(output_tensorlist2);
          output_tensorlist2 = nullptr;
          return RET_ERROR;
        }
        memset(output_tensorlist2->tensors_, 0, output_tensorlist2->element_num_ * sizeof(TensorC *));
        for (size_t j = 0; j < output_tensorlist2->element_num_; j++) {
          output_tensorlist2->tensors_[j] = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
          if (output_tensorlist2->tensors_[j] == nullptr) {
            for (size_t k = 0; k < j; k++) {
              free(output_tensorlist2->tensors_[k]);
              output_tensorlist2->tensors_[k] = nullptr;
            }
            free(output_tensorlist2->tensors_);
            output_tensorlist2->tensors_ = nullptr;
            free(output_tensorlist2);
            output_tensorlist2 = nullptr;
            return RET_ERROR;
          }
          memset(output_tensorlist2->tensors_[j], 0, sizeof(TensorC));
        }
      }

      out_tensor_c->at(i + outputs->size() / 2) = reinterpret_cast<TensorC *const>(output_tensorlist2);
    } else {
      auto *output_tensor1 = NewTensorC();
      if (output_tensor1 == nullptr) {
        MS_LOG(ERROR) << "malloc tensor_c failed";
        ret = RET_ERROR;
        break;
      }
      out_tensor_c->at(i) = reinterpret_cast<TensorC *const>(output_tensor1);
      auto *output_tensor2 = NewTensorC();
      if (output_tensor2 == nullptr) {
        MS_LOG(ERROR) << "malloc tensor_c failed";
        ret = RET_ERROR;
        break;
      }
      out_tensor_c->at(i + outputs->size() / 2) = reinterpret_cast<TensorC *const>(output_tensor2);
    }
  }
  return ret;
}

int GenerateOutTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &inputs,
                       std::vector<lite::Tensor *> *outputs, std::vector<TensorC *> *out_tensor_c) {
  int ret = RET_OK;
  if (parameter->type_ == mindspore::schema::PrimitiveType_TensorListFromTensor ||
      parameter->type_ == mindspore::schema::PrimitiveType_TensorListReserve ||
      parameter->type_ == mindspore::schema::PrimitiveType_TensorListSetItem) {
    // TensorListC ->TensorC
    auto *tensor_list_c = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
    if (tensor_list_c == nullptr) {
      return RET_ERROR;
    }
    memset(tensor_list_c, 0, sizeof(TensorListC));
    out_tensor_c->push_back(reinterpret_cast<TensorC *const>(tensor_list_c));
  } else if (parameter->type_ == mindspore::schema::PrimitiveType_Merge) {
    ret = GenerateMergeOutTensorC(inputs, outputs, out_tensor_c);
  } else if (parameter->type_ == mindspore::schema::PrimitiveType_Switch) {
    ret = GenerateSwitchOutTensorC(inputs, outputs, out_tensor_c);
  } else {
    ret = OutputTensor2TensorC(*outputs, out_tensor_c);
  }
  return ret;
}

int GenerateInTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &inputs,
                      std::vector<lite::Tensor *> *outputs, std::vector<TensorC *> *in_tensor_c) {
  int ret = RET_OK;
  for (auto input : inputs) {
    if (input->data_type() == kObjectTypeTensorType) {
      // Tensor ->TensorList -> TensorListC -> TensorC
      auto *tensor_list = reinterpret_cast<TensorList *>(input);
      auto *tensor_list_c = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
      if (tensor_list_c == nullptr) {
        ret = RET_NULL_PTR;
        break;
      }
      memset(tensor_list_c, 0, sizeof(TensorListC));
      ret = TensorList2TensorListC(tensor_list, tensor_list_c);
      if (ret != RET_OK) {
        return NNACL_ERR;
      }
      in_tensor_c->push_back(reinterpret_cast<TensorC *>(tensor_list_c));
    } else {
      // Tensor -> TensorC
      auto *tensor_c = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
      if (tensor_c == nullptr) {
        ret = RET_NULL_PTR;
        break;
      }
      Tensor2TensorC(input, tensor_c);
      in_tensor_c->emplace_back(tensor_c);
    }
  }
  return ret;
}

}  // namespace lite
}  // namespace mindspore
