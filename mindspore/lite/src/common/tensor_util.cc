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

}  // namespace lite
}  // namespace mindspore
