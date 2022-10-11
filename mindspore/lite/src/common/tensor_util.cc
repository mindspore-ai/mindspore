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
#include <string>
#include <algorithm>
#include <unordered_map>
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#ifdef ENABLE_FP16
#include "src/litert/kernel/cpu/fp16/fp16_op_handler.h"
#endif
#include "nnacl/base/cast_base.h"
namespace mindspore {
namespace lite {
int OutputTensor2TensorC(const std::vector<lite::Tensor *> &tensors, std::vector<TensorC *> *tensors_c,
                         std::shared_ptr<Allocator> allocator) {
  MS_ASSERT(tensors_c != nullptr);
  for (size_t i = 0; i < tensors.size(); ++i) {
    TensorC *tensor_c = nullptr;
    if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
      tensor_c = static_cast<TensorC *>(allocator->Malloc(sizeof(TensorC)));
    } else {
      tensor_c = static_cast<TensorC *>(malloc(sizeof(TensorC)));
    }
    if (tensor_c == nullptr) {
      MS_LOG(ERROR) << "malloc tensor fail!";
      return RET_ERROR;
    }
    tensor_c->data_type_ = kNumberTypeFloat32;
    tensor_c->format_ = tensors[i]->format();
    tensor_c->data_ = nullptr;
    tensor_c->shape_size_ = 0;
    tensors_c->push_back(tensor_c);
  }
  return RET_OK;
}

void FreeAllTensorC(std::vector<TensorC *> *tensors_in, std::shared_ptr<Allocator> allocator) {
  if (tensors_in == nullptr) {
    return;
  }
  for (auto &i : *tensors_in) {
    if (i == nullptr) {
      continue;
    }
    if (i->data_type_ == kObjectTypeTensorType) {
      TensorListC *tensorListC = reinterpret_cast<TensorListC *>(i);
      FreeTensorListC(tensorListC, allocator);
      tensorListC = nullptr;
    } else {
      if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
        allocator->Free(i);
      } else {
        free(i);
      }
      i = nullptr;
    }
  }
  tensors_in->clear();
}

int Tensor2TensorC(const Tensor *src, TensorC *dst) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->is_ready_ = src->IsReady();
  dst->format_ = static_cast<int>(src->format());
  dst->data_ = src->data();
  dst->data_type_ = src->data_type();
  dst->shape_size_ = src->shape().size();
  if (dst->shape_size_ > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "tensor shape size " << dst->shape_size_ << " is larger than max shape size " << MAX_SHAPE_SIZE;
    return RET_ERROR;
  }
  for (size_t i = 0; i < dst->shape_size_; i++) {
    dst->shape_[i] = src->shape().at(i);
  }
  return RET_OK;
}

int TensorC2Tensor(const TensorC *src, Tensor *dst, std::shared_ptr<Allocator> allocator) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->set_format(static_cast<mindspore::Format>(src->format_));
  dst->set_data_type(static_cast<TypeId>(src->data_type_));  // get data during the runtime period
  dst->set_shape(std::vector<int>(src->shape_, src->shape_ + src->shape_size_));
  if (src->data_ != nullptr) {
    auto data = dst->MutableData();
    MS_CHECK_TRUE_RET(data != nullptr, RET_ERROR);
    memcpy(data, src->data_, dst->Size());
    dst->set_category(CONST_TENSOR);
    if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
      allocator->Free(src->data_);
    } else {
      free(src->data_);
    }
  }
  return RET_OK;
}

int GenerateOutTensorC(const OpParameter *const parameter, const std::vector<lite::Tensor *> &outputs,
                       std::vector<TensorC *> *out_tensor_c, std::shared_ptr<Allocator> allocator) {
  MS_CHECK_TRUE_RET(out_tensor_c != nullptr && parameter != nullptr, RET_ERROR);
  if (parameter->type_ == mindspore::schema::PrimitiveType_TensorListFromTensor ||
      parameter->type_ == mindspore::schema::PrimitiveType_TensorListReserve ||
      parameter->type_ == mindspore::schema::PrimitiveType_TensorListSetItem) {
    // TensorListC ->TensorC
    MS_CHECK_TRUE_RET(!outputs.empty() && outputs.front()->data_type() == TypeId::kObjectTypeTensorType, RET_ERROR);
    TensorListC *tensor_list_c = nullptr;
    if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
      tensor_list_c = reinterpret_cast<TensorListC *>(allocator->Malloc(sizeof(TensorListC)));
    } else {
      tensor_list_c = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
    }
    if (tensor_list_c == nullptr) {
      return RET_ERROR;
    }
    memset(tensor_list_c, 0, sizeof(TensorListC));
    out_tensor_c->push_back(reinterpret_cast<TensorC *const>(tensor_list_c));
    return RET_OK;
  } else {
    return OutputTensor2TensorC(outputs, out_tensor_c, allocator);
  }
}

int GenerateInTensorC(const std::vector<lite::Tensor *> &inputs, std::vector<TensorC *> *in_tensor_c,
                      const std::shared_ptr<Allocator> &allocator) {
  MS_CHECK_TRUE_RET(in_tensor_c != nullptr, RET_ERROR);
  int ret = RET_OK;
  for (auto input : inputs) {
    if (input->data_type() == kObjectTypeTensorType) {
      // Tensor ->TensorList -> TensorListC -> TensorC
      auto *tensor_list = reinterpret_cast<TensorList *>(input);
      TensorListC *tensor_list_c = nullptr;
      if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
        tensor_list_c = reinterpret_cast<TensorListC *>(allocator->Malloc(sizeof(TensorListC)));
      } else {
        tensor_list_c = reinterpret_cast<TensorListC *>(malloc(sizeof(TensorListC)));
      }
      if (tensor_list_c == nullptr) {
        ret = RET_NULL_PTR;
        break;
      }
      memset(tensor_list_c, 0, sizeof(TensorListC));
      ret = TensorList2TensorListC(tensor_list, tensor_list_c, allocator);
      if (ret != RET_OK) {
        if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
          allocator->Free(tensor_list_c->tensors_);
          allocator->Free(tensor_list_c);
        } else {
          free(tensor_list_c->tensors_);
          free(tensor_list_c);
        }
        return NNACL_ERR;
      }
      in_tensor_c->push_back(reinterpret_cast<TensorC *>(tensor_list_c));
    } else {
      // Tensor -> TensorC
      TensorC *tensor_c = nullptr;
      if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
        tensor_c = reinterpret_cast<TensorC *>(allocator->Malloc(sizeof(TensorC)));
      } else {
        tensor_c = reinterpret_cast<TensorC *>(malloc(sizeof(TensorC)));
      }
      if (tensor_c == nullptr) {
        ret = RET_NULL_PTR;
        break;
      }
      ret = Tensor2TensorC(input, tensor_c);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Tensor to TensorC failed.";
        if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
          allocator->Free(tensor_c);
        } else {
          free(tensor_c);
        }
        return ret;
      }
      in_tensor_c->emplace_back(tensor_c);
    }
  }
  return ret;
}

int CheckTensorsInvalid(const std::vector<Tensor *> &tensors) {
  for (auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (tensor->data_type() != kObjectTypeTensorType && tensor->data() == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor data is nullptr " << tensor->tensor_name();
      return RET_ERROR;
    }
    const auto &shape = tensor->shape();
    bool valid = all_of(shape.begin(), shape.end(), [](int i) { return i >= 0; });
    if (!valid) {
      MS_LOG(ERROR) << "The shape of tensor contains negative dimension,"
                    << "check the model and assign the input shape with method Resize().";
      return RET_ERROR;
    }
    if (tensor->format() != mindspore::NHWC && tensor->format() != mindspore::NCHW) {
      MS_LOG(ERROR) << "model input's format may be changed, which should be NHWC or NCHW";
      return RET_FORMAT_ERR;
    }
    if (tensor->data() == nullptr) {
      MS_LOG(ERROR) << "tensor data should be filled before run op";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

std::string ShapeToString(const std::vector<int> &shape) {
  std::string result = "[";
  int max_size = 40;
  result.reserve(max_size);
  for (size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

int CheckGraphInputShapes(const std::vector<Tensor *> &inputs,
                          const std::unordered_map<Tensor *, std::vector<int>> &input_shape_map) {
  for (const auto input : inputs) {
    MS_CHECK_TRUE_MSG(input != nullptr, RET_ERROR, "graph input tensor is nullptr.");
    if (input_shape_map.find(input) == input_shape_map.end()) {
      MS_LOG(ERROR) << "can't find " << input->tensor_name() << " in input_shape_map";
      return RET_ERROR;
    }
    if (!input_shape_map.at(input).empty() && input_shape_map.at(input) != input->shape()) {
#ifndef ENABLE_LITE_ACL
      MS_LOG(ERROR) << "graph input:" << input->tensor_name()
                    << " shape has been illegally modified, please modify the input shape with method Resize().";
      return RET_ERROR;
#else
      MS_LOG(WARNING) << "Please check graph input " << input->tensor_name()
                      << " shape:" << ShapeToString(input->shape())
                      << " has been modified by DVPP method to shape:" << ShapeToString(input_shape_map.at(input))
                      << "."
                      << "If not, the modification is illegal, please modify the input shape with method Resize().";
#endif
    }
  }
  return RET_OK;
}

std::vector<mindspore::MSTensor> LiteTensorsToMSTensors(const std::vector<lite::Tensor *> &lite_tensors) {
  std::vector<mindspore::MSTensor> tensors;
  std::transform(lite_tensors.begin(), lite_tensors.end(), std::back_inserter(tensors),
                 [](lite::Tensor *tensor) { return mindspore::MSTensor(std::make_shared<LiteTensorImpl>(tensor)); });

  return tensors;
}

int MoveCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
  MS_ASSERT(src_tensor != dst_tensor);
  if (src_tensor->data() == dst_tensor->data()) {
    MS_LOG(DEBUG) << "no need to move data.";
    return RET_OK;
  }
  dst_tensor->FreeData();
  dst_tensor->ResetRefCount();
  dst_tensor->set_allocator(src_tensor->allocator());

  if (src_tensor->data() != nullptr) {
    dst_tensor->set_data(src_tensor->MutableData()); /* using MutableData to sync GPU data */
  }

  dst_tensor->set_own_data(src_tensor->own_data());
  src_tensor->DecRefCount();
  return RET_OK;
}

int MoveTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
  if (src_tensor == dst_tensor) {
    MS_LOG(INFO) << "no need to move.";
    return RET_OK;
  }
  MS_ASSERT(src_tensor->allocator() != nullptr);
  auto ret = RET_OK;
  if (src_tensor->data_type() == kObjectTypeTensorType) {
    ret =
      MoveTensorListTensorData(reinterpret_cast<TensorList *>(dst_tensor), reinterpret_cast<TensorList *>(src_tensor));
  } else {
    ret = MoveCommonTensorData(dst_tensor, src_tensor);
  }
  return ret;
}

void SetCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
  dst_tensor->set_data(src_tensor->data());
  dst_tensor->set_own_data(false);
}

int SetTensorData(Tensor *dst_tensor, Tensor *src_tensor) {
  auto ret = RET_OK;
  if (src_tensor->data_type() == kObjectTypeTensorType) {
    ret =
      SetTensorListTensorData(reinterpret_cast<TensorList *>(dst_tensor), reinterpret_cast<TensorList *>(src_tensor));
  } else {
    SetCommonTensorData(dst_tensor, src_tensor);
  }
  return ret;
}

int CastTensorData(Tensor *dst, Tensor *src, bool support_fp16) {
  int ret = RET_OK;
  if (src->data_type() != kObjectTypeTensorType) {
    ret = CastCommonTensorData(dst, src, support_fp16);
  } else {
    ret =
      CastTensorListTensorData(reinterpret_cast<TensorList *>(dst), reinterpret_cast<TensorList *>(src), support_fp16);
  }
  src->DecRefCount();
  return ret;
}

int CastCommonTensorData(Tensor *dst, Tensor *src, bool support_fp16) {
  auto dst_data = dst->ReallocData(); /* using MutableData to sync GPU data */
  if (dst_data == nullptr) {
    MS_LOG(ERROR) << "Remalloc memory failed.";
    return RET_NULL_PTR;
  }
  dst->ResetRefCount();
  if (dst->shape() != src->shape()) {
    MS_LOG(ERROR) << "dst tensor: " << dst->tensor_name() << " shape: " << dst->shape() << " vs "
                  << "src tensor: " << src->tensor_name() << " shape: " << src->shape();
    return RET_PARAM_INVALID;
  }
  auto src_data = src->MutableData();
  auto src_nums_size = src->ElementsNum();
  auto dst_data_type = static_cast<int>(dst->data_type());
  auto src_data_type = static_cast<int>(src->data_type());
  // Some case dst data type is unknown, we will set to float32. In this case, need case is true, but actually no need
  // cast data
  if (dst_data_type == src_data_type) {
    memcpy(dst_data, src_data, src_nums_size);
    return RET_OK;
  }
  if (dst_data_type == kNumberTypeFloat32 && src_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    Float16ToFloat32_fp16_handler(src_data, dst_data, src_nums_size, support_fp16);
#else
    MS_LOG(ERROR) << "not enable fp16.";
    return RET_NOT_SUPPORT;
#endif
  } else if (dst_data_type == kNumberTypeFloat16 && src_data_type == kNumberTypeFloat32) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    Float32ToFloat16_fp16_handler(src_data, dst_data, src_nums_size, support_fp16);
#else
    MS_LOG(ERROR) << "not enable fp16.";
    return RET_NOT_SUPPORT;
#endif
  } else if (dst_data_type == kNumberTypeFloat32 && src_data_type == kNumberTypeInt32) {
    Int32ToFloat32(static_cast<const int32_t *>(src_data), static_cast<float *>(dst_data), src_nums_size);
  } else if (dst_data_type == kNumberTypeInt32 && src_data_type == kNumberTypeFloat32) {
    Float32ToInt32(static_cast<const float *>(src_data), static_cast<int32_t *>(dst_data), src_nums_size);
  } else {
    MS_LOG(ERROR) << "not support dst_data_type: " << dst_data_type << " src_data_type: " << src_data_type;
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

bool NeedCastData(Tensor *dst_tensor, Tensor *src_tensor) {
  if (IsUnKnownDtype(dst_tensor) || IsUnKnownDtype(src_tensor)) {
    MS_LOG(INFO) << "Type unknown, no need cast.";
    return false;
  }
  return !IsSameDtype(dst_tensor, src_tensor);
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP

int SetTensorShape(Tensor *dst, Tensor *src) {
  if (dst->data_type() != kObjectTypeTensorType && src->data_type() != kObjectTypeTensorType) {
    dst->set_shape(src->shape());
    dst->set_format(src->format());
    return RET_OK;
  } else if (dst->data_type() == kObjectTypeTensorType && src->data_type() == kObjectTypeTensorType) {
    auto input_tensorlist = reinterpret_cast<TensorList *>(dst);
    auto input_data_tensorlist = reinterpret_cast<TensorList *>(src);
    MS_CHECK_FALSE_MSG(input_tensorlist == nullptr, RET_ERROR, "cast to tensorlist failed.");
    MS_CHECK_FALSE_MSG(input_data_tensorlist == nullptr, RET_ERROR, "cast to tensorlist failed.");
    input_tensorlist->set_element_shape(input_data_tensorlist->element_shape());
    // because some model shape is not same as tensors().size(), we need the real shape, which is the tensors().size().
    int real_shape_val = static_cast<int>(input_data_tensorlist->tensors().size());
    std::vector<int> real_shape{real_shape_val};
    input_tensorlist->set_shape(real_shape);
    // hard code for some model
    if (input_data_tensorlist->tensors_data_type() != kTypeUnknown &&
        input_tensorlist->tensors_data_type() == kTypeUnknown) {
      input_tensorlist->set_tensors_data_type(input_data_tensorlist->tensors_data_type());
    }
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "not able to set tensor shape between tensor and tensorlist.";
    return RET_ERROR;
  }
}

int CastTensorListTensorData(TensorList *dst_tensorlist, TensorList *src_tensorlist, bool support_fp16) {
  MS_ASSERT(src_tensorlist != nullptr);
  MS_ASSERT(dst_tensorlist != nullptr);
  dst_tensorlist->set_shape(src_tensorlist->shape());
  std::vector<std::vector<int>> tensors_shapes{};
  tensors_shapes.resize(src_tensorlist->tensors().size());
  for (size_t i = 0; i < tensors_shapes.size(); ++i) {
    tensors_shapes[i] = src_tensorlist->tensors()[i]->shape();
  }
  if (!dst_tensorlist->shape().empty()) {
    if (src_tensorlist->tensors_data_type() == kNumberTypeFloat16) {
      dst_tensorlist->MallocTensorListData(kNumberTypeFloat32, tensors_shapes);
    }
    if (src_tensorlist->tensors_data_type() == kNumberTypeFloat32) {
      dst_tensorlist->MallocTensorListData(kNumberTypeFloat16, tensors_shapes);
    }
  }
  dst_tensorlist->set_allocator(src_tensorlist->allocator());
  dst_tensorlist->ResetRefCount();

  for (size_t i = 0; i < src_tensorlist->tensors().size(); ++i) {
    auto src_tensor = src_tensorlist->tensors()[i];
    auto dst_tensor = dst_tensorlist->tensors()[i];
    auto ret = CastCommonTensorData(dst_tensor, src_tensor, support_fp16);
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "cast tensor data failed.");
  }
  return RET_OK;
}

int MoveTensorListTensorData(TensorList *dst_tensorlist, TensorList *src_tensorlist) {
  MS_ASSERT(src_tensorlist != nullptr);
  MS_ASSERT(dst_tensorlist != nullptr);
  dst_tensorlist->FreeData();
  dst_tensorlist->ResetRefCount();
  dst_tensorlist->set_allocator(src_tensorlist->allocator());

  auto src_tensorlist_tensors_size = src_tensorlist->tensors().size();
  auto dst_tensorlist_tensors_size = dst_tensorlist->tensors().size();
  if (src_tensorlist_tensors_size != dst_tensorlist_tensors_size) {
    MS_LOG(ERROR) << "src tensorlist: " << src_tensorlist->tensor_name()
                  << " tesnors size: " << src_tensorlist_tensors_size
                  << " vs dst tensorlist: " << dst_tensorlist->tensor_name()
                  << " tensors size: " << dst_tensorlist_tensors_size;
    return RET_ERROR;
  }

  // hard code for some model
  dst_tensorlist->set_tensors_data_type(src_tensorlist->tensors_data_type());

  dst_tensorlist->set_own_data(src_tensorlist->own_data());
  for (size_t i = 0; i < src_tensorlist_tensors_size; ++i) {
    auto src_tensor = src_tensorlist->tensors()[i];
    auto dst_tensor = dst_tensorlist->tensors()[i];

    dst_tensor->set_own_data(src_tensor->own_data());
    if (src_tensor->data() != nullptr) {
      dst_tensor->set_data(src_tensor->MutableData()); /* using MutableData to sync GPU data */
    }
    dst_tensor->set_shape(src_tensor->shape());
  }

  if (src_tensorlist->IsConst() || src_tensorlist->IsGraphInput()) {
    dst_tensorlist->set_own_data(false);
  } else {
    src_tensorlist->DecRefCount();
  }
  return RET_OK;
}

int SetTensorListTensorData(TensorList *dst_tensor_list, TensorList *src_tensor_list) {
  auto ret = dst_tensor_list->FreeTensorListData();
  MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "FreeTensorListData failed.");
  dst_tensor_list->set_own_data(false);
  dst_tensor_list->set_tensors(src_tensor_list->tensors());
  dst_tensor_list->set_tensors_data_type(src_tensor_list->tensors_data_type());
  dst_tensor_list->set_element_shape(src_tensor_list->element_shape());
  return RET_OK;
}

void FreeTensorListC(TensorListC *tensorlist_c, std::shared_ptr<Allocator> allocator) {
  MS_ASSERT(tensorlist_c != nullptr);
  if (tensorlist_c->tensors_ != nullptr) {
    if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
      allocator->Free(tensorlist_c->tensors_);
    } else {
      free(tensorlist_c->tensors_);
    }
    tensorlist_c->tensors_ = nullptr;
  }
  if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
    allocator->Free(tensorlist_c);
  } else {
    free(tensorlist_c);
  }
}

int TensorList2TensorListC(TensorList *src, TensorListC *dst, std::shared_ptr<Allocator> allocator) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->is_ready_ = src->IsReady();
  dst->data_type_ = static_cast<TypeIdC>(src->data_type());
  dst->format_ = static_cast<int>(src->format());
  dst->shape_value_ = src->shape().empty() ? 0 : src->shape().front();
  dst->element_num_ = src->shape().empty() ? 0 : src->tensors().size();

  if ((dst->element_num_ != 0 && SIZE_MAX / dst->element_num_ < sizeof(TensorC)) ||
      dst->element_num_ * sizeof(TensorC) > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "data size error.";
    return RET_ERROR;
  }
  if (allocator != nullptr && !IS_RUNTIME_ALLOCATOR(allocator)) {
    dst->tensors_ = reinterpret_cast<TensorC *>(allocator->Malloc(dst->element_num_ * sizeof(TensorC)));
  } else {
    dst->tensors_ = reinterpret_cast<TensorC *>(malloc(dst->element_num_ * sizeof(TensorC)));
  }
  if (dst->tensors_ == nullptr) {
    return RET_ERROR;
  }
  memset(dst->tensors_, 0, dst->element_num_ * sizeof(TensorC));
  for (size_t i = 0; i < dst->element_num_; i++) {
    auto ret = Tensor2TensorC(src->tensors().at(i), &dst->tensors_[i]);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Tensor to TensorC failed.";
      return ret;
    }
  }

  dst->tensors_data_type_ = src->tensors_data_type();
  dst->element_shape_size_ = src->element_shape().size();
  for (size_t i = 0; i < dst->element_shape_size_; i++) {
    dst->element_shape_[i] = src->element_shape().at(i);
  }
  dst->max_elements_num_ = src->max_elements_num();
  return NNACL_OK;
}

int TensorListC2TensorList(const TensorListC *src, TensorList *dst) {
  MS_CHECK_TRUE_RET(src != nullptr && dst != nullptr, RET_ERROR);
  dst->set_data_type(static_cast<TypeId>(src->data_type_));
  dst->set_format(static_cast<mindspore::Format>(src->format_));
  dst->set_shape(std::vector<int>(1, src->element_num_));
  dst->set_tensors_data_type(static_cast<TypeId>(src->tensors_data_type_));

  // Set Tensors
  for (size_t i = 0; i < src->element_num_; i++) {
    auto ret = TensorC2Tensor(&src->tensors_[i], dst->GetTensor(static_cast<int>(i)));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "TensorC2Tensor failed";
      return ret;
    }
  }

  dst->set_element_shape(std::vector<int>(src->element_shape_, src->element_shape_ + src->element_shape_size_));
  dst->set_max_elements_num(src->max_elements_num_);
  return RET_OK;
}

TypeId TensorListDataType(Tensor *tensor) {
  auto tensor_list = reinterpret_cast<TensorList *>(tensor);
  auto tensor_list_dtype = tensor_list->tensors_data_type();
  if (tensor_list_dtype == kNumberTypeFloat32 || tensor_list_dtype == kNumberTypeFloat16 ||
      tensor_list_dtype == kNumberTypeInt8 || tensor_list_dtype == kNumberTypeInt32 ||
      tensor_list_dtype == kNumberTypeBool) {
    return tensor_list_dtype;
  }
  // if not found, return float32 as default.
  return kNumberTypeFloat32;
}

TensorList *MallocTensorListDataAccordingToTensorListC(Tensor *tensor, TensorListC *tensor_list_c) {
  auto *tensor_list = reinterpret_cast<TensorList *>(tensor);
  tensor_list->set_shape({static_cast<int>(tensor_list_c->element_num_)});
  auto tensor_shape = std::vector<std::vector<int>>(
    tensor_list_c->element_num_, std::vector<int>(tensor_list_c->element_shape_,
                                                  tensor_list_c->element_shape_ + tensor_list_c->element_shape_size_));
  tensor_list->MallocTensorListData(static_cast<TypeId>(tensor_list_c->data_type_), tensor_shape);
  return tensor_list;
}

int DecodeTensorLsit(Tensor *tensor, const int *src_data, size_t length) {
  auto tensor_list = reinterpret_cast<TensorList *>(tensor);
  if (tensor_list->Decode(src_data, length) != RET_OK) {
    MS_LOG(ERROR) << "Decode tensorlist data failed";
    return RET_ERROR;
  }
  return RET_OK;
}

Tensor *CreateTensorList(const std::vector<int> &shape, const Category &src_category, const void *src_data) {
  auto dst_tensor = new (std::nothrow) TensorList(shape, std::vector<int>(), src_category);
  // set tensor list datatype
  auto tensor_list = reinterpret_cast<TensorList *>(dst_tensor);
  MS_CHECK_TRUE_RET(tensor_list != nullptr, nullptr);
  if (src_data != nullptr) {
    auto tensor_data_type = TypeId(reinterpret_cast<const int *>(src_data)[0]);
    tensor_list->set_tensors_data_type(tensor_data_type);
  }
  return dst_tensor;
}

int CopyTensorListTensorDataType(TensorList *dst_tensorlist, TensorList *src_tensorlist) {
  // shape may change, because tensors.size() can be change in RunGraph
  if (dst_tensorlist->data_type() != src_tensorlist->data_type() ||
      dst_tensorlist->format() != src_tensorlist->format()) {
    MS_LOG(ERROR) << "input tensorlist and output tensorlist data_type or format is incompatible";
    MS_LOG(ERROR) << "input tensor data_type: " << src_tensorlist->data_type() << " vs "
                  << "output tensor data_type: " << dst_tensorlist->data_type()
                  << "input tensor format: " << src_tensorlist->format() << " vs "
                  << "output tensor format: " << dst_tensorlist->format();
    return RET_ERROR;
  }
  // when tensorlist malloc is done. this need to check element_shape compatibility
  dst_tensorlist->set_element_shape(src_tensorlist->element_shape());

  auto update_data_type = kTypeUnknown;
  auto dst_tensor_data_type = dst_tensorlist->tensors_data_type();
  auto src_tensor_data_type = src_tensorlist->tensors_data_type();
  if (dst_tensor_data_type != src_tensor_data_type) {
    if (src_tensor_data_type != kTypeUnknown && dst_tensor_data_type != kTypeUnknown) {
      MS_LOG(ERROR) << "input tensorlist and output tensorlist is incompatible";
      return RET_ERROR;
    }
    update_data_type = dst_tensor_data_type != kTypeUnknown ? dst_tensor_data_type : src_tensor_data_type;
  }
  if (update_data_type != kTypeUnknown) {
    src_tensorlist->set_tensors_data_type(update_data_type);
    dst_tensorlist->set_tensors_data_type(update_data_type);
  }
  return RET_OK;
}

void SetTensorListTensorDataType(const TypeId &data_type, Tensor *tensor) {
  if (tensor->data_type() == kObjectTypeTensorType) {
    auto old_tensorlist = reinterpret_cast<TensorList *>(tensor);
    if (old_tensorlist->tensors_data_type() == kNumberTypeFloat16 ||
        old_tensorlist->tensors_data_type() == kNumberTypeFloat32) {
      old_tensorlist->set_tensors_data_type(data_type);
    }
  }
}

bool IsSameDtype(const Tensor *input_1, const Tensor *input_2) {
  if (input_1->data_type() != kObjectTypeTensorType && input_2->data_type() != kObjectTypeTensorType) {
    return input_1->data_type() == input_2->data_type();
  } else if (input_1->data_type() == kObjectTypeTensorType && input_2->data_type() == kObjectTypeTensorType) {
    auto input_tensor_list_1 = reinterpret_cast<const TensorList *>(input_1);
    auto input_tensor_list_2 = reinterpret_cast<const TensorList *>(input_2);
    return input_tensor_list_1->tensors_data_type() == input_tensor_list_2->tensors_data_type();
  } else {
    return false;
  }
}

bool IsSameShape(const Tensor *input_1, const Tensor *input_2) {
  if (input_1->data_type() != kObjectTypeTensorType && input_2->data_type() != kObjectTypeTensorType) {
    return input_1->shape() == input_2->shape();
  } else if (input_1->data_type() == kObjectTypeTensorType && input_2->data_type() == kObjectTypeTensorType) {
    auto input_tensor_list_1 = reinterpret_cast<const TensorList *>(input_1);
    auto input_tensor_list_2 = reinterpret_cast<const TensorList *>(input_2);
    return input_tensor_list_1->shape() == input_tensor_list_2->shape() &&
           input_tensor_list_1->element_shape() == input_tensor_list_2->element_shape();
  } else {
    return false;
  }
}

int MallocTensorData(Tensor *tensor) {
  auto ret = RET_OK;
  if (tensor->data_type() != kObjectTypeTensorType) {
    tensor->FreeData();
    auto size = tensor->ElementsNum();
    if (size <= 0) {
      return RET_OK;
    }
    ret = tensor->MallocData();
  } else {
    auto tensor_list = reinterpret_cast<TensorList *>(tensor);
    ret = tensor_list->FreeTensorListData();
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "free tensor list data failed.");
    auto size = tensor->ElementsNum();
    if (size <= 0) {
      return RET_OK;
    }
    std::vector<std::vector<int>> tensors_shape{};
    for (int i = 0; i < size; ++i) {
      tensors_shape.push_back(tensor_list->element_shape());
    }
    ret = tensor_list->MallocTensorListData(tensor_list->tensors_data_type(), tensors_shape);
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "malloc tensor list data failed.");
  }
  return ret;
}

bool IsUnKnownDtype(const Tensor *input) {
  if (input->data_type() == kTypeUnknown) {
    return true;
  } else if (input->data_type() == kObjectTypeTensorType) {
    auto input_tensor_list = reinterpret_cast<const TensorList *>(input);
    return input_tensor_list->tensors_data_type() == kTypeUnknown;
  }
  return false;
}

#else

int SetTensorShape(Tensor *dst, Tensor *src) {
  if (dst->data_type() != kObjectTypeTensorType && src->data_type() != kObjectTypeTensorType) {
    dst->set_shape(src->shape());
    dst->set_format(src->format());
    return RET_OK;
  } else {
    MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
    return RET_ERROR;
  }
}

int CastTensorListTensorData(TensorList *dst_tensorlist, TensorList *src_tensorlist, bool support_fp16) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return RET_ERROR;
}

int MoveTensorListTensorData(TensorList *dst_tensorlist, TensorList *src_tensorlist) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return RET_ERROR;
}

int SetTensorListTensorData(TensorList *dst_tensor_list, TensorList *src_tensor_list) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return RET_ERROR;
}

void FreeTensorListC(TensorListC *tensorlist_c, std::shared_ptr<Allocator> allocator) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return;
}

int TensorList2TensorListC(TensorList *src, TensorListC *dst, std::shared_ptr<Allocator> allocator) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return NNACL_ERR;
}

int TensorListC2TensorList(const TensorListC *src, TensorList *dst) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return RET_ERROR;
}

TypeId TensorListDataType(Tensor *tensor) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return kTypeUnknown;
}

TensorList *MallocTensorListDataAccordingToTensorListC(Tensor *tensor, TensorListC *tensor_list_c) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return nullptr;
}

int DecodeTensorLsit(Tensor *tensor, const int *src_data, size_t length) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return RET_ERROR;
}

Tensor *CreateTensorList(const std::vector<int> &shape, const Category &src_category, const void *src_data) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return nullptr;
}

int CopyTensorListTensorDataType(TensorList *dst_tensorlist, TensorList *src_tensorlist) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return RET_ERROR;
}

void SetTensorListTensorDataType(const TypeId &data_type, Tensor *tensor) {
  MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
  return;
}

bool IsSameDtype(const Tensor *input_1, const Tensor *input_2) {
  if (input_1->data_type() != kObjectTypeTensorType && input_2->data_type() != kObjectTypeTensorType) {
    return input_1->data_type() == input_2->data_type();
  } else {
    MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
    return false;
  }
}

bool IsSameShape(const Tensor *input_1, const Tensor *input_2) {
  if (input_1->data_type() != kObjectTypeTensorType && input_2->data_type() != kObjectTypeTensorType) {
    return input_1->shape() == input_2->shape();
  } else {
    MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
    return false;
  }
}

int MallocTensorData(Tensor *tensor) {
  auto ret = RET_OK;
  if (tensor->data_type() != kObjectTypeTensorType) {
    tensor->FreeData();
    auto size = tensor->ElementsNum();
    if (size <= 0) {
      return RET_OK;
    }
    ret = tensor->MallocData();
  } else {
    MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
    return RET_ERROR;
  }
  return ret;
}

bool IsUnKnownDtype(const Tensor *input) {
  if (input->data_type() == kTypeUnknown) {
    return true;
  }
  return false;
}

#endif
}  // namespace lite
}  // namespace mindspore
