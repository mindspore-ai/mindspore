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
#include <cmath>
#include <string>
#include <memory>
#include "src/dequant.h"
#include "src/huffman_decode.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::lite {
int DequantUtil::DequantWeight(lite::Tensor *input_tensor, bool channel_first, TypeId dst_data_type) {
  MS_ASSERT(input_tensor != nullptr);
  if (input_tensor->data_type() != kNumberTypeInt8 && input_tensor->data_type() != kNumberTypeInt16) {
    MS_LOG(ERROR) << "Conv weight input type error." << input_tensor->data_type();
    return RET_ERROR;
  }
  if (input_tensor->quant_params().empty()) {
    MS_LOG(ERROR) << "No quant param.";
    return RET_ERROR;
  }
  if (input_tensor->data_type() == kNumberTypeInt16 && dst_data_type == kNumberTypeFloat32) {
    auto new_const_data = DequantData<int16_t, float>(input_tensor, channel_first);
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
  } else if (input_tensor->data_type() == kNumberTypeInt16 && dst_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM64) && defined(ENABLE_FP16)
    auto new_const_data = DequantData<int16_t, float16_t>(input_tensor, channel_first);
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
#else
    MS_LOG(ERROR) << "Float16 is not supported";
    return RET_NOT_SUPPORT;
#endif
  } else if (input_tensor->data_type() == kNumberTypeInt8 && dst_data_type == kNumberTypeFloat32) {
    auto new_const_data = DequantData<int8_t, float>(input_tensor, channel_first);
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
  } else if (input_tensor->data_type() == kNumberTypeInt8 && dst_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM64) && defined(ENABLE_FP16)
    auto new_const_data = DequantData<int8_t, float16_t>(input_tensor, channel_first);
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
#else
    MS_LOG(ERROR) << "Float16 is not supported";
    return RET_NOT_SUPPORT;
#endif
  } else {
    MS_LOG(ERROR) << "Unsupported dequant from data_type(" << (input_tensor->data_type()) << ") to data_type("
                  << dst_data_type << ")";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int DequantUtil::DecodeHuffmanCode(const schema::Tensor &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
  if (!dst_tensor->IsConst() || !src_tensor.enableHuffmanCode()) {
    return RET_NO_CHANGE;
  }
  auto data = reinterpret_cast<const char *>(src_tensor.data()->data());
  MS_ASSERT(data != nullptr);
  std::string encode_str(data, src_tensor.data()->size());
  dst_tensor->set_data(nullptr);
  auto ret = dst_tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    return RET_NULL_PTR;
  }
  auto dst_data = dst_tensor->data_c();
  MS_ASSERT(dst_data != nullptr);
  ret = HuffmanDecode::DoHuffmanDecode(encode_str, dst_data);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoHuffmanDecode failed.";
    return ret;
  }
  return RET_OK;
}

int DequantUtil::UnPackToInt(const schema::Tensor &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
  if (!dst_tensor->IsConst()) {
    return RET_NO_CHANGE;
  }
  auto quant_params = src_tensor.quantParams();
  if (quant_params == nullptr || quant_params->size() == 0) {
    return RET_NO_CHANGE;
  }
  auto quant_param = quant_params->Get(0);
  if (quant_param == nullptr || !quant_param->inited()) {
    return RET_NO_CHANGE;
  }
  auto dst_data = dst_tensor->data_c();
  if (dst_data != nullptr) {
    MS_LOG(ERROR) << "lite Tensor has already malloced data";
    return RET_ERROR;
  }
  auto ret = dst_tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    return RET_NULL_PTR;
  }
  dst_data = dst_tensor->data_c();
  int origin_bit = quant_param->numBits();
  if (origin_bit < 8 && origin_bit > 0) {
    UnPackUtil<int8_t, uint8_t>(&src_tensor, origin_bit, dst_data);
    return RET_OK;
  } else if (origin_bit < 16 && origin_bit > 8) {
    UnPackUtil<int16_t, uint16_t>(&src_tensor, origin_bit, dst_data);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupported bit number: " << origin_bit;
    return RET_NOT_SUPPORT;
  }
}

Tensor *DequantUtil::DequantTensor(Tensor *tensor, TypeId data_type, bool channel_first, TypeId dst_data_type) {
  MS_ASSERT(tensor != nullptr);
  Tensor *restore_tensor = nullptr;
  if (!tensor->IsConst() || !(data_type == TypeId::kNumberTypeFloat32 || data_type == TypeId::kNumberTypeFloat16)) {
    return nullptr;
  }
  auto restore_type = tensor->data_type();
  bool need_dequant = !tensor->quant_params().empty() && tensor->quant_params().front().inited &&
                      (restore_type == kNumberTypeInt8 || restore_type == kNumberTypeInt16);
  if (!need_dequant) {
    return nullptr;
  }
  restore_tensor = Tensor::CopyTensor(*tensor, false);
  restore_tensor->set_data(tensor->data_c());
  restore_tensor->set_own_data(tensor->own_data());
  auto ret = DequantUtil::DequantWeight(tensor, channel_first, dst_data_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Dequant data failed: " << ret;
    return nullptr;
  }
  return restore_tensor;
}
}  // namespace mindspore::lite
