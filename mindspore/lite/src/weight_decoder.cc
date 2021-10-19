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
#include "src/weight_decoder.h"
#include "src/huffman_decode.h"

namespace mindspore::lite {
constexpr int kBit8 = 8;
constexpr int kBit32 = 32;
std::vector<bool> StringToBitVector(const std::string &str) {
  std::vector<bool> vec(str.size() * kBit8);
  size_t index = 0;
  for (auto ch : str) {
    for (size_t shift = kBit8; shift > 0; shift--) {
      vec[index++] = (ch >> static_cast<size_t>(shift - 1)) & 0x1;
    }
  }
  return vec;
}

STATUS IndexingDecompress(const schema::Tensor &src_tensor, Tensor *dst_tensor) {
  MS_LOG(DEBUG) << "un-index weight";
  MS_CHECK_TRUE_MSG(src_tensor.quantParams() != nullptr, RET_ERROR, "quant params is nullptr");
  MS_CHECK_TRUE_MSG((*src_tensor.quantParams()).size() > 0, RET_ERROR, "quant params size need bigger than 0");
  MS_CHECK_TRUE_MSG(src_tensor.quantParams()->Get(0) != nullptr, RET_ERROR, "quant param is nullptr");
  auto bit_num = src_tensor.quantParams()->Get(0)->numBits();

  std::string str(reinterpret_cast<const char *>(src_tensor.data()->data()), src_tensor.data()->size());
  auto bit_vec = StringToBitVector(str);
  size_t index = 0;
  // parse unique_value_cnt
  size_t unique_value_cnt = 0;
  for (int i = 0; i < bit_num; i++) {
    bool bit = bit_vec[index++];
    unique_value_cnt |= bit << static_cast<size_t>((bit_num - i - 1));
  }
  if (unique_value_cnt == 0) {
    unique_value_cnt = 1 << bit_num;
  }
  // parse unique_value_set
  std::vector<int> unique_values;
  for (size_t i = 0; i < unique_value_cnt; i++) {
    int unique_value = 0;
    for (int j = 0; j < bit_num; j++) {
      bool bit = bit_vec[index++];
      unique_value |= bit << static_cast<size_t>((bit_num - j - 1));
    }
    // unsigned to signed
    unique_values.push_back(unique_value - (1 << static_cast<size_t>((bit_num - 1))));
  }
  // parse index
  std::vector<size_t> unique_value_index_vec;
  auto elem_cnt = dst_tensor->ElementsNum();
  size_t unique_value_bit = ceil(log2(unique_value_cnt));
  for (int i = 0; i < elem_cnt; i++) {
    size_t unique_value_index = 0;
    for (size_t j = 0; j < unique_value_bit; j++) {
      bool bit = bit_vec[index++];
      unique_value_index |= bit << (static_cast<size_t>(unique_value_bit - j - 1));
    }
    unique_value_index_vec.push_back(unique_value_index);
  }

  if (dst_tensor->data() != nullptr) {
    MS_LOG(ERROR) << "data_c not null";
    return RET_ERROR;
  }
  auto ret = dst_tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    return RET_NULL_PTR;
  }
  auto dst_data = dst_tensor->data();
  if (bit_num <= kBit8) {
    ret = UnIndexTensorData<int8_t>(unique_values, unique_value_index_vec, dst_data, dst_tensor->Size());
  } else {
    ret = UnIndexTensorData<int16_t>(unique_values, unique_value_index_vec, dst_data, dst_tensor->Size());
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnIndexTensorData error";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS SparseDecompress(const schema::Tensor &src_tensor, Tensor *dst_tensor) {
  MS_LOG(DEBUG) << "un-sparse weight";
  MS_CHECK_TRUE_MSG(src_tensor.quantParams() != nullptr, RET_ERROR, "quant params is nullptr");
  MS_CHECK_TRUE_MSG((*src_tensor.quantParams()).size() > 0, RET_ERROR, "quant params size need bigger than 0");
  MS_CHECK_TRUE_MSG(src_tensor.quantParams()->Get(0) != nullptr, RET_ERROR, "quant param is nullptr");
  size_t bit_num = src_tensor.quantParams()->Get(0)->numBits();

  std::string str(reinterpret_cast<const char *>(src_tensor.data()->data()), src_tensor.data()->size());
  auto bit_vec = StringToBitVector(str);
  size_t index = 0;
  // parse coor_best_bit
  size_t coor_best_bit = 0;
  for (size_t i = 0; i < kBit8; i++) {
    bool bit = bit_vec[index++];
    coor_best_bit |= bit << static_cast<size_t>((kBit8 - i - 1));
  }
  // parse nz_cnt
  size_t nz_cnt = 0;
  for (size_t i = 0; i < kBit32; i++) {
    bool bit = bit_vec[index++];
    nz_cnt |= bit << static_cast<size_t>((kBit32 - i - 1));
  }
  // parse unique_value cnt
  size_t unique_value_cnt = 0;
  for (size_t i = 0; i < bit_num; i++) {
    bool bit = bit_vec[index++];
    unique_value_cnt |= bit << static_cast<size_t>((bit_num - i - 1));
  }
  if (unique_value_cnt == 0) {
    unique_value_cnt = 1 << bit_num;
  }
  // parse unique_values
  std::vector<int> unique_values;
  for (size_t i = 0; i < unique_value_cnt; i++) {
    int unique_value = 0;
    for (size_t j = 0; j < bit_num; j++) {
      bool bit = bit_vec[index++];
      unique_value |= bit << static_cast<size_t>((bit_num - j - 1));
    }
    // unsigned to signed
    unique_values.push_back(unique_value - (1 << static_cast<size_t>((bit_num - 1))));
  }
  // parse index
  std::vector<size_t> unique_value_index_vec;
  auto elem_cnt = dst_tensor->ElementsNum();
  size_t unique_value_bit = static_cast<size_t>(ceil(log2(unique_value_cnt)));
  for (size_t i = 0; i < nz_cnt; i++) {
    size_t unique_value_index = 0;
    for (size_t j = 0; j < unique_value_bit; j++) {
      bool bit = bit_vec[index++];
      unique_value_index |= bit << (unique_value_bit - j - 1);
    }
    unique_value_index_vec.push_back(unique_value_index);
  }

  // parse coors
  std::vector<size_t> coor_vec;
  for (size_t i = 0; i < nz_cnt; i++) {
    size_t coor = 0;
    for (size_t j = 0; j < coor_best_bit; j++) {
      bool bit = bit_vec[index++];
      coor |= bit << static_cast<size_t>((coor_best_bit - j - 1));
    }
    coor_vec.push_back(coor);
  }

  if (dst_tensor->data() != nullptr) {
    MS_LOG(ERROR) << "data_c not null";
    return RET_ERROR;
  }
  auto ret = dst_tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    return RET_NULL_PTR;
  }
  auto dst_data = dst_tensor->data();

  if (bit_num <= kBit8) {
    ret = UnSparseTensorData<int8_t>(unique_values, unique_value_index_vec, coor_vec, src_tensor.quantParams(),
                                     elem_cnt, coor_best_bit, dst_data, dst_tensor->Size());
  } else {
    ret = UnSparseTensorData<int16_t>(unique_values, unique_value_index_vec, coor_vec, src_tensor.quantParams(),
                                      elem_cnt, coor_best_bit, dst_data, dst_tensor->Size());
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnSparseTensorData error";
    return RET_ERROR;
  }
  return RET_OK;
}

int WeightDecoder::DequantWeight(lite::Tensor *input_tensor, bool channel_first, TypeId dst_data_type) {
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
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
  } else if (input_tensor->data_type() == kNumberTypeInt16 && dst_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    auto new_const_data = DequantData<int16_t, float16_t>(input_tensor, channel_first);
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
#else
    MS_LOG(ERROR) << "Float16 is not supported";
    return RET_NOT_SUPPORT;
#endif
  } else if (input_tensor->data_type() == kNumberTypeInt8 && dst_data_type == kNumberTypeFloat32) {
    auto new_const_data = DequantData<int8_t, float>(input_tensor, channel_first);
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
  } else if (input_tensor->data_type() == kNumberTypeInt8 && dst_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    auto new_const_data = DequantData<int8_t, float16_t>(input_tensor, channel_first);
    input_tensor->FreeData();
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

int WeightDecoder::DecodeHuffmanCode(const schema::Tensor &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
  if (!dst_tensor->IsConst() || !src_tensor.enableHuffmanCode()) {
    return RET_NO_CHANGE;
  }
  if (src_tensor.data() == nullptr) {
    return RET_NO_CHANGE;
  }
  auto data = reinterpret_cast<const char *>(src_tensor.data()->data());
  if (data == nullptr) {
    return RET_NO_CHANGE;
  }
  std::string encode_str(data, src_tensor.data()->size());
  dst_tensor->FreeData();
  dst_tensor->set_data(nullptr);
  auto ret = dst_tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    return RET_NULL_PTR;
  }
  auto dst_data = dst_tensor->data();
  MS_ASSERT(dst_data != nullptr);
  ret = HuffmanDecode::DoHuffmanDecode(encode_str, dst_data, dst_tensor->Size());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoHuffmanDecode failed.";
    return ret;
  }
  return RET_OK;
}

int WeightDecoder::UnPackToInt(const schema::Tensor &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
  auto quant_params = src_tensor.quantParams();
  if (quant_params == nullptr || quant_params->size() == 0) {
    return RET_NO_CHANGE;
  }
  auto quant_param = quant_params->Get(0);
  if (quant_param == nullptr) {
    return RET_NO_CHANGE;
  }
  auto dst_data = dst_tensor->data();
  if (dst_data != nullptr) {
    MS_LOG(ERROR) << "lite Tensor has already malloced data";
    return RET_ERROR;
  }
  auto ret = dst_tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    return RET_NULL_PTR;
  }
  dst_data = dst_tensor->data();
  int origin_bit = quant_param->numBits();
  if (origin_bit < kBitNum8 && origin_bit >= kBitNum1) {
    UnPackUtil<int8_t, uint8_t>(&src_tensor, origin_bit, dst_data);
    return RET_OK;
  } else if (origin_bit < kBitNum16 && origin_bit > kBitNum8) {
    UnPackUtil<int16_t, uint16_t>(&src_tensor, origin_bit, dst_data);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupported bit number: " << origin_bit;
    return RET_NOT_SUPPORT;
  }
}

int WeightDecoder::UnPack(const schema::Tensor &src_tensor, lite::Tensor *dst_tensor) {
  STATUS ret = RET_OK;
  if (src_tensor.enableHuffmanCode()) {
    ret = WeightDecoder::DecodeHuffmanCode(src_tensor, dst_tensor);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Decode huffman code failed: " << ret;
    }
  } else {
    ret = WeightDecoder::UnPackToInt(src_tensor, dst_tensor);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Unpack to int8 failed: " << ret;
    }
  }
  return ret;
}

int WeightDecoder::DequantNode(OpParameter *op_parameter, const std::vector<Tensor *> &in_tensors,
                               TypeId dst_data_type) {
  if (op_parameter->quant_type_ != schema::QuantType_QUANT_WEIGHT) {
    return RET_OK;
  }
  int index = 0;
  for (auto &tensor : in_tensors) {
    MS_CHECK_TRUE_RET(tensor != nullptr, RET_ERROR);
    auto channel_first = IsChannelFirst(index++, op_parameter);
    auto ret = WeightDecoder::DequantTensor(tensor, channel_first, dst_data_type);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(DEBUG) << "Dequant tensor failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int WeightDecoder::DequantTensor(Tensor *tensor, bool channel_first, TypeId dst_data_type) {
  MS_ASSERT(tensor != nullptr);
  if (!tensor->IsConst() ||
      !(dst_data_type == TypeId::kNumberTypeFloat32 || dst_data_type == TypeId::kNumberTypeFloat16)) {
    return RET_NO_CHANGE;
  }
  bool need_dequant = !tensor->quant_params().empty() && tensor->quant_params().front().inited &&
                      (tensor->data_type() == kNumberTypeInt8 || tensor->data_type() == kNumberTypeInt16);
  if (!need_dequant) {
    return RET_NO_CHANGE;
  }
  auto ret = WeightDecoder::DequantWeight(tensor, channel_first, dst_data_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Dequant data failed: " << ret;
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
