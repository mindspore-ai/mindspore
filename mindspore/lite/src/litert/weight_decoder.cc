/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "src/litert/weight_decoder.h"
#include "src/litert/huffman_decode.h"
#include "tools/converter/quantizer/fse_decoder.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite {
#ifndef WEIGHT_DECODE_CLIP
int WeightDecoder::DequantWeight(lite::Tensor *input_tensor, int preferred_dim, TypeId dst_data_type) {
  MS_ASSERT(input_tensor != nullptr);
  if (input_tensor->quant_params().empty()) {
    MS_LOG(ERROR) << "No quant param.";
    return RET_ERROR;
  }
  if (input_tensor->data_type() == kNumberTypeInt16 && dst_data_type == kNumberTypeFloat32) {
    auto new_const_data = DequantData<int16_t, float>(input_tensor, preferred_dim);
    CHECK_NULL_RETURN(new_const_data);
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
  } else if (input_tensor->data_type() == kNumberTypeInt16 && dst_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    auto new_const_data = DequantData<int16_t, float16_t>(input_tensor, preferred_dim);
    CHECK_NULL_RETURN(new_const_data);
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
#else
    MS_LOG(ERROR) << "Float16 is not supported";
    return RET_NOT_SUPPORT;
#endif
  } else if (input_tensor->data_type() == kNumberTypeInt8 && dst_data_type == kNumberTypeFloat32) {
    auto new_const_data = DequantData<int8_t, float>(input_tensor, preferred_dim);
    CHECK_NULL_RETURN(new_const_data);
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
  } else if (input_tensor->data_type() == kNumberTypeInt8 && dst_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    auto new_const_data = DequantData<int8_t, float16_t>(input_tensor, preferred_dim);
    CHECK_NULL_RETURN(new_const_data);
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
#else
    MS_LOG(ERROR) << "Float16 is not supported";
    return RET_NOT_SUPPORT;
#endif
  } else if (input_tensor->data_type() == kNumberTypeInt32 && dst_data_type == kNumberTypeFloat32) {
    auto new_const_data = DequantData<int32_t, float>(input_tensor, preferred_dim);
    CHECK_NULL_RETURN(new_const_data);
    input_tensor->FreeData();
    input_tensor->set_data(new_const_data);
    input_tensor->set_own_data(true);
    input_tensor->set_data_type(dst_data_type);
  } else {
    MS_LOG(ERROR) << "Unsupported dequant from data_type(" << (input_tensor->data_type()) << ") to data_type("
                  << dst_data_type << ")";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int WeightDecoder::DecodeKMeansWeight(lite::Tensor *tensor, TypeId dst_data_type = kNumberTypeFloat32) {
  void *dequant_data = nullptr;
  if (dst_data_type == kNumberTypeFloat32) {
    auto dequant_data_ptr = static_cast<float *>(dequant_data);
    DecodeKMeansData(tensor, &dequant_data_ptr);
    dequant_data = dequant_data_ptr;
  } else if (dst_data_type == kNumberTypeFloat16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    auto dequant_data_ptr = static_cast<float16_t *>(dequant_data);
    DecodeKMeansData(tensor, &dequant_data_ptr);
    dequant_data = dequant_data_ptr;
#else
    MS_LOG(ERROR) << "Current library or hardware don't support FP16.";
    return RET_ERROR;
#endif
  } else {
    MS_LOG(ERROR) << dst_data_type << " data type is not support KMeans.";
    return RET_ERROR;
  }
  tensor->FreeData();
  tensor->set_data(dequant_data);
  tensor->set_own_data(true);
  tensor->set_data_type(dst_data_type);
  return RET_OK;
}

int WeightDecoder::DecodeHuffmanCode(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor.handler() != nullptr);
  MS_ASSERT(src_tensor.data() != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
  if (!dst_tensor->IsConst() || !src_tensor.handler()->enableHuffmanCode()) {
    return RET_NO_CHANGE;
  }
  if (src_tensor.data() == nullptr) {
    return RET_NO_CHANGE;
  }
  auto data = reinterpret_cast<const char *>(src_tensor.data());
  if (data == nullptr) {
    return RET_NO_CHANGE;
  }
  std::string encode_str(data, src_tensor.length());
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

int WeightDecoder::UnPackToInt(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor.handler() != nullptr);
  MS_ASSERT(src_tensor.data() != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
  auto quant_params = src_tensor.handler()->quantParams();
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
  auto dst_element_num = dst_tensor->ElementsNum();
  int origin_bit = quant_param->numBits();
  if (origin_bit < kBitNum8 && origin_bit >= kBitNum1) {
    return UnPackUtil<int8_t, uint8_t>(src_tensor, dst_element_num, origin_bit, dst_data);
  } else if (origin_bit < kBitNum16 && origin_bit > kBitNum8) {
    return UnPackUtil<int16_t, uint16_t>(src_tensor, dst_element_num, origin_bit, dst_data);
  } else {
    MS_LOG(ERROR) << "Unsupported bit number: " << origin_bit;
    return RET_NOT_SUPPORT;
  }
}

int WeightDecoder::UnPack(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor.handler() != nullptr);
  MS_ASSERT(src_tensor.data() != nullptr);
  STATUS ret = RET_OK;
  if (src_tensor.handler()->enableHuffmanCode()) {
    ret = WeightDecoder::DecodeHuffmanCode(src_tensor, dst_tensor);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Decode huffman code failed: " << ret;
    }
  } else {
    if (src_tensor.handler()->dims()->size() == 0) {
      MS_LOG(ERROR) << src_tensor.handler()->name()->c_str() << " shape is empty.";
      return RET_ERROR;
    }
    ret = WeightDecoder::UnPackToInt(src_tensor, dst_tensor);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Unpack to int8 failed: " << ret;
      return ret;
    }
  }
  return ret;
}

STATUS WeightDecoder::SparseDecompress(const SchemaTensorWrapper &src_tensor, Tensor *dst_tensor) {
  MS_ASSERT(src_tensor.handler() != nullptr);
  MS_ASSERT(src_tensor.data() != nullptr);
  MS_LOG(DEBUG) << "un-sparse weight";
  MS_CHECK_TRUE_MSG(src_tensor.handler()->quantParams() != nullptr, RET_ERROR, "quant params is nullptr");
  MS_CHECK_TRUE_MSG((*src_tensor.handler()->quantParams()).size() > 0, RET_ERROR,
                    "quant params size need bigger than 0");
  MS_CHECK_TRUE_MSG(src_tensor.handler()->quantParams()->Get(0) != nullptr, RET_ERROR, "quant param is nullptr");
  size_t bit_num = static_cast<size_t>(src_tensor.handler()->quantParams()->Get(0)->numBits());

  std::string str(static_cast<const char *>(src_tensor.data()), src_tensor.length());
  auto bit_vec = StringToBitVector(str);
  size_t index = 0;
  // parse coor_best_bit
  size_t coor_best_bit = 0;
  for (size_t i = 0; i < kBitNum8; i++) {
    bool bit = bit_vec[index++];
    coor_best_bit |= bit << static_cast<size_t>((kBitNum8 - i - 1));
  }
  // parse nz_cnt
  size_t nz_cnt = 0;
  for (size_t i = 0; i < kBitNum32; i++) {
    bool bit = bit_vec[index++];
    nz_cnt |= bit << static_cast<size_t>((kBitNum32 - i - 1));
  }
  // parse unique_value cnt
  size_t unique_value_cnt = 0;
  for (size_t i = 0; i < bit_num; i++) {
    bool bit = bit_vec[index++];
    unique_value_cnt |= bit << (bit_num - i - 1);
  }
  if (unique_value_cnt == 0) {
    unique_value_cnt = 1u << bit_num;
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
    unique_values.push_back(unique_value - (1u << static_cast<size_t>((bit_num - 1))));
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

  if (bit_num <= kBitNum8) {
    ret =
      UnSparseTensorData<int8_t>(unique_values, unique_value_index_vec, coor_vec, src_tensor.handler()->quantParams(),
                                 elem_cnt, coor_best_bit, dst_data, dst_tensor->Size());
  } else {
    ret =
      UnSparseTensorData<int16_t>(unique_values, unique_value_index_vec, coor_vec, src_tensor.handler()->quantParams(),
                                  elem_cnt, coor_best_bit, dst_data, dst_tensor->Size());
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnSparseTensorData error";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<bool> WeightDecoder::StringToBitVector(const std::string &str) {
  std::vector<bool> vec(str.size() * kBitNum8);
  size_t index = 0;
  for (auto ch : str) {
    for (size_t shift = kBitNum8; shift > 0; shift--) {
      vec[index++] = (static_cast<unsigned char>(ch) >> static_cast<size_t>(shift - 1)) & 0x1;
    }
  }
  return vec;
}

STATUS WeightDecoder::IndexingDecompress(const SchemaTensorWrapper &src_tensor, Tensor *dst_tensor) {
  MS_ASSERT(src_tensor.handler() != nullptr);
  MS_ASSERT(src_tensor.data() != nullptr);
  MS_LOG(DEBUG) << "un-index weight";
  MS_CHECK_TRUE_MSG(src_tensor.handler()->quantParams() != nullptr, RET_ERROR, "quant params is nullptr");
  MS_CHECK_TRUE_MSG((*src_tensor.handler()->quantParams()).size() > 0, RET_ERROR,
                    "quant params size need bigger than 0");
  MS_CHECK_TRUE_MSG(src_tensor.handler()->quantParams()->Get(0) != nullptr, RET_ERROR, "quant param is nullptr");
  auto bit_num = src_tensor.handler()->quantParams()->Get(0)->numBits();

  std::string str(static_cast<const char *>(src_tensor.data()), src_tensor.length());
  auto bit_vec = StringToBitVector(str);
  size_t index = 0;
  // parse unique_value_cnt
  size_t unique_value_cnt = 0;
  for (int i = 0; i < bit_num; i++) {
    bool bit = bit_vec[index++];
    unique_value_cnt |= bit << static_cast<size_t>((bit_num - i - 1));
  }
  if (unique_value_cnt == 0) {
    unique_value_cnt = 1u << bit_num;
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
    unique_values.push_back(unique_value - (1u << static_cast<size_t>((bit_num - 1))));
  }
  // parse index
  std::vector<size_t> unique_value_index_vec;
  auto elem_cnt = dst_tensor->ElementsNum();
  size_t unique_value_bit = static_cast<size_t>(ceil(log2(unique_value_cnt)));
  for (int i = 0; i < elem_cnt; i++) {
    size_t unique_value_index = 0;
    for (size_t j = 0; j < unique_value_bit; j++) {
      bool bit = bit_vec[index++];
      unique_value_index |= bit << (static_cast<size_t>(unique_value_bit - j - 1));
    }
    unique_value_index_vec.push_back(unique_value_index);
  }

  MS_CHECK_FALSE_MSG(dst_tensor->data() != nullptr, RET_ERROR, "data_c not null");
  if (dst_tensor->MallocData() != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    return RET_NULL_PTR;
  }
  auto dst_data = dst_tensor->data();
  int ret;
  if (bit_num <= kBitNum8) {
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

int WeightDecoder::DequantTensor(Tensor *tensor, int preferred_dim, TypeId dst_data_type) {
  MS_ASSERT(tensor != nullptr);
  if (!tensor->IsConst() ||
      !(dst_data_type == TypeId::kNumberTypeFloat32 || dst_data_type == TypeId::kNumberTypeFloat16)) {
    return RET_NO_CHANGE;
  }
  if (!tensor->quant_params().empty()) {
    bool need_dequant = tensor->quant_params().front().inited &&
                        (tensor->data_type() == kNumberTypeInt8 || tensor->data_type() == kNumberTypeInt16 ||
                         tensor->data_type() == kNumberTypeInt32);
    if (!need_dequant) {
      return RET_NO_CHANGE;
    }
    auto ret = WeightDecoder::DequantWeight(tensor, preferred_dim, dst_data_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << tensor->tensor_name() << " Dequant data failed: " << ret;
      return ret;
    }
  } else if (!tensor->quant_clusters().empty()) {
    auto ret = DecodeKMeansWeight(tensor, dst_data_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << tensor->tensor_name() << " Decode KMeans weight failed: " << ret;
      return ret;
    }
  }
  return RET_OK;
}

int WeightDecoder::GetMatMulPreferredDim(const OpParameter *op_parameter, int input_index,
                                         const std::vector<int> &dims) {
  int last_first_index = static_cast<int>(dims.size()) - 1;
  int last_second_index = static_cast<int>(dims.size()) - 2;
  auto matmul_parameter = reinterpret_cast<const MatMulParameter *>(op_parameter);
  MS_ASSERT(matmul_parameter != nullptr);
  // For MatMul A
  if (input_index == 0) {
    if (matmul_parameter->a_transpose_) {
      return last_first_index;
    } else {
      return last_second_index;
    }
  }
  // For MatMul B
  if (input_index == 1) {
    if (matmul_parameter->b_transpose_) {
      return last_second_index;
    } else {
      return last_first_index;
    }
  }
  return 0;
}

int WeightDecoder::GetDeConvPreferredDim(const OpParameter *op_parameter, const std::vector<int> &dims) {
  MS_ASSERT(op_parameter != nullptr);
  auto parameter = reinterpret_cast<const ConvParameter *>(op_parameter);
  if (parameter->input_channel_ == parameter->group_ && parameter->output_channel_ == parameter->group_) {
    // DepthWise-DeConv (CO\CI) KH KW 1
    return 0;
  } else {
    // DeConv:CI KH KW CO
    return dims.size() - 1;
  }
}

bool WeightDecoder::IsChannelFirst(int index, const OpParameter *op_parameter) {
  MS_ASSERT(op_parameter != nullptr);
  if (op_parameter->type_ == schema::PrimitiveType_MatMulFusion) {
    const auto *param = reinterpret_cast<const MatMulParameter *>(op_parameter);
    if (index == 0) {
      return !(param->a_transpose_);
    } else if (index == 1) {
      return param->b_transpose_;
    }
  }
  return true;
}

// A * stride_a + bucket_index * stride_b + C
int WeightDecoder::GetDataIndex(const std::vector<int> &dims, int preferred_dim, int bucket_index,
                                int bucket_in_index) {
  int stride_a = 1;
  for (size_t i = static_cast<size_t>(preferred_dim); i < dims.size(); i++) {
    stride_a *= dims[i];
  }
  int stride_b = 1;
  for (size_t i = static_cast<size_t>(preferred_dim) + 1; i < dims.size(); i++) {
    stride_b *= dims[i];
  }
  MS_ASSERT(stride_b > 0);
  int A = bucket_in_index / stride_b;
  int C = bucket_in_index % stride_b;
  return A * stride_a + bucket_index * stride_b + C;
}

#endif

bool NeedBitUppackCheck(const SchemaTensorWrapper &src_tensor) {
  MS_ASSERT(src_tensor.handler() != nullptr);
  MS_ASSERT(src_tensor.data() != nullptr);
  if (src_tensor.handler()->enableHuffmanCode()) {
    return true;
  }
  bool need_bit_unpack = src_tensor.handler()->quantParams() != nullptr &&
                         src_tensor.handler()->quantParams()->size() > 0 &&
                         src_tensor.handler()->quantParams()->Get(0) != nullptr;
  if (need_bit_unpack) {
    auto num_bits = src_tensor.handler()->quantParams()->Get(0)->numBits();
    need_bit_unpack = ((num_bits >= kBitNum1 && num_bits < kBitNum8) || (num_bits > kBitNum8 && num_bits < kBitNum16));
  }

  return need_bit_unpack;
}

int WeightDecoder::DequantNode(const OpParameter *op_parameter, const std::vector<Tensor *> &in_tensors,
                               TypeId dst_data_type, const std::string &model_version, bool float_mode) {
#ifndef WEIGHT_DECODE_CLIP
  if (op_parameter->quant_type_ != static_cast<int>(schema::QuantType_QUANT_WEIGHT) &&
      !(op_parameter->quant_type_ == static_cast<int>(schema::QuantType_QUANT_ALL) && float_mode)) {
    return RET_OK;
  }
  int index = 0;
  for (auto &tensor : in_tensors) {
    MS_CHECK_TRUE_RET(tensor != nullptr, RET_ERROR);
    auto preferred_dim = GetPreferredDim(in_tensors, op_parameter, index++, tensor->shape(), model_version);
    auto ret = WeightDecoder::DequantTensor(tensor, preferred_dim, dst_data_type);
    if (ret != RET_OK && ret != RET_NO_CHANGE) {
      MS_LOG(DEBUG) << "Dequant tensor failed";
      return RET_ERROR;
    }
    tensor->ClearQuantParam();
  }
  return RET_OK;
#else
  if (op_parameter->quant_type_ != schema::QuantType_QUANT_WEIGHT &&
      !(op_parameter->quant_type_ == schema::QuantType_QUANT_ALL && float_mode)) {
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Do not support dequant node.";
    return RET_NOT_SUPPORT;
  }
#endif
}

int WeightDecoder::DecompressTensor(const SchemaTensorWrapper &src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor.handler() != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
#ifndef WEIGHT_DECODE_CLIP
  if (src_tensor.handler()->weightQuantCompressType() == schema::WeightQuantCompressType_FSE ||
      src_tensor.handler()->weightQuantCompressType() == schema::WeightQuantCompressType_FSE_INT) {
    return quant::FSEDecoder::DeCompress(src_tensor, dst_tensor, src_tensor.handler()->weightQuantCompressType());
  } else if (src_tensor.handler()->weightQuantCompressType() == schema::WeightQuantCompressType_INDEXING) {
    return IndexingDecompress(src_tensor, dst_tensor);
  } else if (src_tensor.handler()->weightQuantCompressType() == schema::WeightQuantCompressType_SPARSE) {
    return SparseDecompress(src_tensor, dst_tensor);
  }
  if (!NeedBitUppackCheck(src_tensor)) {
    return RET_NO_CHANGE;
  } else {
    return WeightDecoder::UnPack(src_tensor, dst_tensor);
  }
#else
  if (src_tensor.handler()->weightQuantCompressType() != schema::WeightQuantCompressType_NONE) {
    MS_LOG(ERROR) << unsupport_weight_decode_log;
    return RET_ERROR;
  }
  if (NeedBitUppackCheck(src_tensor)) {
    MS_LOG(ERROR) << unsupport_weight_decode_log;
    return RET_ERROR;
  }
  return RET_NO_CHANGE;
#endif
}
}  // namespace mindspore::lite
