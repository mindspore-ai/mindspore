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

#include "minddata/mindrecord/include/shard_column.h"

#include "utils/ms_utils.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_error.h"

namespace mindspore {
namespace mindrecord {
ShardColumn::ShardColumn(const std::shared_ptr<ShardHeader> &shard_header, bool compress_integer) {
  auto first_schema = shard_header->GetSchemas()[0];
  json schema_json = first_schema->GetSchema();
  Init(schema_json, compress_integer);
}

ShardColumn::ShardColumn(const json &schema_json, bool compress_integer) { Init(schema_json, compress_integer); }

void ShardColumn::Init(const json &schema_json, bool compress_integer) {
  auto schema = schema_json["schema"];
  auto blob_fields = schema_json["blob_fields"];

  bool has_integer_array = false;
  for (json::iterator it = schema.begin(); it != schema.end(); ++it) {
    const std::string &column_name = it.key();
    column_name_.push_back(column_name);

    json it_value = it.value();

    std::string str_type = it_value["type"];
    column_data_type_.push_back(ColumnDataTypeMap.at(str_type));
    if (it_value.find("shape") != it_value.end()) {
      std::vector<int64_t> vec(it_value["shape"].size());
      std::copy(it_value["shape"].begin(), it_value["shape"].end(), vec.begin());
      column_shape_.push_back(vec);
      if (str_type == "int32" || str_type == "int64") {
        has_integer_array = true;
      }
    } else {
      std::vector<int64_t> vec = {};
      column_shape_.push_back(vec);
    }
  }

  for (uint64_t i = 0; i < column_name_.size(); i++) {
    column_name_id_[column_name_[i]] = i;
  }

  for (const auto &field : blob_fields) {
    blob_column_.push_back(field);
  }

  for (uint64_t i = 0; i < blob_column_.size(); i++) {
    blob_column_id_[blob_column_[i]] = i;
  }

  has_compress_blob_ = (compress_integer && has_integer_array);
  num_blob_column_ = blob_column_.size();
}

std::pair<MSRStatus, ColumnCategory> ShardColumn::GetColumnTypeByName(const std::string &column_name,
                                                                      ColumnDataType *column_data_type,
                                                                      uint64_t *column_data_type_size,
                                                                      std::vector<int64_t> *column_shape) {
  // Skip if column not found
  auto column_category = CheckColumnName(column_name);
  if (column_category == ColumnNotFound) {
    return {FAILED, ColumnNotFound};
  }

  // Get data type and size
  auto column_id = column_name_id_[column_name];
  *column_data_type = column_data_type_[column_id];
  *column_data_type_size = ColumnDataTypeSize[*column_data_type];
  *column_shape = column_shape_[column_id];

  return {SUCCESS, column_category};
}

MSRStatus ShardColumn::GetColumnValueByName(const std::string &column_name, const std::vector<uint8_t> &columns_blob,
                                            const json &columns_json, const unsigned char **data,
                                            std::unique_ptr<unsigned char[]> *data_ptr, uint64_t *const n_bytes,
                                            ColumnDataType *column_data_type, uint64_t *column_data_type_size,
                                            std::vector<int64_t> *column_shape) {
  // Skip if column not found
  auto column_category = CheckColumnName(column_name);
  if (column_category == ColumnNotFound) {
    return FAILED;
  }

  // Get data type and size
  auto column_id = column_name_id_[column_name];
  *column_data_type = column_data_type_[column_id];
  *column_data_type_size = ColumnDataTypeSize[*column_data_type];
  *column_shape = column_shape_[column_id];

  // Retrieve value from json
  if (column_category == ColumnInRaw) {
    if (GetColumnFromJson(column_name, columns_json, data_ptr, n_bytes) == FAILED) {
      MS_LOG(ERROR) << "Error when get data from json, column name is " << column_name << ".";
      return FAILED;
    }
    *data = reinterpret_cast<const unsigned char *>(data_ptr->get());
    return SUCCESS;
  }

  // Retrieve value from blob
  if (GetColumnFromBlob(column_name, columns_blob, data, data_ptr, n_bytes) == FAILED) {
    MS_LOG(ERROR) << "Error when get data from blob, column name is " << column_name << ".";
    return FAILED;
  }
  if (*data == nullptr) {
    *data = reinterpret_cast<const unsigned char *>(data_ptr->get());
  }
  return SUCCESS;
}

MSRStatus ShardColumn::GetColumnFromJson(const std::string &column_name, const json &columns_json,
                                         std::unique_ptr<unsigned char[]> *data_ptr, uint64_t *n_bytes) {
  auto column_id = column_name_id_[column_name];
  auto column_data_type = column_data_type_[column_id];

  // Initialize num bytes
  *n_bytes = ColumnDataTypeSize[column_data_type];
  auto json_column_value = columns_json[column_name];
  if (!json_column_value.is_string() && !json_column_value.is_number()) {
    MS_LOG(ERROR) << "Conversion failed (" << json_column_value << ").";
    return FAILED;
  }
  switch (column_data_type) {
    case ColumnFloat32: {
      return GetFloat<float>(data_ptr, json_column_value, false);
    }
    case ColumnFloat64: {
      return GetFloat<double>(data_ptr, json_column_value, true);
    }
    case ColumnInt32: {
      return GetInt<int32_t>(data_ptr, json_column_value);
    }
    case ColumnInt64: {
      return GetInt<int64_t>(data_ptr, json_column_value);
    }
    default: {
      // Convert string to c_str
      std::string tmp_string;
      if (json_column_value.is_string()) {
        tmp_string = json_column_value.get<string>();
      } else {
        tmp_string = json_column_value.dump();
      }
      *n_bytes = tmp_string.size();
      auto data = reinterpret_cast<const unsigned char *>(common::SafeCStr(tmp_string));
      *data_ptr = std::make_unique<unsigned char[]>(*n_bytes);
      for (uint32_t i = 0; i < *n_bytes; i++) {
        (*data_ptr)[i] = *(data + i);
      }
      break;
    }
  }
  return SUCCESS;
}

template <typename T>
MSRStatus ShardColumn::GetFloat(std::unique_ptr<unsigned char[]> *data_ptr, const json &json_column_value,
                                bool use_double) {
  std::unique_ptr<T[]> array_data = std::make_unique<T[]>(1);
  if (json_column_value.is_number()) {
    array_data[0] = json_column_value;
  } else {
    // Convert string to float
    try {
      if (use_double) {
        array_data[0] = json_column_value.get<double>();
      } else {
        array_data[0] = json_column_value.get<float>();
      }
    } catch (json::exception &e) {
      MS_LOG(ERROR) << "Conversion to float failed (" << json_column_value << ").";
      return FAILED;
    }
  }

  auto data = reinterpret_cast<const unsigned char *>(array_data.get());
  *data_ptr = std::make_unique<unsigned char[]>(sizeof(T));
  for (uint32_t i = 0; i < sizeof(T); i++) {
    (*data_ptr)[i] = *(data + i);
  }

  return SUCCESS;
}

template <typename T>
MSRStatus ShardColumn::GetInt(std::unique_ptr<unsigned char[]> *data_ptr, const json &json_column_value) {
  std::unique_ptr<T[]> array_data = std::make_unique<T[]>(1);
  int64_t temp_value;
  bool less_than_zero = false;

  if (json_column_value.is_number_integer()) {
    const json json_zero = 0;
    if (json_column_value < json_zero) less_than_zero = true;
    temp_value = json_column_value;
  } else if (json_column_value.is_string()) {
    std::string string_value = json_column_value;

    if (!string_value.empty() && string_value[0] == '-') {
      try {
        temp_value = std::stoll(string_value);
        less_than_zero = true;
      } catch (std::invalid_argument &e) {
        MS_LOG(ERROR) << "Conversion to int failed, invalid argument.";
        return FAILED;
      } catch (std::out_of_range &e) {
        MS_LOG(ERROR) << "Conversion to int failed, out of range.";
        return FAILED;
      }
    } else {
      try {
        temp_value = static_cast<int64_t>(std::stoull(string_value));
      } catch (std::invalid_argument &e) {
        MS_LOG(ERROR) << "Conversion to int failed, invalid argument.";
        return FAILED;
      } catch (std::out_of_range &e) {
        MS_LOG(ERROR) << "Conversion to int failed, out of range.";
        return FAILED;
      }
    }
  } else {
    MS_LOG(ERROR) << "Conversion to int failed.";
    return FAILED;
  }

  if ((less_than_zero && temp_value < static_cast<int64_t>(std::numeric_limits<T>::min())) ||
      (!less_than_zero && static_cast<uint64_t>(temp_value) > static_cast<uint64_t>(std::numeric_limits<T>::max()))) {
    MS_LOG(ERROR) << "Conversion to int failed. Out of range";
    return FAILED;
  }
  array_data[0] = static_cast<T>(temp_value);

  auto data = reinterpret_cast<const unsigned char *>(array_data.get());
  *data_ptr = std::make_unique<unsigned char[]>(sizeof(T));
  for (uint32_t i = 0; i < sizeof(T); i++) {
    (*data_ptr)[i] = *(data + i);
  }

  return SUCCESS;
}

MSRStatus ShardColumn::GetColumnFromBlob(const std::string &column_name, const std::vector<uint8_t> &columns_blob,
                                         const unsigned char **data, std::unique_ptr<unsigned char[]> *data_ptr,
                                         uint64_t *const n_bytes) {
  uint64_t offset_address = 0;
  auto column_id = column_name_id_[column_name];
  if (GetColumnAddressInBlock(column_id, columns_blob, n_bytes, &offset_address) == FAILED) {
    return FAILED;
  }

  auto column_data_type = column_data_type_[column_id];
  if (has_compress_blob_ && column_data_type == ColumnInt32) {
    if (UncompressInt<int32_t>(column_id, data_ptr, columns_blob, n_bytes, offset_address) == FAILED) {
      return FAILED;
    }
  } else if (has_compress_blob_ && column_data_type == ColumnInt64) {
    if (UncompressInt<int64_t>(column_id, data_ptr, columns_blob, n_bytes, offset_address) == FAILED) {
      return FAILED;
    }
  } else {
    *data = reinterpret_cast<const unsigned char *>(&(columns_blob[offset_address]));
  }

  return SUCCESS;
}

ColumnCategory ShardColumn::CheckColumnName(const std::string &column_name) {
  auto it_column = column_name_id_.find(column_name);
  if (it_column == column_name_id_.end()) {
    return ColumnNotFound;
  }
  auto it_blob = blob_column_id_.find(column_name);
  return it_blob == blob_column_id_.end() ? ColumnInRaw : ColumnInBlob;
}

std::vector<uint8_t> ShardColumn::CompressBlob(const std::vector<uint8_t> &blob, int64_t *compression_size) {
  // Skip if no compress columns
  *compression_size = 0;
  if (!CheckCompressBlob()) return blob;

  std::vector<uint8_t> dst_blob;
  uint64_t i_src = 0;
  for (int64_t i = 0; i < num_blob_column_; i++) {
    // Get column data type
    auto src_data_type = column_data_type_[column_name_id_[blob_column_[i]]];
    auto int_type = src_data_type == ColumnInt32 ? kInt32Type : kInt64Type;

    // Compress and return is blob has 1 column only
    if (num_blob_column_ == 1) {
      dst_blob = CompressInt(blob, int_type);
      *compression_size = static_cast<int64_t>(blob.size()) - static_cast<int64_t>(dst_blob.size());
      return dst_blob;
    }

    // Just copy and continue if column dat type is not int32/int64
    uint64_t num_bytes = BytesBigToUInt64(blob, i_src, kInt64Type);
    if (src_data_type != ColumnInt32 && src_data_type != ColumnInt64) {
      dst_blob.insert(dst_blob.end(), blob.begin() + i_src, blob.begin() + i_src + kInt64Len + num_bytes);
      i_src += kInt64Len + num_bytes;
      continue;
    }

    // Get column slice in source blob
    std::vector<uint8_t> blob_slice(blob.begin() + i_src + kInt64Len, blob.begin() + i_src + kInt64Len + num_bytes);
    // Compress column
    auto dst_blob_slice = CompressInt(blob_slice, int_type);
    // Get new column size
    auto new_blob_size = UIntToBytesBig(dst_blob_slice.size(), kInt64Type);
    // Append new column size
    dst_blob.insert(dst_blob.end(), new_blob_size.begin(), new_blob_size.end());
    // Append new column data
    dst_blob.insert(dst_blob.end(), dst_blob_slice.begin(), dst_blob_slice.end());
    i_src += kInt64Len + num_bytes;
  }
  MS_LOG(DEBUG) << "Compress all blob from " << blob.size() << " to " << dst_blob.size() << ".";
  *compression_size = static_cast<int64_t>(blob.size()) - static_cast<int64_t>(dst_blob.size());
  return dst_blob;
}

vector<uint8_t> ShardColumn::CompressInt(const vector<uint8_t> &src_bytes, const IntegerType &int_type) {
  uint64_t i_size = kUnsignedOne << static_cast<uint8_t>(int_type);
  // Get number of elements
  uint64_t src_n_int = src_bytes.size() / i_size;
  // Calculate bitmap size (bytes)
  uint64_t bitmap_size = (src_n_int + kNumDataOfByte - 1) / kNumDataOfByte;

  // Initialize destination blob, more space than needed, will be resized
  vector<uint8_t> dst_bytes(kBytesOfColumnLen + bitmap_size + src_bytes.size(), 0);

  // Write number of elements to destination blob
  vector<uint8_t> size_by_bytes = UIntToBytesBig(src_n_int, kInt32Type);
  for (uint64_t n = 0; n < kBytesOfColumnLen; n++) {
    dst_bytes[n] = size_by_bytes[n];
  }

  // Write compressed int
  uint64_t i_dst = kBytesOfColumnLen + bitmap_size;
  for (uint64_t i = 0; i < src_n_int; i++) {
    // Initialize destination data type
    IntegerType dst_int_type = kInt8Type;
    // Shift to next int position
    uint64_t pos = i * (kUnsignedOne << static_cast<uint8_t>(int_type));
    // Narrow down this int
    int64_t i_n = BytesLittleToMinIntType(src_bytes, pos, int_type, &dst_int_type);

    // Write this int to destination blob
    uint64_t u_n = *reinterpret_cast<uint64_t *>(&i_n);
    auto temp_bytes = UIntToBytesLittle(u_n, dst_int_type);
    for (uint64_t j = 0; j < (kUnsignedOne << static_cast<uint8_t>(dst_int_type)); j++) {
      dst_bytes[i_dst++] = temp_bytes[j];
    }

    // Update date type in bit map
    dst_bytes[i / kNumDataOfByte + kBytesOfColumnLen] |=
      (static_cast<uint8_t>(dst_int_type) << (kDataTypeBits * (kNumDataOfByte - kUnsignedOne - (i % kNumDataOfByte))));
  }
  // Resize destination blob
  dst_bytes.resize(i_dst);
  MS_LOG(DEBUG) << "Compress blob field from " << src_bytes.size() << " to " << dst_bytes.size() << ".";
  return dst_bytes;
}

MSRStatus ShardColumn::GetColumnAddressInBlock(const uint64_t &column_id, const std::vector<uint8_t> &columns_blob,
                                               uint64_t *num_bytes, uint64_t *shift_idx) {
  if (num_blob_column_ == 1) {
    *num_bytes = columns_blob.size();
    *shift_idx = 0;
    return SUCCESS;
  }
  auto blob_id = blob_column_id_[column_name_[column_id]];

  for (int32_t i = 0; i < blob_id; i++) {
    *shift_idx += kInt64Len + BytesBigToUInt64(columns_blob, *shift_idx, kInt64Type);
  }
  *num_bytes = BytesBigToUInt64(columns_blob, *shift_idx, kInt64Type);

  (*shift_idx) += kInt64Len;

  return SUCCESS;
}

template <typename T>
MSRStatus ShardColumn::UncompressInt(const uint64_t &column_id, std::unique_ptr<unsigned char[]> *const data_ptr,
                                     const std::vector<uint8_t> &columns_blob, uint64_t *num_bytes,
                                     uint64_t shift_idx) {
  auto num_elements = BytesBigToUInt64(columns_blob, shift_idx, kInt32Type);
  *num_bytes = sizeof(T) * num_elements;

  // Parse integer array
  uint64_t i_source = shift_idx + kBytesOfColumnLen + (num_elements + kNumDataOfByte - 1) / kNumDataOfByte;
  auto array_data = std::make_unique<T[]>(num_elements);

  for (uint64_t i = 0; i < num_elements; i++) {
    uint8_t iBitMap = columns_blob[shift_idx + kBytesOfColumnLen + i / kNumDataOfByte];
    uint64_t i_type = (iBitMap >> ((kNumDataOfByte - 1 - (i % kNumDataOfByte)) * kDataTypeBits)) & kDataTypeBitMask;
    auto mr_int_type = static_cast<IntegerType>(i_type);
    int64_t i64 = BytesLittleToMinIntType(columns_blob, i_source, mr_int_type);
    i_source += (kUnsignedOne << i_type);
    array_data[i] = static_cast<T>(i64);
  }

  auto data = reinterpret_cast<const unsigned char *>(array_data.get());
  *data_ptr = std::make_unique<unsigned char[]>(*num_bytes);
  int ret_code = memcpy_s(data_ptr->get(), *num_bytes, data, *num_bytes);
  if (ret_code != 0) {
    MS_LOG(ERROR) << "Failed to copy data!";
  }

  return SUCCESS;
}

uint64_t ShardColumn::BytesBigToUInt64(const std::vector<uint8_t> &bytes_array, const uint64_t &pos,
                                       const IntegerType &i_type) {
  uint64_t result = 0;
  for (uint64_t i = 0; i < (kUnsignedOne << static_cast<uint8_t>(i_type)); i++) {
    result = (result << kBitsOfByte) + bytes_array[pos + i];
  }
  return result;
}

std::vector<uint8_t> ShardColumn::UIntToBytesBig(uint64_t value, const IntegerType &i_type) {
  uint64_t n_bytes = kUnsignedOne << static_cast<uint8_t>(i_type);
  std::vector<uint8_t> result(n_bytes, 0);
  for (uint64_t i = 0; i < n_bytes; i++) {
    result[n_bytes - 1 - i] = value & std::numeric_limits<uint8_t>::max();
    value >>= kBitsOfByte;
  }
  return result;
}

std::vector<uint8_t> ShardColumn::UIntToBytesLittle(uint64_t value, const IntegerType &i_type) {
  uint64_t n_bytes = kUnsignedOne << static_cast<uint8_t>(i_type);
  std::vector<uint8_t> result(n_bytes, 0);
  for (uint64_t i = 0; i < n_bytes; i++) {
    result[i] = value & std::numeric_limits<uint8_t>::max();
    value >>= kBitsOfByte;
  }
  return result;
}

int64_t ShardColumn::BytesLittleToMinIntType(const std::vector<uint8_t> &bytes_array, const uint64_t &pos,
                                             const IntegerType &src_i_type, IntegerType *dst_i_type) {
  uint64_t u_temp = 0;
  for (uint64_t i = 0; i < (kUnsignedOne << static_cast<uint8_t>(src_i_type)); i++) {
    u_temp = (u_temp << kBitsOfByte) +
             bytes_array[pos + (kUnsignedOne << static_cast<uint8_t>(src_i_type)) - kUnsignedOne - i];
  }

  int64_t i_out;
  switch (src_i_type) {
    case kInt8Type: {
      i_out = (int8_t)(u_temp & std::numeric_limits<uint8_t>::max());
      break;
    }
    case kInt16Type: {
      i_out = (int16_t)(u_temp & std::numeric_limits<uint16_t>::max());
      break;
    }
    case kInt32Type: {
      i_out = (int32_t)(u_temp & std::numeric_limits<uint32_t>::max());
      break;
    }
    case kInt64Type: {
      i_out = (int64_t)(u_temp & std::numeric_limits<uint64_t>::max());
      break;
    }
    default: {
      i_out = 0;
    }
  }

  if (!dst_i_type) {
    return i_out;
  }

  if (i_out >= static_cast<int64_t>(std::numeric_limits<int8_t>::min()) &&
      i_out <= static_cast<int64_t>(std::numeric_limits<int8_t>::max())) {
    *dst_i_type = kInt8Type;
  } else if (i_out >= static_cast<int64_t>(std::numeric_limits<int16_t>::min()) &&
             i_out <= static_cast<int64_t>(std::numeric_limits<int16_t>::max())) {
    *dst_i_type = kInt16Type;
  } else if (i_out >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
             i_out <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    *dst_i_type = kInt32Type;
  } else {
    *dst_i_type = kInt64Type;
  }
  return i_out;
}
}  // namespace mindrecord
}  // namespace mindspore
