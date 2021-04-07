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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_COLUMN_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_COLUMN_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_header.h"

namespace mindspore {
namespace mindrecord {
const uint64_t kUnsignedOne = 1;
const uint64_t kBitsOfByte = 8;
const uint64_t kDataTypeBits = 2;
const uint64_t kNumDataOfByte = 4;
const uint64_t kBytesOfColumnLen = 4;
const uint64_t kDataTypeBitMask = 3;
const uint64_t kDataTypes = 6;

enum IntegerType { kInt8Type = 0, kInt16Type, kInt32Type, kInt64Type };

enum ColumnCategory { ColumnInRaw, ColumnInBlob, ColumnNotFound };

enum ColumnDataType {
  ColumnBytes = 0,
  ColumnString = 1,
  ColumnInt32 = 2,
  ColumnInt64 = 3,
  ColumnFloat32 = 4,
  ColumnFloat64 = 5,
  ColumnNoDataType = 6
};

const uint32_t ColumnDataTypeSize[kDataTypes] = {1, 1, 4, 8, 4, 8};

const std::vector<std::string> ColumnDataTypeNameNormalized = {"uint8", "string",  "int32",
                                                               "int64", "float32", "float64"};

const std::unordered_map<std::string, ColumnDataType> ColumnDataTypeMap = {
  {"bytes", ColumnBytes}, {"string", ColumnString},   {"int32", ColumnInt32},
  {"int64", ColumnInt64}, {"float32", ColumnFloat32}, {"float64", ColumnFloat64}};

class __attribute__((visibility("default"))) ShardColumn {
 public:
  explicit ShardColumn(const std::shared_ptr<ShardHeader> &shard_header, bool compress_integer = true);
  explicit ShardColumn(const json &schema_json, bool compress_integer = true);

  ~ShardColumn() = default;

  /// \brief get column value by column name
  MSRStatus GetColumnValueByName(const std::string &column_name, const std::vector<uint8_t> &columns_blob,
                                 const json &columns_json, const unsigned char **data,
                                 std::unique_ptr<unsigned char[]> *data_ptr, uint64_t *const n_bytes,
                                 ColumnDataType *column_data_type, uint64_t *column_data_type_size,
                                 std::vector<int64_t> *column_shape);

  /// \brief compress blob
  std::vector<uint8_t> CompressBlob(const std::vector<uint8_t> &blob, int64_t *compression_size);

  /// \brief check if blob compressed
  bool CheckCompressBlob() const { return has_compress_blob_; }

  /// \brief getter
  uint64_t GetNumBlobColumn() const { return num_blob_column_; }

  /// \brief getter
  std::vector<std::string> GetColumnName() { return column_name_; }

  /// \brief getter
  std::vector<ColumnDataType> GeColumnDataType() { return column_data_type_; }

  /// \brief getter
  std::vector<std::vector<int64_t>> GetColumnShape() { return column_shape_; }

  /// \brief get column value from blob
  MSRStatus GetColumnFromBlob(const std::string &column_name, const std::vector<uint8_t> &columns_blob,
                              const unsigned char **data, std::unique_ptr<unsigned char[]> *data_ptr,
                              uint64_t *const n_bytes);

  /// \brief get column type
  std::pair<MSRStatus, ColumnCategory> GetColumnTypeByName(const std::string &column_name,
                                                           ColumnDataType *column_data_type,
                                                           uint64_t *column_data_type_size,
                                                           std::vector<int64_t> *column_shape);

  /// \brief get column value from json
  MSRStatus GetColumnFromJson(const std::string &column_name, const json &columns_json,
                              std::unique_ptr<unsigned char[]> *data_ptr, uint64_t *n_bytes);

 private:
  /// \brief initialization
  void Init(const json &schema_json, bool compress_integer = true);

  /// \brief get float value from json
  template <typename T>
  MSRStatus GetFloat(std::unique_ptr<unsigned char[]> *data_ptr, const json &json_column_value, bool use_double);

  /// \brief get integer value from json
  template <typename T>
  MSRStatus GetInt(std::unique_ptr<unsigned char[]> *data_ptr, const json &json_column_value);

  /// \brief get column offset address and size from blob
  MSRStatus GetColumnAddressInBlock(const uint64_t &column_id, const std::vector<uint8_t> &columns_blob,
                                    uint64_t *num_bytes, uint64_t *shift_idx);

  /// \brief check if column name is available
  ColumnCategory CheckColumnName(const std::string &column_name);

  /// \brief compress integer column
  static vector<uint8_t> CompressInt(const vector<uint8_t> &src_bytes, const IntegerType &int_type);

  /// \brief uncompress integer array column
  template <typename T>
  static MSRStatus UncompressInt(const uint64_t &column_id, std::unique_ptr<unsigned char[]> *const data_ptr,
                                 const std::vector<uint8_t> &columns_blob, uint64_t *num_bytes, uint64_t shift_idx);

  /// \brief convert big-endian bytes to unsigned int
  /// \param bytes_array bytes array
  /// \param pos shift address in bytes array
  /// \param i_type integer type
  /// \return unsigned int
  static uint64_t BytesBigToUInt64(const std::vector<uint8_t> &bytes_array, const uint64_t &pos,
                                   const IntegerType &i_type);

  /// \brief convert unsigned int to big-endian bytes
  /// \param value integer value
  /// \param i_type integer type
  /// \return bytes
  static std::vector<uint8_t> UIntToBytesBig(uint64_t value, const IntegerType &i_type);

  /// \brief convert unsigned int to little-endian bytes
  /// \param value integer value
  /// \param i_type integer type
  /// \return bytes
  static std::vector<uint8_t> UIntToBytesLittle(uint64_t value, const IntegerType &i_type);

  /// \brief convert unsigned int to little-endian bytes
  /// \param bytes_array bytes array
  /// \param pos shift address in bytes array
  /// \param src_i_type source integer typ0e
  /// \param dst_i_type (output), destination integer type
  /// \return integer
  static int64_t BytesLittleToMinIntType(const std::vector<uint8_t> &bytes_array, const uint64_t &pos,
                                         const IntegerType &src_i_type, IntegerType *dst_i_type = nullptr);

 private:
  std::vector<std::string> column_name_;                      // column name list
  std::vector<ColumnDataType> column_data_type_;              // column data type list
  std::vector<std::vector<int64_t>> column_shape_;            // column shape list
  std::unordered_map<string, uint64_t> column_name_id_;       // column name id map
  std::vector<std::string> blob_column_;                      // blob column list
  std::unordered_map<std::string, uint64_t> blob_column_id_;  // blob column name id map
  bool has_compress_blob_;                                    // if has compress blob
  uint64_t num_blob_column_;                                  // number of blob columns
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_COLUMN_H_
