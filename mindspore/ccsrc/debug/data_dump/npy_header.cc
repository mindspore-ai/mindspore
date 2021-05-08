/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "debug/data_dump/npy_header.h"

#include <unordered_map>
#include <utility>
#include <sstream>
#include <cmath>

#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/core/ir/dtype.h"

namespace mindspore {
namespace {
// npy file header start information
const char kMagicPrefix[] = "\x93NUMPY";
// magical length include kMagicPrefix length and version length
const size_t kMagicLen = 8;
const size_t kArrayAlign = 64;

// first: header_length_type, second: encoding_type
// header_length_type: 1 represents 2 bytes; 2 and 3 represents 4 bytes
// encoding_type: 1 and 2 represents 'latin1'; 3 represents 'utf8'
using version_type = std::pair<int, int>;

// data type description
// byteorder char: '<' is little endian; '>' is big endian; '|' is ignore(no change to byte order)
// type char: 'b' represents bool; 'u' represents uint; 'i' represents int; 'f' represents float
struct DtypeDescr {
  char byteorder;
  char type;
  size_t length;

  std::string str() const;
};

// npy file header description, includes data type description, fortran_order and array shape
// fortran_order: true represents the array data Fortran-contiguous; false represents the array data C-contiguity
struct NpyHeader {
 public:
  DtypeDescr dtype_descr;
  bool fortran_order;
  ShapeVector shape;

  std::string str() const;

 private:
  std::string fortran_order_to_str() const;
  std::string shape_to_str() const;
};

std::string DtypeDescr::str() const {
  std::ostringstream buffer;
  buffer << "\'" << byteorder << type << length << "\'";
  return buffer.str();
}

std::string NpyHeader::str() const {
  const std::string first_field = "'descr': ";
  const std::string second_field = "'fortran_order': ";
  const std::string third_field = "'shape': ";
  std::ostringstream buffer;
  buffer << "{" << first_field << dtype_descr.str() << ", " << second_field << fortran_order_to_str() << ", "
         << third_field << shape_to_str() << ", }";
  return buffer.str();
}

std::string NpyHeader::fortran_order_to_str() const { return fortran_order ? "True" : "False"; }

std::string NpyHeader::shape_to_str() const {
  std::ostringstream buffer;
  buffer << "(";
  for (const auto i : shape) {
    buffer << std::to_string(i) << ",";
  }
  buffer << ")";
  return buffer.str();
}
}  // namespace

void int_to_byte(size_t number, char *byte, size_t length) {
  for (size_t i = 0; i < length; i++) {
    byte[i] = (number >> (i * 8)) & 0xff;
  }
}

std::string GenerateNpyHeader(const ShapeVector &shape, TypeId type_id, bool fortran_order) {
  static std::unordered_map<TypeId, DtypeDescr> type_desc_map = {
    {kNumberTypeBool, DtypeDescr{'|', 'b', 1}},    {kNumberTypeInt8, DtypeDescr{'|', 'i', 1}},
    {kNumberTypeInt16, DtypeDescr{'<', 'i', 2}},   {kNumberTypeInt32, DtypeDescr{'<', 'i', 4}},
    {kNumberTypeInt64, DtypeDescr{'<', 'i', 8}},   {kNumberTypeUInt8, DtypeDescr{'|', 'u', 1}},
    {kNumberTypeUInt16, DtypeDescr{'<', 'u', 2}},  {kNumberTypeUInt32, DtypeDescr{'<', 'u', 4}},
    {kNumberTypeUInt64, DtypeDescr{'<', 'u', 8}},  {kNumberTypeFloat16, DtypeDescr{'<', 'f', 2}},
    {kNumberTypeFloat32, DtypeDescr{'<', 'f', 4}}, {kNumberTypeFloat64, DtypeDescr{'<', 'f', 8}},
  };
  auto type_desc = type_desc_map.find(type_id);
  if (type_desc == type_desc_map.end()) {
    MS_LOG(EXCEPTION) << "Not support dump the " << TypeIdToType(type_id)->ToString() << " data to npy file.";
  }

  NpyHeader npy_header{type_desc->second, fortran_order, shape};
  std::string header_str = npy_header.str();
  size_t header_len = header_str.length();
  version_type version{1, 0};
  size_t total_len = kMagicLen + 2 + header_len + 1;
  if (total_len > std::pow(2, 16)) {
    version = {2, 0};
    total_len = kMagicLen + 4 + header_len + 1;
  }
  std::ostringstream out;
  out << kMagicPrefix;
  out.put(version.first);
  out.put(version.second);

  size_t pad_len = kArrayAlign - total_len % kArrayAlign;
  size_t padding_header_len = header_len + pad_len + 1;
  if (version == version_type{1, 0}) {
    char length_byte[2];
    int_to_byte(padding_header_len, length_byte, 2);
    out.write(length_byte, 2);
  } else {
    char length_byte[4];
    int_to_byte(padding_header_len, length_byte, 4);
    out.write(length_byte, 4);
  }

  std::string padding(pad_len, ' ');
  out << header_str << padding << "\n";
  return out.str();
}
}  // namespace mindspore
