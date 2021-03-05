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
#include "minddata/dataset/include/data_helper.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>

#include "minddata/dataset/util/json_helper.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/path.h"
#include "include/api/status.h"

namespace mindspore {
namespace dataset {
// Create a numbered json file from image folder
Status DataHelper::CreateAlbumIF(const std::vector<char> &in_dir, const std::vector<char> &out_dir) {
  auto jh = JsonHelper();
  return jh.CreateAlbum(CharToString(in_dir), CharToString(out_dir));
}

// A print method typically used for debugging
void DataHelper::Print(std::ostream &out) const {
  out << "  Data Helper"
      << "\n";
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<std::vector<char>> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), VectorCharToString(value), CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<bool> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<int8_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<uint8_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<int16_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<uint16_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<int32_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<uint32_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<int64_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<uint64_t> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<float> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateArrayIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<double> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateArray(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key,
                                 const std::vector<char> &value, const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), CharToString(value), CharToString(out_file));
}

Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const bool &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int8_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint8_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int16_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint16_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int32_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint32_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const int64_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const uint64_t &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}
Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const float &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::UpdateValueIF(const std::vector<char> &in_file, const std::vector<char> &key, const double &value,
                                 const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.UpdateValue(CharToString(in_file), CharToString(key), value, CharToString(out_file));
}

Status DataHelper::RemoveKeyIF(const std::vector<char> &in_file, const std::vector<char> &key,
                               const std::vector<char> &out_file) {
  auto jh = JsonHelper();
  return jh.RemoveKey(CharToString(in_file), CharToString(key), CharToString(out_file));
}

size_t DataHelper::DumpData(const unsigned char *tensor_addr, const size_t &tensor_size, void *addr,
                            const size_t &buffer_size) {
  auto jh = JsonHelper();
  return jh.DumpData(tensor_addr, tensor_size, addr, buffer_size);
}

}  // namespace dataset
}  // namespace mindspore
