/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_UTILS_H
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_UTILS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "ir/tensor.h"
namespace mindspore {

constexpr auto csvHeaderComm = "Op Type,Op Name,Task ID,Stream ID,Timestamp,IO,Slot,Data Size,Data Type,Shape";
const std::unordered_map<std::string, std::string> header_map = {{"max", "Max Value"},
                                                                 {"min", "Min Value"},
                                                                 {"avg", "Avg Value"},
                                                                 {"count", "Count"},
                                                                 {"negative zero count", "Negative Zero Count"},
                                                                 {"positive zero count", "Positive Zero Count"},
                                                                 {"nan count", "NaN Count"},
                                                                 {"negative inf count", "Negative Inf Count"},
                                                                 {"positive inf count", "Positive Inf Count"},
                                                                 {"zero count", "Zero Count"},
                                                                 {"l2norm", "L2Norm Value"},
                                                                 {"md5", "MD5"},
                                                                 {"sha1", "SHA1"}};

class CsvHeaderUtil {
 public:
  static CsvHeaderUtil &GetInstance() {
    static CsvHeaderUtil instance;
    return instance;
  }
  void SetStatCsvHeader(std::vector<std::string> headers) {
    csv_header.assign(csvHeaderComm);
    for (const auto &str : headers) {
      // JsonDumpParser guarantee headers are valid.
      csv_header = csv_header + "," + header_map.at(str);
    }
    csv_header += '\n';
  }
  std::string GetStatCsvHeader() { return csv_header; }

 private:
  CsvHeaderUtil() : csv_header("") {}
  std::string csv_header;
};

bool CheckStoul(size_t *const output_digit, const std::string &input_str);

string ShapeToString(const ShapeVector &shape);
namespace datadump {
std::uint32_t GetRankID();
bool StartsWith(const std::string &, const std::string &);
bool EndsWith(const std::string &, const std::string &);
void SaveTensor2NPY(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr);
}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_UTILS_H
