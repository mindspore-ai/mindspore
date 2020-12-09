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
#ifndef MINDSPORE_INCLUDE_API_TYPES_H
#define MINDSPORE_INCLUDE_API_TYPES_H

#include <string>
#include <vector>
#include <memory>

#define MS_API __attribute__((visibility("default")))

namespace mindspore {
namespace api {
enum ModelType {
  kMindIR = 0,
  kAIR = 1,
  kOM = 2,
  kONNX = 3,
  // insert new data type here
  kUnknownType = 0xFFFFFFFF
};

enum DataType {
  kMsUnknown = 0,
  kMsBool = 1,
  kMsInt8 = 2,
  kMsInt16 = 3,
  kMsInt32 = 4,
  kMsInt64 = 5,
  kMsUint8 = 6,
  kMsUint16 = 7,
  kMsUint32 = 8,
  kMsUint64 = 9,
  kMsFloat16 = 10,
  kMsFloat32 = 11,
  kMsFloat64 = 12,
  // insert new data type here
  kInvalidDataType = 0xFFFFFFFF
};

class MS_API Tensor {
 public:
  Tensor();
  Tensor(const std::string &name, DataType type, const std::vector<int64_t> &shape, const void *data, size_t data_len);
  ~Tensor();

  const std::string &Name() const;
  void SetName(const std::string &name);

  api::DataType DataType() const;
  void SetDataType(api::DataType type);

  const std::vector<int64_t> &Shape() const;
  void SetShape(const std::vector<int64_t> &shape);

  const void *Data() const;
  void *MutableData();
  size_t DataSize() const;

  bool ResizeData(size_t data_len);
  bool SetData(const void *data, size_t data_len);

  int64_t ElementNum() const;
  static int GetTypeSize(api::DataType type);
  Tensor Clone() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

class MS_API Buffer {
 public:
  Buffer();
  Buffer(const void *data, size_t data_len);
  ~Buffer();

  const void *Data() const;
  void *MutableData();
  size_t DataSize() const;

  bool ResizeData(size_t data_len);
  bool SetData(const void *data, size_t data_len);

  Buffer Clone() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

extern MS_API const char *kDeviceTypeAscend310;
extern MS_API const char *kDeviceTypeAscend910;

constexpr auto kModelOptionDumpCfgPath = "mindspore.option.dump_config_file_path";
constexpr auto kModelOptionInsertOpCfgPath = "mindspore.option.insert_op_config_file_path";  // aipp config file
constexpr auto kModelOptionInputFormat = "mindspore.option.input_format";                    // nchw or nhwc
// Mandatory while dynamic batch: e.g. "input_op_name1: n1,c2,h3,w4;input_op_name2: n4,c3,h2,w1"
constexpr auto kModelOptionInputShape = "mindspore.option.input_shape";
constexpr auto kModelOptionOutputType = "mindspore.option.output_type";  // "FP32", "UINT8" or "FP16", default as "FP32"
constexpr auto kModelOptionPrecisionMode = "mindspore.option.precision_mode";
// "force_fp16", "allow_fp32_to_fp16", "must_keep_origin_dtype" or "allow_mix_precision", default as "force_fp16"
constexpr auto kModelOptionOpSelectImplMode = "mindspore.option.op_select_impl_mode";
// "high_precision" or "high_performance", default as "high_performance"
}  // namespace api
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_TYPES_H
