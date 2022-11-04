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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DATA_PREPROCESSOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DATA_PREPROCESSOR_H_

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <numeric>
#include "common/file_util.h"
#include "common/data_transpose_utils.h"
#include "include/errorcode.h"
#include "common/op_enum.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/anf.h"
#include "src/mapper_config_parser.h"
#include "opencv2/core/mat.hpp"
#include "common/check_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace dpico {
class DataPreprocessor {
 public:
  static DataPreprocessor *GetInstance();
  int Run(const api::AnfNodePtrList &inputs);
  const std::string &GetPreprocessedDataDir() const { return preprocessed_data_dir_; }
  size_t GetBatchSize() const { return batch_size_; }

 private:
  DataPreprocessor() = default;
  ~DataPreprocessor() = default;
  int ModifyDynamicInputShape(std::vector<int64_t> *input_shape);
  int GetOutputBinDir(const std::string &op_name, std::string *output_bin_dir);
  int WriteCvMatToBin(const cv::Mat &image, const std::string &op_name);
  int GenerateInputBinFromTxt(const std::string &raw_data_path, const std::string &op_name,
                              const std::vector<int64_t> &op_shape, TypeId type_id);
  int GenerateInputBinFromImages(const std::string &raw_data_path, const std::string &op_name,
                                 const std::vector<int64_t> &op_shape, const struct AippModule &aipp_module);

  template <typename T>
  int WriteVectorToBin(std::vector<T> *nums, const std::string &op_name) {
    if (nums == nullptr) {
      MS_LOG(ERROR) << "input vector is nullptr.";
      return RET_ERROR;
    }
    std::string generated_bin_dir;
    if (GetOutputBinDir(op_name, &generated_bin_dir) != RET_OK) {
      MS_LOG(ERROR) << "get output bin dir failed.";
      return RET_ERROR;
    }
    if (Mkdir(generated_bin_dir) != RET_OK) {
      MS_LOG(ERROR) << "mkdir failed. " << generated_bin_dir;
      return RET_ERROR;
    }
    std::string output_bin_path = generated_bin_dir + "/input.bin";
    if (WriteToBin(output_bin_path, reinterpret_cast<void *>(nums->data()), nums->size() * sizeof(T)) != RET_OK) {
      MS_LOG(ERROR) << "write to bin failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }

  template <typename T, typename U = T>
  int GenerateInputBin(const std::string &preprocessed_line, const std::vector<int64_t> &op_shape,
                       const std::string &op_name) {
    size_t shape_size = 1;
    for (size_t i = 0; i < op_shape.size(); i++) {
      if (op_shape.at(i) < 0) {
        MS_LOG(ERROR) << "dim val should be equal or greater than 0";
        return RET_ERROR;
      }
      if (SIZE_MUL_OVERFLOW(shape_size, static_cast<size_t>(op_shape.at(i)))) {
        MS_LOG(ERROR) << "size_t mul overflow.";
        return RET_ERROR;
      }
      shape_size *= static_cast<size_t>(op_shape.at(i));
    }
    std::stringstream ss(preprocessed_line);
    std::vector<T> nums;
    U num;
    while (ss >> num) {
      (void)nums.emplace_back(num);
    }
    if (nums.size() != shape_size) {
      MS_LOG(ERROR) << op_name << "'s input size " << nums.size() << " is not equal to origin input shape size "
                    << shape_size;
      return RET_ERROR;
    }
    if (op_shape.size() < kDims4) {
      MS_LOG(WARNING) << op_name << "'s input shape is " << op_shape.size()
                      << " dims, and its data will not be transposed.";
      if (WriteVectorToBin(&nums, op_name) != RET_OK) {
        MS_LOG(ERROR) << "write vector to bin failed.";
        return RET_ERROR;
      }
    } else if (op_shape.size() == kDims4) {
      std::vector<T> nhwc_nums(nums.size());
      std::vector<int32_t> shape;
      (void)std::transform(op_shape.begin(), op_shape.end(), std::back_inserter(shape),
                           [](const int64_t dim) { return static_cast<int32_t>(dim); });
      if (NCHW2NHWC<T>(nums.data(), nhwc_nums.data(), shape) != RET_OK) {
        MS_LOG(ERROR) << "nchw to nhwc failed. " << op_name;
        return RET_ERROR;
      }
      if (WriteVectorToBin<T>(&nhwc_nums, op_name) != RET_OK) {
        MS_LOG(ERROR) << "write vector to bin failed.";
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << op_name << "'s input shape is " << op_shape.size() << " dims, which is not supported for now.";
      return RET_ERROR;
    }
    return RET_OK;
  }

  std::string preprocessed_data_dir_;
  size_t batch_size_{0};
};
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DATA_PREPROCESSOR_H_
