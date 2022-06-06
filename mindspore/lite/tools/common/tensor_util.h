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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_TENSOR_UTIL_H_
#define MINDSPORE_LITE_TOOLS_COMMON_TENSOR_UTIL_H_

#include <cmath>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <utility>
#include <string>
#include <vector>
#include <random>
#include <cfloat>
#include "schema/inner/model_generated.h"
#include "src/common/log_util.h"
#include "src/common/log_adapter.h"
#include "ir/dtype/type_id.h"
#include "ir/tensor.h"
#include "src/common/utils.h"
#include "tools/common/statistic_utils.h"
#include "src/tensor.h"
#include "include/api/model.h"

namespace mindspore {
namespace lite {
using schema::CNodeT;
using schema::Format;
using schema::FusedBatchNormT;
using schema::MetaGraphT;
using schema::QuantParamT;
using schema::TensorT;

std::unique_ptr<QuantParamT> GetTensorQuantParam(const std::unique_ptr<TensorT> &tensor);

tensor::TensorPtr CreateTensorInfo(const void *data, size_t data_size, const std::vector<int64_t> &shape,
                                   TypeId data_type);

AbstractBasePtr CreateTensorAbstract(const std::vector<int64_t> &shape, TypeId data_type);

int SetParameterAbstractAndParam(const ParameterPtr &parameter, const void *data, size_t data_size,
                                 const std::vector<int64_t> &shape, TypeId data_type);

int SetTensorData(const tensor::TensorPtr &tensor_info, const void *data, size_t data_size);

std::unique_ptr<schema::TensorT> CreateTensorTFromTensorInfo(const tensor::TensorPtr &tensor_info,
                                                             const std::string &tensor_name = "");

int UpdateTensorTFromTensorInfo(const tensor::TensorPtr &src_tensor, std::unique_ptr<schema::TensorT> *dst_tensor);

int InitParameterFromTensorInfo(const ParameterPtr &param_node, const tensor::TensorPtr &tensor_info);

size_t GetElementSize(const TensorT &tensor);

size_t GetElementSize(const TypeId &dataType);

size_t GetShapeSize(const TensorT &tensor);

size_t GetShapeSize(const std::vector<int32_t> &shape);

std::unique_ptr<TensorT> CopyTensorDefT(const std::unique_ptr<TensorT> &);

size_t GetRefCount(schema::MetaGraphT *graphT, uint32_t tensorIdx);

std::unique_ptr<schema::QuantParamT> CopyQuantParamT(const std::unique_ptr<schema::QuantParamT> &srcQuantParam);

int GenerateRandomData(mindspore::MSTensor *tensors);

int GenerateRandomData(size_t size, void *data, int data_type);

template <typename T, typename Distribution>
void FillInputData(size_t size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  MS_ASSERT(data != nullptr);
  size_t elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&]() { return static_cast<T>(distribution(random_engine)); });
}

struct CheckTensor {
  CheckTensor(const std::string &tensor_name, const std::vector<size_t> &shape, const std::vector<float> &data,
              const std::vector<std::string> &strings_data = {""}) {
    this->tensor_name = tensor_name;
    this->shape = shape;
    this->data = data;
    this->strings_data = strings_data;
  }
  std::string tensor_name;
  std::vector<size_t> shape;
  std::vector<float> data;
  std::vector<std::string> strings_data;
};

// tensorData need to be converter first
template <typename T>
float CompareDataByCosineDistance(const std::shared_ptr<mindspore::Model> &origin_model,
                                  const std::shared_ptr<mindspore::Model> &quant_model) {
  CHECK_NULL_RETURN(origin_model);
  CHECK_NULL_RETURN(quant_model);
  if (origin_model->GetOutputs().empty() || quant_model->GetOutputs().empty()) {
    MS_LOG(ERROR) << "calib or out tenor is empty.";
    return RET_ERROR;
  }
  float total_cos = 0;
  auto calib_tensors = origin_model->GetOutputs();
  for (const auto &calib_tensor : calib_tensors) {
    size_t error_count = 0;
    float mean_error = 0;
    auto calib_data = reinterpret_cast<const T *>(calib_tensor.Data().get());
    auto out_tensor = quant_model->GetOutputByTensorName(calib_tensor.Name());
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "Cant find " << calib_tensor.Name() << " in out_tensors";
      return RET_ERROR;
    }
    auto out_data = reinterpret_cast<const T *>(out_tensor.Data().get());
    auto cos = mindspore::lite::GetCosSimilarity<T>(calib_data, out_data, static_cast<size_t>(out_tensor.ElementNum()));
    total_cos += cos;
    MS_LOG(INFO) << "tensor_name:" << calib_tensor.Name() << " cos_sim: " << mean_error
                 << " error_count:" << error_count;
  }
  return total_cos / calib_tensors.size();
}

template <typename T>
float CompareData(const std::shared_ptr<mindspore::Model> &origin_model,
                  const std::shared_ptr<mindspore::Model> &quant_model) {
  CHECK_NULL_RETURN(origin_model);
  CHECK_NULL_RETURN(quant_model);
  if (origin_model->GetOutputs().empty() || quant_model->GetOutputs().empty()) {
    MS_LOG(ERROR) << "calib or out tenor is empty.";
    return RET_ERROR;
  }
  float total_meam_error = 0;
  auto calib_tensors = origin_model->GetOutputs();
  for (const auto &calib_tensor : calib_tensors) {
    size_t error_count = 0;
    float mean_error = 0;
    auto calib_data = reinterpret_cast<const T *>(calib_tensor.Data().get());
    auto out_tensor = quant_model->GetOutputByTensorName(calib_tensor.Name());
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "Cant find " << calib_tensor.Name() << " in out_tensors";
      return RET_ERROR;
    }
    auto out_data = reinterpret_cast<const T *>(out_tensor.Data().get());
    for (int j = 0; j < calib_tensor.ElementNum(); j++) {
      if (std::is_same<T, float>::value && (std::isnan(out_data[j]) || std::isinf(out_data[j]))) {
        MS_LOG(ERROR) << "Output tensor has nan or inf data, compare fail";
        return RET_ERROR;
      }
      constexpr float relativeTolerance = 1e-5;
      constexpr float absoluteTolerance = 1e-8;
      auto tolerance = absoluteTolerance + relativeTolerance * fabs(calib_data[j]);
      auto absolute_error = std::fabs(out_data[j] - calib_data[j]);
      if (absolute_error > tolerance) {
        if (fabs(calib_data[j] - 0.0f) < FLT_EPSILON) {
          if (absolute_error > 1e-5) {
            mean_error += absolute_error;
            error_count++;
          } else {
            continue;
          }
        } else {
          // just assume that atol = rtol
          mean_error += absolute_error / (fabs(calib_data[j]) + FLT_MIN);
          error_count++;
        }
      }
    }
    if (mean_error > 0.0f && error_count > 0) {
      mean_error /= error_count;
    }
    total_meam_error += std::abs(mean_error);
    MS_LOG(INFO) << "tensor_name:" << calib_tensor.Name() << " mean_error: " << mean_error
                 << " error_count:" << error_count;
  }
  return total_meam_error / calib_tensors.size();
}
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_TENSOR_UTIL_H_
