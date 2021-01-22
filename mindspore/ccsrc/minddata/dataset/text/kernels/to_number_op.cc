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

#include "minddata/dataset/text/kernels/to_number_op.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

ToNumberOp::ToNumberOp(const DataType &cast_to_type) : cast_to_type_(cast_to_type) {}

ToNumberOp::ToNumberOp(const std::string &cast_to_type) : cast_to_type_(DataType(cast_to_type)) {}

Status ToNumberOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "ToNumber: input should be string datatype.");

  switch (cast_to_type_.value()) {
    case DataType::DE_INT8:
      RETURN_IF_NOT_OK(ToSignedIntegral<int8_t>(input, output));
      break;
    case DataType::DE_INT16:
      RETURN_IF_NOT_OK(ToSignedIntegral<int16_t>(input, output));
      break;
    case DataType::DE_INT32:
      RETURN_IF_NOT_OK(ToSignedIntegral<int32_t>(input, output));
      break;
    case DataType::DE_INT64:
      RETURN_IF_NOT_OK(ToSignedIntegral<int64_t>(input, output));
      break;
    case DataType::DE_UINT8:
      RETURN_IF_NOT_OK(ToUnsignedIntegral<uint8_t>(input, output));
      break;
    case DataType::DE_UINT16:
      RETURN_IF_NOT_OK(ToUnsignedIntegral<uint16_t>(input, output));
      break;
    case DataType::DE_UINT32:
      RETURN_IF_NOT_OK(ToUnsignedIntegral<uint32_t>(input, output));
      break;
    case DataType::DE_UINT64:
      RETURN_IF_NOT_OK(ToUnsignedIntegral<uint64_t>(input, output));
      break;
    case DataType::DE_FLOAT16:
      RETURN_IF_NOT_OK(this->ToFloat16(input, output));
      break;
    case DataType::DE_FLOAT32:
      RETURN_IF_NOT_OK(ToFloat(input, output));
      break;
    case DataType::DE_FLOAT64:
      RETURN_IF_NOT_OK(ToDouble(input, output));
      break;
    default:
      RETURN_STATUS_UNEXPECTED(
        "ToNumber: "
        "unsupported cast type: " +
        cast_to_type_.ToString());
  }

  return Status::OK();
}

void ToNumberOp::Print(std::ostream &out) const { out << "ToNumberOp: casting to " << '\n'; }

Status ToNumberOp::OutputShape(const std::vector<TensorShape> &input_shapes, std::vector<TensorShape> &output_shapes) {
  (void)std::copy(input_shapes.begin(), input_shapes.end(), std::back_inserter(output_shapes));
  return Status::OK();
}

template <typename T>
Status ToNumberOp::ToSignedIntegral(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::vector<T> casted;

  for (auto it = input->begin<std::string_view>(); it != input->end<std::string_view>(); ++it) {
    bool is_cast_out_of_range = false;
    int64_t result = 0;

    try {
      result = std::stoll(std::string(*it));
    } catch (const std::out_of_range &) {
      is_cast_out_of_range = true;
    } catch (const std::invalid_argument &) {
      RETURN_STATUS_UNEXPECTED(
        "ToNumber: "
        "it is invalid to convert \"" +
        std::string(*it) + "\" to a number.");
    }

    if (result > std::numeric_limits<T>::max() || result < std::numeric_limits<T>::min() || is_cast_out_of_range) {
      std::string error_message =
        "ToNumber: "
        "string input " +
        std::string(*it) + " will be out of bounds if cast to " + cast_to_type_.ToString() + ". The valid range is: [" +
        std::to_string(std::numeric_limits<T>::min()) + ", " + std::to_string(std::numeric_limits<T>::max()) + "].";

      RETURN_STATUS_UNEXPECTED(error_message);
    }

    T casted_result = static_cast<T>(result);
    casted.push_back(casted_result);
  }

  RETURN_IF_NOT_OK(Tensor::CreateFromVector(casted, input->shape(), output));
  return Status::OK();
}

template <typename T>
Status ToNumberOp::ToUnsignedIntegral(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::vector<T> casted;

  for (auto it = input->begin<std::string_view>(); it != input->end<std::string_view>(); ++it) {
    bool is_cast_out_of_range = false;
    uint64_t result = 0;

    // If there is a - at the start of the string, it is considered by us to
    // be out of bounds. If the - is somewhere else in the string, it is
    // deemed invalid by std::stoull and will throw std::invalid_argument
    for (int i = 0; i < (*it).size(); i++) {
      if ((*it)[i] == '-') {
        is_cast_out_of_range = true;
        break;
      }
    }

    try {
      result = std::stoull(std::string(*it));
    } catch (const std::out_of_range &) {
      is_cast_out_of_range = true;
    } catch (const std::invalid_argument &) {
      RETURN_STATUS_UNEXPECTED(
        "ToNumber: "
        "It is invalid to convert \"" +
        std::string(*it) + "\" to an unsigned integer.");
    }

    if (result > std::numeric_limits<T>::max() || result < std::numeric_limits<T>::min() || is_cast_out_of_range) {
      std::string error_message =
        "ToNumber: "
        "string input " +
        std::string(*it) + " will be out of bounds if cast to " + cast_to_type_.ToString() + ". The valid range is: [" +
        std::to_string(std::numeric_limits<T>::min()) + ", " + std::to_string(std::numeric_limits<T>::max()) + "].";

      RETURN_STATUS_UNEXPECTED(error_message);
    }

    T casted_result = static_cast<T>(result);
    casted.push_back(casted_result);
  }

  RETURN_IF_NOT_OK(Tensor::CreateFromVector(casted, input->shape(), output));
  return Status::OK();
}

Status ToNumberOp::ToFloat16(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // special case, float16 does not exist in c++, no native support for
  // casting, so cast to float first then use this method, which use Eigen.
  std::shared_ptr<Tensor> temp;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), DataType("float32"), &temp));
  RETURN_IF_NOT_OK(ToFloat(input, &temp));
  RETURN_IF_NOT_OK(mindspore::dataset::ToFloat16(temp, output));
  return Status::OK();
}

Status ToNumberOp::ToFloat(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::vector<float> casted;

  for (auto it = input->begin<std::string_view>(); it != input->end<std::string_view>(); ++it) {
    bool is_cast_out_of_range = false;
    float result = 0;

    try {
      result = std::stof(std::string(*it));
    } catch (const std::out_of_range &) {
      is_cast_out_of_range = true;
    } catch (const std::invalid_argument &) {
      RETURN_STATUS_UNEXPECTED(
        "ToNumber: "
        "it is invalid to convert \"" +
        std::string(*it) + "\" to an unsigned integer.");
    }

    if (result > std::numeric_limits<float>::max() || result < std::numeric_limits<float>::lowest() ||
        is_cast_out_of_range) {
      std::string error_message =
        "ToNumber: "
        "string input " +
        std::string(*it) + " will be out of bounds if cast to " + cast_to_type_.ToString() + ". The valid range is: [" +
        std::to_string(std::numeric_limits<float>::lowest()) + ", " +
        std::to_string(std::numeric_limits<float>::max()) + "].";

      RETURN_STATUS_UNEXPECTED(error_message);
    }

    float casted_result = static_cast<float>(result);
    casted.push_back(casted_result);
  }

  RETURN_IF_NOT_OK(Tensor::CreateFromVector(casted, input->shape(), output));
  return Status::OK();
}

Status ToNumberOp::ToDouble(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::vector<double> casted;

  for (auto it = input->begin<std::string_view>(); it != input->end<std::string_view>(); ++it) {
    bool is_cast_out_of_range = false;
    double result = 0;

    try {
      result = std::stod(std::string(*it));
    } catch (const std::out_of_range &) {
      is_cast_out_of_range = true;
    } catch (const std::invalid_argument &) {
      RETURN_STATUS_UNEXPECTED(
        "ToNumber: "
        "it is invalid to convert \"" +
        std::string(*it) + "\" to an unsigned integer.");
    }

    if (result > std::numeric_limits<double>::max() || result < std::numeric_limits<double>::lowest() ||
        is_cast_out_of_range) {
      std::string error_message =
        "ToNumber: "
        "string input " +
        std::string(*it) + " will be out of bounds if cast to " + cast_to_type_.ToString() + ". The valid range is: [" +
        std::to_string(std::numeric_limits<double>::lowest()) + ", " +
        std::to_string(std::numeric_limits<double>::max()) + "].";

      RETURN_STATUS_UNEXPECTED(error_message);
    }

    double casted_result = static_cast<double>(result);
    casted.push_back(casted_result);
  }

  RETURN_IF_NOT_OK(Tensor::CreateFromVector(casted, input->shape(), output));
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
