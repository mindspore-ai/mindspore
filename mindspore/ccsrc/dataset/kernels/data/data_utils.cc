/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "dataset/kernels/data/data_utils.h"
#include <vector>
#include "dataset/core/constants.h"
#include "dataset/core/tensor.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/core/data_type.h"
#include "dataset/core/pybind_support.h"

namespace mindspore {
namespace dataset {
Status OneHotEncodingUnsigned(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                              dsize_t num_classes, int64_t index) {
  uint64_t class_idx;
  if (input->Rank() == 0) {
    RETURN_IF_NOT_OK(input->GetItemAt<uint64_t>(&class_idx, {}));
  } else {
    RETURN_IF_NOT_OK(input->GetItemAt<uint64_t>(&class_idx, {index}));
  }
  if (class_idx >= static_cast<uint64_t>(num_classes)) {
    RETURN_STATUS_UNEXPECTED("One_hot index values are not in range");
  }
  if (input->type() == DataType::DE_UINT64) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<uint64_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else if (input->type() == DataType::DE_UINT32) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<uint32_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else if (input->type() == DataType::DE_UINT16) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<uint16_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else if (input->type() == DataType::DE_UINT8) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<uint8_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else {
    RETURN_STATUS_UNEXPECTED("One hot unsigned only supports unsigned int as input.");
  }
  return Status::OK();
}

Status OneHotEncodingSigned(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, dsize_t num_classes,
                            int64_t index) {
  int64_t class_idx;
  if (input->Rank() == 0) {
    RETURN_IF_NOT_OK(input->GetItemAt<int64_t>(&class_idx, {}));
  } else {
    RETURN_IF_NOT_OK(input->GetItemAt<int64_t>(&class_idx, {index}));
  }
  if (class_idx >= static_cast<int64_t>(num_classes)) {
    RETURN_STATUS_UNEXPECTED("One_hot index values are not in range");
  }
  if (input->type() == DataType::DE_INT64) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<int64_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else if (input->type() == DataType::DE_INT32) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<int32_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else if (input->type() == DataType::DE_INT16) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<int16_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else if (input->type() == DataType::DE_INT8) {
    RETURN_IF_NOT_OK((*output)->SetItemAt<int8_t>({index, static_cast<dsize_t>(class_idx)}, 1));
  } else {
    RETURN_STATUS_UNEXPECTED("One hot signed only supports signed int as input.");
  }
  return Status::OK();
}

Status OneHotEncoding(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, dsize_t num_classes) {
  input->Squeeze();
  if (input->Rank() > 1) {  // We expect the input to be int he first dimension
    RETURN_STATUS_UNEXPECTED("One hot only supports scalars or 1D shape Tensors.");
  }
  if (!input->type().IsInt()) {
    RETURN_STATUS_UNEXPECTED("One hot does not support input of this type.");
  }
  try {
    dsize_t num_elements = 1;
    if (input->Rank() == 1) num_elements = input->shape()[0];
    TensorShape out_shape({num_elements, num_classes});
    std::shared_ptr<Tensor> out;
    RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, TensorImpl::kFlexible, out_shape, input->type()));
    RETURN_IF_NOT_OK(out->Zero());
    for (dsize_t i = 0; i < num_elements; ++i) {
      if (input->type().IsUnsignedInt()) {
        RETURN_IF_NOT_OK(OneHotEncodingUnsigned(input, &out, num_classes, i));
      } else {
        RETURN_IF_NOT_OK(OneHotEncodingSigned(input, &out, num_classes, i));
      }
    }
    out->Squeeze();
    *output = out;
    return Status::OK();
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Unexpected error in OneHotOp");
  }
}

template <typename FROM, typename TO>
void Cast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  auto in_itr = input->begin<FROM>();
  auto out_itr = (*output)->begin<TO>();
  auto out_end = (*output)->end<TO>();
  for (; out_itr != out_end; static_cast<void>(in_itr++), static_cast<void>(out_itr++))
    *out_itr = static_cast<TO>(*in_itr);
}

template <typename T>
void CastFrom(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  switch ((*output)->type().value()) {
    case DataType::DE_BOOL:
      Cast<T, bool>(input, output);
      break;
    case DataType::DE_INT8:
      Cast<T, int8_t>(input, output);
      break;
    case DataType::DE_UINT8:
      Cast<T, uint8_t>(input, output);
      break;
    case DataType::DE_INT16:
      Cast<T, int16_t>(input, output);
      break;
    case DataType::DE_UINT16:
      Cast<T, uint16_t>(input, output);
      break;
    case DataType::DE_INT32:
      Cast<T, int32_t>(input, output);
      break;
    case DataType::DE_UINT32:
      Cast<T, uint32_t>(input, output);
      break;
    case DataType::DE_INT64:
      Cast<T, int64_t>(input, output);
      break;
    case DataType::DE_UINT64:
      Cast<T, uint64_t>(input, output);
      break;
    case DataType::DE_FLOAT16:
      Cast<T, float16>(input, output);
      break;
    case DataType::DE_FLOAT32:
      Cast<T, float>(input, output);
      break;
    case DataType::DE_FLOAT64:
      Cast<T, double>(input, output);
      break;
    case DataType::DE_UNKNOWN:
      MS_LOG(ERROR) << "Unknown data type.";
      break;
  }
}

// Type cast operator
Status TypeCast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const DataType &data_type) {
  RETURN_IF_NOT_OK(Tensor::CreateTensor(output, TensorImpl::kFlexible, input->shape(), data_type));

  static_cast<void>((*output)->StartAddr());
  switch (input->type().value()) {
    case DataType::DE_BOOL:
      CastFrom<bool>(input, output);
      break;
    case DataType::DE_INT8:
      CastFrom<int8_t>(input, output);
      break;
    case DataType::DE_UINT8:
      CastFrom<uint8_t>(input, output);
      break;
    case DataType::DE_INT16:
      CastFrom<int16_t>(input, output);
      break;
    case DataType::DE_UINT16:
      CastFrom<uint16_t>(input, output);
      break;
    case DataType::DE_INT32:
      CastFrom<int32_t>(input, output);
      break;
    case DataType::DE_UINT32:
      CastFrom<uint32_t>(input, output);
      break;
    case DataType::DE_INT64:
      CastFrom<int64_t>(input, output);
      break;
    case DataType::DE_UINT64:
      CastFrom<uint64_t>(input, output);
      break;
    case DataType::DE_FLOAT16:
      CastFrom<float16>(input, output);
      break;
    case DataType::DE_FLOAT32:
      CastFrom<float>(input, output);
      break;
    case DataType::DE_FLOAT64:
      CastFrom<double>(input, output);
      break;
    case DataType::DE_UNKNOWN:
      // sanity check, unreachable code.
      RETURN_STATUS_UNEXPECTED("TypeCast does not support input of this type.");
  }
  return Status::OK();
}

Status ToFloat16(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // initiate new tensor for type cast
  DataType new_type = DataType("float16");
  RETURN_IF_NOT_OK(Tensor::CreateTensor(output, TensorImpl::kFlexible, input->shape(), new_type));
  static_cast<void>((*output)->StartAddr());

  auto in_itr = input->begin<float>();
  auto out_itr = (*output)->begin<float16>();
  auto out_end = (*output)->end<float16>();
  for (; out_itr != out_end; in_itr++, out_itr++) *out_itr = Eigen::half(*in_itr);

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
