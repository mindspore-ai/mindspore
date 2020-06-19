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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "dataset/core/constants.h"
#include "dataset/core/data_type.h"
#include "dataset/core/pybind_support.h"
#include "dataset/core/tensor.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/kernels/data/type_cast_op.h"
#include "dataset/util/status.h"

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

Status Fill(const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, std::shared_ptr<Tensor> fill_value) {
  CHECK_FAIL_RETURN_UNEXPECTED(!((fill_value->type() == DataType::DE_STRING) && (input->type() != DataType::DE_STRING)),
                               "Types do not match");

  CHECK_FAIL_RETURN_UNEXPECTED(fill_value->shape() == TensorShape({}), "fill_value is not a scalar");

  std::shared_ptr<Tensor> out;

  const DataType &to = input->type();
  std::unique_ptr<TypeCastOp> op(new TypeCastOp(to));

  std::shared_ptr<Tensor> fill_output;
  RETURN_IF_NOT_OK(op->Compute(fill_value, &fill_output));

  RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, TensorImpl::kFlexible, input->shape(), input->type()));

  switch (input->type().value()) {
    case DataType::DE_BOOL: {
      bool value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<bool>(value);
      break;
    }
    case DataType::DE_INT8: {
      int8_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<int8_t>(value);
      break;
    }
    case DataType::DE_UINT8: {
      uint8_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<uint8_t>(value);
      break;
    }
    case DataType::DE_UINT16: {
      uint16_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<uint16_t>(value);
      break;
    }
    case DataType::DE_INT16: {
      int16_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<int16_t>(value);
      break;
    }
    case DataType::DE_UINT32: {
      uint32_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<uint32_t>(value);
      break;
    }
    case DataType::DE_INT32: {
      int32_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<int32_t>(value);
      break;
    }
    case DataType::DE_UINT64: {
      uint64_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<uint64_t>(value);
      break;
    }
    case DataType::DE_INT64: {
      int64_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<int64_t>(value);
      break;
    }
    case DataType::DE_FLOAT16: {
      int64_t value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<float>(value);
      break;
    }
    case DataType::DE_FLOAT32: {
      float value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<float>(value);
      break;
    }
    case DataType::DE_FLOAT64: {
      double value = 0;
      RETURN_IF_NOT_OK(fill_output->GetItemAt(&value, {}));
      out->Fill<double>(value);
      break;
    }
    case DataType::DE_STRING: {
      std::vector<std::string> strings;
      std::string_view fill_string_view;
      RETURN_IF_NOT_OK(fill_value->GetItemAt(&fill_string_view, {}));
      std::string fill_string = std::string(fill_string_view);
      for (int i = 0; i < input->shape().NumOfElements(); i++) {
        strings.emplace_back(fill_string);
      }
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, strings, input->shape()));
      break;
    }
    case DataType::DE_UNKNOWN: {
      RETURN_STATUS_UNEXPECTED("FillOp does not support input of this type.");
      break;
    }
  }

  *output = out;
  return Status::OK();
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

  RETURN_IF_NOT_OK((*output)->AllocateBuffer((*output)->SizeInBytes()));
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
  RETURN_IF_NOT_OK((*output)->AllocateBuffer((*output)->SizeInBytes()));

  auto in_itr = input->begin<float>();
  auto out_itr = (*output)->begin<float16>();
  auto out_end = (*output)->end<float16>();

  for (; out_itr != out_end; in_itr++, out_itr++) {
    float element = *in_itr;
    float float16_max = static_cast<float>(std::numeric_limits<Eigen::half>::max());
    float float16_min = static_cast<float>(std::numeric_limits<Eigen::half>::lowest());
    if (element > float16_max || element < float16_min) {
      RETURN_STATUS_UNEXPECTED("Value " + std::to_string(element) + " is outside of valid float16 range [" +
                               std::to_string(float16_max) + ", " + std::to_string(float16_min) + "].");
    }

    *out_itr = Eigen::half(*in_itr);
  }

  return Status::OK();
}

Status PadEnd(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> *dst, const std::vector<dsize_t> &pad_shape,
              const std::shared_ptr<Tensor> &pad_val) {
  if (pad_val == nullptr) {
    if (src->type().IsNumeric()) {
      return PadEndNumeric(src, dst, pad_shape, 0);
    } else {
      return PadEndString(src, dst, pad_shape, "");
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(src->type().IsNumeric() == pad_val->type().IsNumeric(),
                               "Source and pad_value tensors are not of the same type.");
  if (pad_val->type().IsNumeric()) {
    std::shared_ptr<Tensor> float_pad_value;
    RETURN_IF_NOT_OK(TypeCast(pad_val, &float_pad_value, DataType(DataType::DE_FLOAT32)));
    float val = 0;
    RETURN_IF_NOT_OK(float_pad_value->GetItemAt<float>(&val, {}));
    return PadEndNumeric(src, dst, pad_shape, val);
  }
  std::string_view val;
  RETURN_IF_NOT_OK(pad_val->GetItemAt(&val, {}));
  return PadEndString(src, dst, pad_shape, std::string(val));
}

Status PadEndNumeric(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> *dst,
                     const std::vector<dsize_t> &pad_shape, float pad_val) {
  CHECK_FAIL_RETURN_UNEXPECTED(src != nullptr && dst != nullptr, "tensor can't be nullptr");
  if (src->Rank() == 0 || src->shape().AsVector() == pad_shape) {
    (*dst) = src;  // if no padding, copy the pointer
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(src->Rank() == pad_shape.size(), "Pad to diff rank not allowed");
    RETURN_IF_NOT_OK(Tensor::CreateTensor(dst, TensorImpl::kFlexible, TensorShape(pad_shape), src->type()));
    auto tensor_type = src->type().value();
    if (pad_val == 0) {  // if pad with zero, don't care what type it is
      RETURN_IF_NOT_OK((*dst)->Zero());
    } else if (tensor_type == DataType::DE_INT8) {
      RETURN_IF_NOT_OK((*dst)->Fill<int8_t>(pad_val));
    } else if (tensor_type == DataType::DE_BOOL) {
      RETURN_IF_NOT_OK((*dst)->Fill<bool>(pad_val));
    } else if (tensor_type == DataType::DE_UINT8) {
      RETURN_IF_NOT_OK((*dst)->Fill<uint8_t>(pad_val));
    } else if (tensor_type == DataType::DE_INT16) {
      RETURN_IF_NOT_OK((*dst)->Fill<int16_t>(pad_val));
    } else if (tensor_type == DataType::DE_FLOAT16) {
      RETURN_IF_NOT_OK((*dst)->Fill<float16>(static_cast<float16>(pad_val)));
    } else if (tensor_type == DataType::DE_UINT16) {
      RETURN_IF_NOT_OK((*dst)->Fill<uint16_t>(pad_val));
    } else if (tensor_type == DataType::DE_INT32) {
      RETURN_IF_NOT_OK((*dst)->Fill<int32_t>(pad_val));
    } else if (tensor_type == DataType::DE_UINT32) {
      RETURN_IF_NOT_OK((*dst)->Fill<uint32_t>(pad_val));
    } else if (tensor_type == DataType::DE_INT64) {
      RETURN_IF_NOT_OK((*dst)->Fill<int64_t>(pad_val));
    } else if (tensor_type == DataType::DE_UINT64) {
      RETURN_IF_NOT_OK((*dst)->Fill<uint64_t>(pad_val));
    } else if (tensor_type == DataType::DE_FLOAT32) {
      RETURN_IF_NOT_OK((*dst)->Fill<float>(pad_val));
    } else if (tensor_type == DataType::DE_FLOAT64) {
      RETURN_IF_NOT_OK((*dst)->Fill<double>(pad_val));
    } else {
      RETURN_STATUS_UNEXPECTED("Incorrect/Unknown tensor type");
    }
    std::vector<dsize_t> cur_ind(src->Rank(), 0);
    RETURN_IF_NOT_OK(PadEndNumericHelper(src, *dst, cur_ind, 0));
  }
  return Status::OK();
}
Status PadEndNumericHelper(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> dst,
                           std::vector<dsize_t> cur_ind, size_t cur_dim) {
  if (cur_dim == src->Rank() - 1) {  // if this is the last dimension, copy the data
    dst->CopyLastDimAt(src, cur_ind);
  } else {  // not the last dimension, keep doing recursion
    dsize_t min_ind = std::min(dst->shape()[cur_dim], src->shape()[cur_dim]);
    for (dsize_t i = 0; i < min_ind; i++) {
      cur_ind[cur_dim] = i;
      RETURN_IF_NOT_OK(PadEndNumericHelper(src, dst, cur_ind, cur_dim + 1));
    }
  }
  return Status::OK();
}

Status PadEndString(const std::shared_ptr<Tensor> &src, std::shared_ptr<Tensor> *dst,
                    const std::vector<dsize_t> &pad_shape, const std::string &pad_val) {
  CHECK_FAIL_RETURN_UNEXPECTED(src != nullptr && dst != nullptr, "tensor can't be nullptr");
  if (src->Rank() == 0 || src->shape().AsVector() == pad_shape) {
    (*dst) = src;  // if no padding, copy the pointer
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(src->Rank() == pad_shape.size(), "Pad to diff rank not allowed");
    std::vector<dsize_t> cur_ind(src->Rank(), 0);
    std::vector<std::string> strings;
    RETURN_IF_NOT_OK(PadEndStringHelper(src, &strings, TensorShape(pad_shape), cur_ind, 0, pad_val));
    RETURN_IF_NOT_OK(Tensor::CreateTensor(dst, strings, TensorShape(pad_shape)));
  }
  return Status::OK();
}

Status PadEndStringHelper(const std::shared_ptr<Tensor> &src, std::vector<std::string> *dst,
                          const TensorShape &dst_shape, std::vector<dsize_t> cur_ind, size_t cur_dim,
                          const std::string &pad_value) {
  if (cur_dim == src->Rank() - 1) {  // if this is the last dimension, copy the data
    dsize_t min_ind = std::min(dst_shape[cur_dim], src->shape()[cur_dim]);
    for (dsize_t i = 0; i < min_ind; i++) {
      cur_ind[cur_dim] = i;
      std::string_view item;
      RETURN_IF_NOT_OK(src->GetItemAt(&item, cur_ind));
      dst->emplace_back(item);
    }
    for (dsize_t i = min_ind; i < dst_shape[cur_dim]; i++) {
      dst->emplace_back(pad_value);
    }

  } else {  // not the last dimension, keep doing recursion
    dsize_t min_ind = std::min(dst_shape[cur_dim], src->shape()[cur_dim]);
    for (dsize_t i = 0; i < min_ind; i++) {
      cur_ind[cur_dim] = i;
      RETURN_IF_NOT_OK(PadEndStringHelper(src, dst, dst_shape, cur_ind, cur_dim + 1, pad_value));
    }
    dsize_t count = (dst_shape[cur_dim] - min_ind) * dst_shape.Strides()[cur_dim];
    for (dsize_t i = 0; i < count; i++) {
      dst->emplace_back(pad_value);
    }
  }
  return Status::OK();
}

template <typename T>
Status MaskHelper(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &output,
                  const std::shared_ptr<Tensor> &value_tensor, RelationalOp op) {
  T value;
  RETURN_IF_NOT_OK(value_tensor->GetItemAt(&value, {}));
  auto in_itr = input->begin<T>();
  auto out_itr = output->begin<bool>();
  for (; in_itr != input->end<T>(); in_itr++, out_itr++) {
    switch (op) {
      case RelationalOp::kEqual:
        *out_itr = (*in_itr == value);
        break;
      case RelationalOp::kNotEqual:
        *out_itr = (*in_itr != value);
        break;
      case RelationalOp::kGreater:
        *out_itr = (*in_itr > value);
        break;
      case RelationalOp::kGreaterEqual:
        *out_itr = (*in_itr >= value);
        break;
      case RelationalOp::kLess:
        *out_itr = (*in_itr < value);
        break;
      case RelationalOp::kLessEqual:
        *out_itr = (*in_itr <= value);
        break;
      default:
        RETURN_STATUS_UNEXPECTED("Unknown relational operator.");
    }
  }
  return Status::OK();
}

Status Mask(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::shared_ptr<Tensor> &value,
            RelationalOp op) {
  CHECK_FAIL_RETURN_UNEXPECTED(input->type().IsNumeric() == value->type().IsNumeric(),
                               "Cannot convert constant value to the type of the input tensor.");
  CHECK_FAIL_RETURN_UNEXPECTED(value->shape() == TensorShape::CreateScalar(), "Value is not a scalar");

  RETURN_IF_NOT_OK(Tensor::CreateTensor(output, TensorImpl::kFlexible, input->shape(), DataType(DataType::DE_BOOL)));

  std::unique_ptr<TypeCastOp> value_cast_op(new TypeCastOp(input->type()));
  std::shared_ptr<Tensor> casted_value;
  if (input->type().IsNumeric()) {
    RETURN_IF_NOT_OK(value_cast_op->Compute(value, &casted_value));
  } else {
    casted_value = value;
  }

  switch (input->type().value()) {
    case DataType::DE_BOOL:
      RETURN_IF_NOT_OK(MaskHelper<bool>(input, *output, casted_value, op));
      break;
    case DataType::DE_INT8:
      RETURN_IF_NOT_OK(MaskHelper<int8_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_UINT8:
      RETURN_IF_NOT_OK(MaskHelper<uint8_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_UINT16:
      RETURN_IF_NOT_OK(MaskHelper<uint16_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_INT16:
      RETURN_IF_NOT_OK(MaskHelper<int16_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_UINT32:
      RETURN_IF_NOT_OK(MaskHelper<uint32_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_INT32:
      RETURN_IF_NOT_OK(MaskHelper<int32_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_UINT64:
      RETURN_IF_NOT_OK(MaskHelper<uint64_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_INT64:
      RETURN_IF_NOT_OK(MaskHelper<int64_t>(input, *output, casted_value, op));
      break;
    case DataType::DE_FLOAT16:
      RETURN_IF_NOT_OK(MaskHelper<float16>(input, *output, casted_value, op));
      break;
    case DataType::DE_FLOAT32:
      RETURN_IF_NOT_OK(MaskHelper<float>(input, *output, casted_value, op));
      break;
    case DataType::DE_FLOAT64:
      RETURN_IF_NOT_OK(MaskHelper<double>(input, *output, casted_value, op));
      break;
    case DataType::DE_STRING:
      RETURN_IF_NOT_OK(MaskHelper<std::string_view>(input, *output, casted_value, op));
      break;
    case DataType::DE_UNKNOWN:
      RETURN_STATUS_UNEXPECTED("Unsupported input type.");
      break;
  }
  return Status::OK();
}

Status Concatenate(const TensorRow &input, TensorRow *output, int8_t axis, std::shared_ptr<Tensor> prepend,
                   std::shared_ptr<Tensor> append) {
  CHECK_FAIL_RETURN_UNEXPECTED(input[0]->shape().Rank() == 1, "Only 1D tensors supported");
  CHECK_FAIL_RETURN_UNEXPECTED(axis == 0 || axis == -1, "Only concatenation along the last dimension supported");

  axis = Tensor::HandleNeg(axis, input[0]->shape().Rank());
  CHECK_FAIL_RETURN_UNEXPECTED(axis == 0, "Only axis=0 is supported");

  std::shared_ptr<Tensor> out;
  if (prepend != nullptr) {
    CHECK_FAIL_RETURN_UNEXPECTED(prepend->shape().Rank() == 1, "Only 1D tensors supported");
    RETURN_IF_NOT_OK(ConcatenateHelper(prepend, &out, axis, input[0]));
  } else {
    out = input[0];
  }
  for (dsize_t i = 1; i < input.size(); i++) {
    std::shared_ptr<Tensor> out_t;
    CHECK_FAIL_RETURN_UNEXPECTED(input[i]->shape().Rank() == 1, "Only 1D tensors supported");
    RETURN_IF_NOT_OK(ConcatenateHelper(out, &out_t, axis, input[i]));
    out = out_t;
  }
  std::shared_ptr<Tensor> out_t;
  if (append != nullptr) {
    CHECK_FAIL_RETURN_UNEXPECTED(append->shape().Rank() == 1, "Only 1D tensors supported");
    RETURN_IF_NOT_OK(ConcatenateHelper(out, &out_t, axis, append));
  } else {
    out_t = out;
  }
  output->push_back(out_t);

  return Status::OK();
}

Status ConcatenateHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int8_t axis,
                         std::shared_ptr<Tensor> append) {
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == append->type(), "Tensor types do not match");

  TensorShape t({});

  for (dsize_t i = 0; i < input->shape().Rank(); i++) {
    if (i != axis) {
      t = t.AppendDim(input->shape()[i]);
    } else {
      dsize_t new_shape = input->shape()[i] + append->shape()[i];

      t = t.AppendDim(new_shape);
    }
  }
  std::shared_ptr<Tensor> out;

  if (input->type().IsNumeric()) {
    RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, TensorImpl::kFlexible, t, input->type()));

    RETURN_IF_NOT_OK(out->Concatenate({0}, input));
    RETURN_IF_NOT_OK(out->Concatenate({input->shape()[0]}, append));
    *output = out;
  } else {
    std::vector<std::string> strings;

    auto itr = input->begin<std::string_view>();
    for (; itr != input->end<std::string_view>(); itr++) {
      strings.emplace_back(*itr);
    }
    itr = append->begin<std::string_view>();
    for (; itr != append->end<std::string_view>(); itr++) {
      strings.emplace_back(*itr);
    }
    RETURN_IF_NOT_OK(Tensor::CreateTensor(&out, strings, t));

    *output = out;
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
