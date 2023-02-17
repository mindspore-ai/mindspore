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

#include "transform/graph_ir/transform_util.h"
#include <utility>
#include <map>
#include <algorithm>

#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"
#ifndef ENABLE_LITE_ACL
#include "include/common/utils/python_adapter.h"
#endif

namespace mindspore {
namespace transform {
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

const size_t kErrorSize = 0;
const size_t kIdx0 = 0;
const size_t kIdx1 = 1;
const size_t kIdx2 = 2;
const size_t kIdx3 = 3;

vector<int64_t> TransformUtil::ConvertIntToList(int64_t data, int size) {
  vector<int64_t> list{};
  if (size <= 0) {
    MS_LOG(WARNING) << "size <= 0";
    return list;
  }
  for (int i = 0; i < size; ++i) {
    list.push_back(data);
  }
  return list;
}

static std::map<MeDataType, GeDataType> datatype_trans_map = {
  {MeDataType::kNumberTypeFloat16, GeDataType::DT_FLOAT16}, {MeDataType::kNumberTypeFloat32, GeDataType::DT_FLOAT},
  {MeDataType::kNumberTypeFloat64, GeDataType::DT_DOUBLE},  {MeDataType::kNumberTypeInt8, GeDataType::DT_INT8},
  {MeDataType::kNumberTypeInt16, GeDataType::DT_INT16},     {MeDataType::kNumberTypeInt32, GeDataType::DT_INT32},
  {MeDataType::kNumberTypeInt64, GeDataType::DT_INT64},     {MeDataType::kNumberTypeUInt8, GeDataType::DT_UINT8},
  {MeDataType::kNumberTypeUInt16, GeDataType::DT_UINT16},   {MeDataType::kNumberTypeUInt32, GeDataType::DT_UINT32},
  {MeDataType::kNumberTypeUInt64, GeDataType::DT_UINT64},   {MeDataType::kNumberTypeBool, GeDataType::DT_BOOL},
  {MeDataType::kObjectTypeString, GeDataType::DT_STRING},   {MeDataType::kNumberTypeFloat, GeDataType::DT_FLOAT}};

GeDataType TransformUtil::ConvertDataType(const MeDataType &type) {
  MS_LOG(DEBUG) << "Convert me data type: " << TypeIdLabel(type) << " to ge data type";
  if (datatype_trans_map.find(type) != datatype_trans_map.end()) {
    return datatype_trans_map[type];
  } else {
    return GeDataType::DT_UNDEFINED;
  }
}

static std::map<MeDataType, size_t> datatype_size_map = {
  {MeDataType::kNumberTypeFloat16, sizeof(float) / 2}, {MeDataType::kNumberTypeFloat32, sizeof(float)},  // 1/2 of float
  {MeDataType::kNumberTypeFloat64, sizeof(double)},    {MeDataType::kNumberTypeInt8, sizeof(int8_t)},
  {MeDataType::kNumberTypeInt16, sizeof(int16_t)},     {MeDataType::kNumberTypeInt32, sizeof(int32_t)},
  {MeDataType::kNumberTypeInt64, sizeof(int64_t)},     {MeDataType::kNumberTypeUInt8, sizeof(uint8_t)},
  {MeDataType::kNumberTypeUInt16, sizeof(uint16_t)},   {MeDataType::kNumberTypeUInt32, sizeof(uint32_t)},
  {MeDataType::kNumberTypeUInt64, sizeof(uint64_t)},   {MeDataType::kNumberTypeBool, sizeof(bool)},
  {MeDataType::kNumberTypeFloat, sizeof(float)}};

size_t TransformUtil::GetDataTypeSize(const MeDataType &type) {
  if (datatype_size_map.find(type) != datatype_size_map.end()) {
    return datatype_size_map[type];
  } else {
    MS_LOG(ERROR) << "Illegal tensor data type!";
    return kErrorSize;
  }
}

GeFormat TransformUtil::ConvertFormat(const string &format, const size_t shape_size) {
  static constexpr size_t k4dSize = 4;
  static const std::map<std::string, GeFormat> format_map = {
    {kOpFormat_DEFAULT, GeFormat::FORMAT_NCHW},
    {kOpFormat_NC1KHKWHWC0, GeFormat::FORMAT_NC1KHKWHWC0},
    {kOpFormat_ND, GeFormat::FORMAT_ND},
    {kOpFormat_NCHW, GeFormat::FORMAT_NCHW},
    {kOpFormat_NHWC, GeFormat::FORMAT_NHWC},
    {kOpFormat_HWCN, GeFormat::FORMAT_HWCN},
    {kOpFormat_NC1HWC0, GeFormat::FORMAT_NC1HWC0},
    {kOpFormat_FRAC_Z, GeFormat::FORMAT_FRACTAL_Z},
    {kOpFormat_FRAC_NZ, GeFormat::FORMAT_FRACTAL_NZ},
    {kOpFormat_C1HWNCoC0, GeFormat::FORMAT_C1HWNCoC0},
    {kOpFormat_NC1HWC0_C04, GeFormat::FORMAT_NC1HWC0_C04},
    {kOpFormat_FRACTAL_Z_C04, GeFormat::FORMAT_FRACTAL_Z_C04},
    {kOpFormat_NDHWC, GeFormat::FORMAT_NDHWC},
    {kOpFormat_NCDHW, GeFormat::FORMAT_NCDHW},
    {kOpFormat_DHWNC, GeFormat::FORMAT_DHWNC},
    {kOpFormat_DHWCN, GeFormat::FORMAT_DHWCN},
    {kOpFormat_NDC1HWC0, GeFormat::FORMAT_NDC1HWC0},
    {kOpFormat_FRACTAL_Z_3D, GeFormat::FORMAT_FRACTAL_Z_3D},
    {kOpFormat_FRACTAL_ZN_LSTM, GeFormat::FORMAT_FRACTAL_ZN_LSTM},
    {kOpFormat_ND_RNN_BIAS, GeFormat::FORMAT_ND_RNN_BIAS},
    {kOpFormat_FRACTAL_ZN_RNN, GeFormat::FORMAT_FRACTAL_ZN_RNN}};
  if (format == kOpFormat_DEFAULT) {
    return shape_size == k4dSize ? GeFormat::FORMAT_NCHW : GeFormat::FORMAT_ND;
  }
  auto iter = format_map.find(format);
  if (iter == format_map.end()) {
    MS_LOG(ERROR) << "Illegal tensor data format: (" << format << "). Use ND format instead.";
    return GeFormat::FORMAT_ND;
  }
  return iter->second;
}

std::shared_ptr<GeTensorDesc> TransformUtil::GetGeTensorDesc(const ShapeVector &me_shape, const MeDataType &me_type,
                                                             const std::string &format, const ShapeVector &ori_shape,
                                                             const std::string &ori_format) {
  // convert me shape to ge shape
  GeShape shape(me_shape);
  if (shape.GetDimNum() == 0) {
    MS_LOG(DEBUG) << "The dims size of Ge tensor is zero";
  }
  // convert me format to ge format
  GeFormat ge_format = ConvertFormat(format, me_shape.size());
  if (ge_format == GeFormat::FORMAT_ND) {
    MS_LOG(DEBUG) << "Set ND data format";
  }
  // convert me datatype to ge datatype
  GeDataType data_type = ConvertDataType(me_type);
  if (data_type == GeDataType::DT_UNDEFINED) {
    MS_LOG(ERROR) << "undefined data type :" << me_type;
    return nullptr;
  }

  auto desc = std::make_shared<GeTensorDesc>(shape, ge_format, data_type);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Create GeTensorDesc failed!";
    return nullptr;
  }

  if (!ori_shape.empty()) {
    GeShape ge_ori_shape(ori_shape);
    desc->SetOriginShape(ge_ori_shape);
  }

  if (!ori_format.empty()) {
    GeFormat ge_ori_format = ConvertFormat(ori_format, ori_shape.size());
    desc->SetOriginFormat(ge_ori_format);
  }

  MS_LOG(DEBUG) << "SetRealDimCnt is :" << me_shape.size();
  desc->SetRealDimCnt(SizeToInt(me_shape.size()));
  return desc;
}

// if failed, return empty vector.
std::vector<GeTensorPtr> TransformUtil::ConvertInputTensors(const std::vector<MeTensorPtr> &me_tensors,
                                                            const std::string &format) {
  std::vector<GeTensorPtr> ge_tensors;

  for (size_t index = 0; index < me_tensors.size(); index++) {
    MS_EXCEPTION_IF_NULL(me_tensors[index]);
    MS_LOG(INFO) << "me_tensor " << index << " 's data size is: " << me_tensors[index]->DataSize();
    auto shape = me_tensors[index]->shape();
    std::string shape_str;
    for (size_t i = 0; i < shape.size(); i++) {
      shape_str += std::to_string(shape[i]);
      shape_str += " ";
    }
    MS_LOG(INFO) << "me_tensor " << index << " 's shape is: { " << shape_str << "}";
    MS_LOG(INFO) << "me_tensor " << index << " 's type is: " << me_tensors[index]->data_type();

    auto ge_tensor_ptr = TransformUtil::ConvertTensor(me_tensors[index], format);
    if (ge_tensor_ptr != nullptr) {
      (void)ge_tensors.emplace_back(ge_tensor_ptr);
    } else {
      MS_LOG(ERROR) << "Convert me_tensor " << index << " to Ge Tensor failed!";
      ge_tensors.clear();
      return ge_tensors;
    }
  }
  return ge_tensors;
}

#ifndef ENABLE_LITE_ACL
GeTensorPtr ConvertStringTensor(const MeTensorPtr &tensor, const std::string &format) {
  auto desc = TransformUtil::GetGeTensorDesc(tensor->shape_c(), tensor->data_type(), format);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Failed to get Tensor Desc";
    return nullptr;
  }
  GeTensorPtr tensor_ptr = nullptr;
  auto data_buff_size = tensor->data().nbytes();

  auto py_array = python_adapter::PyAdapterCallback::TensorToNumpy(*tensor);
  auto buf = py_array.request();
  auto data_ptr = static_cast<char *>(tensor->data().data());
  size_t single_char_offset = 4;

  if (buf.format.back() == 'w') {
    auto max_length = buf.format.substr(0, buf.format.length() - 1);
    auto string_max_length = LongToSize(std::stol(max_length));
    if (string_max_length == 0) {
      MS_LOG(ERROR) << "Failed to get Tensor Desc. Please check string length";
      return nullptr;
    }
    size_t elements_num = (data_buff_size / single_char_offset) / string_max_length;
    std::vector<std::string> string_vector;
    char *string_element = new char[string_max_length];
    size_t string_length = 0;
    for (size_t i = 0; i < elements_num; i++) {
      (void)std::fill_n(string_element, string_max_length, '\0');
      for (size_t j = 0; j < string_max_length; j++) {
        char char_element = data_ptr[i * string_max_length * single_char_offset + single_char_offset * j];
        if (static_cast<int>(char_element) == 0) {
          break;
        } else {
          string_element[j] = char_element;
          string_length += 1;
        }
      }
      std::string string_to_add(string_element, string_length);
      (void)string_vector.emplace_back(string_to_add);
    }
    delete[] string_element;
    string_element = nullptr;
    tensor_ptr = make_shared<GeTensor>(*desc);
    (void)tensor_ptr->SetData(string_vector);
  } else {
    auto string_length = LongToSize(std::stol(buf.format.substr(0, buf.format.length() - 1)));
    if (string_length == 0) {
      MS_LOG(ERROR) << "Failed to get Tensor Desc. Please check string length";
      return nullptr;
    }
    char *string_element = new char[string_length];
    for (size_t i = 0; i < string_length; i++) {
      string_element[i] = data_ptr[i];
    }
    std::string string_to_add(string_element, string_length);
    tensor_ptr = make_shared<GeTensor>(*desc);
    (void)tensor_ptr->SetData(string_to_add);
    delete[] string_element;
    string_element = nullptr;
  }
  return tensor_ptr;
}
#endif

GeTensorPtr TransformUtil::ConvertTensor(const MeTensorPtr &tensor, const std::string &format) {
  // get tensor data type size
  MS_EXCEPTION_IF_NULL(tensor);
  auto me_data_type = tensor->data_type();
#ifndef ENABLE_LITE_ACL
  if (me_data_type == mindspore::kObjectTypeString) {
    return ConvertStringTensor(tensor, format);
  }
#endif
  size_t type_size = GetDataTypeSize(me_data_type);
  if (type_size == kErrorSize) {
    MS_LOG(ERROR) << "The Me Tensor data type size is wrong, type size is: " << type_size;
    return nullptr;
  }
  size_t elements_num = IntToSize(tensor->ElementsNum());

  // get tensor buff size
  size_t data_buff_size = elements_num * type_size;
  if (data_buff_size == 0) {
    MS_LOG(INFO) << "The Me Tensor data buff size is 0.";
  }
  // create ge tensor
  auto desc = GetGeTensorDesc(tensor->shape_c(), tensor->data_type(), format);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Failed to get Tensor Desc";
    return nullptr;
  }
  GeTensorPtr tensor_ptr = make_shared<GeTensor>(*desc, static_cast<uint8_t *>(tensor->data_c()), data_buff_size);
  if (tensor_ptr != nullptr) {
    MS_LOG(INFO) << "Convert Me Tensor to Ge Tensor success!";
  }
  return tensor_ptr;
}

std::vector<MeTensorPtr> TransformUtil::ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors,
                                                         const std::vector<ShapeVector> &request_dims) {
  std::vector<MeTensorPtr> outputs;

  for (size_t index = 0; index < ge_tensors.size(); index++) {
    MeTensorPtr me_tensor_ptr = nullptr;
    if (index < request_dims.size()) {
      me_tensor_ptr = ConvertGeTensor(ge_tensors[index], request_dims[index]);
    } else {
      ShapeVector empty_shape;
      me_tensor_ptr = ConvertGeTensor(ge_tensors[index], empty_shape);
    }

    if (me_tensor_ptr != nullptr) {
      (void)outputs.emplace_back(me_tensor_ptr);
    } else {
      MS_LOG(ERROR) << "Convert Ge Tensor " << index << " to Me Tensor failed!";
      return outputs;
    }
  }
  return outputs;
}

std::vector<MeTensorPtr> TransformUtil::ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors) {
  std::vector<MeTensorPtr> outputs;

  for (size_t index = 0; index < ge_tensors.size(); index++) {
    MeTensorPtr me_tensor_ptr = ConvertGeTensor(ge_tensors[index]);
    if (me_tensor_ptr != nullptr) {
      (void)outputs.emplace_back(me_tensor_ptr);
    } else {
      MS_LOG(ERROR) << "Convert Ge Tensor " << index << " to Me Tensor failed!";
      return outputs;
    }
  }
  return outputs;
}

MeDataType TransformUtil::ConvertGeDataType(const GeDataType &type) {
  switch (type) {
    case GeDataType::DT_FLOAT16:
      return MeDataType::kNumberTypeFloat16;
    case GeDataType::DT_FLOAT:
      return MeDataType::kNumberTypeFloat32;
    case GeDataType::DT_DOUBLE:
      return MeDataType::kNumberTypeFloat64;
    case GeDataType::DT_INT64:
      return MeDataType::kNumberTypeInt64;
    case GeDataType::DT_INT32:
      return MeDataType::kNumberTypeInt32;
    case GeDataType::DT_INT16:
      return MeDataType::kNumberTypeInt16;
    case GeDataType::DT_INT8:
      return MeDataType::kNumberTypeInt8;
    case GeDataType::DT_BOOL:
      return MeDataType::kNumberTypeBool;
    case GeDataType::DT_UINT8:
      return MeDataType::kNumberTypeUInt8;
    case GeDataType::DT_UINT16:
      return MeDataType::kNumberTypeUInt16;
    case GeDataType::DT_UINT32:
      return MeDataType::kNumberTypeUInt32;
    case GeDataType::DT_UINT64:
      return MeDataType::kNumberTypeUInt64;
    case GeDataType::DT_UNDEFINED:
    case GeDataType::DT_DUAL_SUB_UINT8:
    case GeDataType::DT_DUAL_SUB_INT8:
    case GeDataType::DT_DUAL:
      return MeDataType::kTypeUnknown;
    default:
      return MeDataType::kTypeUnknown;
  }
}

namespace {
bool IsGeShapeCompatible(const GeShape &ge_shape, const ShapeVector &request_dims) {
  MS_LOG(INFO) << "GeTensor's shape is " << TransformUtil::PrintVector(ge_shape.GetDims());
  MS_LOG(INFO) << "Me request shape is " << TransformUtil::PrintVector(request_dims);

  const int GE_DIMS = 4;
  std::vector<int64_t> ge_dims = ge_shape.GetDims();
  if (request_dims.size() > ge_dims.size()) {
    MS_LOG(ERROR) << "Request shape's dims count greater than ge shape's";
    return false;
  }

  // convert NHWC to NCHW
  if ((request_dims.size() == 1) && (ge_dims.size() == GE_DIMS) && (request_dims[kIdx0] == ge_dims[kIdx1]) &&
      (ge_dims[kIdx0] == 1) && (ge_dims[kIdx2] == 1) && (ge_dims[kIdx3] == 1)) {
    MS_LOG(INFO) << "Ge tensor shape and request shape is compatible";
    return true;
  }

  std::string::size_type i = 0;
  for (; i < request_dims.size(); i++) {
    if (ge_dims[i] != request_dims[i]) {
      MS_LOG(ERROR) << "Request shape's dims value not equal to ge shape's";
      return false;
    }
  }

  for (; i < ge_dims.size(); i++) {
    if (ge_dims[i] != 1) {
      MS_LOG(ERROR) << "GeShape's extend dims is not equal to 1";
      return false;
    }
  }
  MS_LOG(INFO) << "Ge tensor shape and request shape is compatible";
  return true;
}
}  // namespace

GeShape TransformUtil::ConvertMeShape(const ShapeVector &me_dims) {
  std::vector<int64_t> ge_dims;
  (void)std::copy(me_dims.begin(), me_dims.end(), std::back_inserter(ge_dims));
  return GeShape(ge_dims);
}

ShapeVector TransformUtil::ConvertGeShape(const GeShape &ge_shape) {
  ShapeVector me_dims;
  std::vector<int64_t> ge_dims = ge_shape.GetDims();
  (void)std::copy(ge_dims.begin(), ge_dims.end(), std::back_inserter(me_dims));
  return me_dims;
}

ShapeVector TransformUtil::ConvertGeShape(const GeShape &ge_shape, const ShapeVector &request_dims) {
  vector<int64_t> ret;
  if (ge_shape.GetDimNum() == 0) {
    MS_LOG(DEBUG) << "GeTensor's shape is scalar";
    return ret;
  }

  if (IsGeShapeCompatible(ge_shape, request_dims) == true) {
    ret = request_dims;
  } else {
    MS_LOG(ERROR) << "GeShape and Me request shape are incompatible, return GeShape";
    ret = ConvertGeShape(ge_shape);
  }
  return ret;
}

MeTensorPtr TransformUtil::GenerateMeTensor(const GeTensorPtr &ge_tensor, const ShapeVector &me_dims,
                                            const TypeId &me_type) {
  MeTensor me_tensor(me_type, me_dims);

  // Get the writable data pointer of the tensor and cast it to its data type.
  auto me_data_ptr = me_tensor.data_c();
  size_t me_data_size = static_cast<size_t>(me_tensor.data().nbytes());
  MS_EXCEPTION_IF_NULL(me_data_ptr);
  MS_EXCEPTION_IF_NULL(ge_tensor);
  if (me_data_size < ge_tensor->GetSize()) {
    MS_LOG(ERROR) << "ME tensor data size[" << me_data_size << " bytes] is less than GE tensor ["
                  << ge_tensor->GetSize() << " bytes]";
    return nullptr;
  }

  // Copy or use the writable data pointer of the ME tensor.
  MS_EXCEPTION_IF_NULL(ge_tensor->GetData());
  if (ge_tensor->GetSize() == 0) {
    MS_LOG(ERROR) << "GE tensor data size is zero!";
    return nullptr;
  }

  // Use memcpy here, not memcpy_s, just because the size of ge_tensor may be bigger than 2GB
  // which is the size limit of memcpy_s.
  (void)memcpy(me_data_ptr, ge_tensor->GetData(), ge_tensor->GetSize());

  return make_shared<MeTensor>(me_tensor);
}

MeTensorPtr TransformUtil::ConvertGeTensor(const GeTensorPtr &ge_tensor) {
  MS_EXCEPTION_IF_NULL(ge_tensor);
  GeShape ge_shape = ge_tensor->GetTensorDesc().GetShape();
  vector<int64_t> me_dims = ConvertGeShape(ge_shape);

  TypeId type_id = ConvertGeDataType(ge_tensor->GetTensorDesc().GetDataType());
  if (type_id == MeDataType::kTypeUnknown) {
    MS_LOG(ERROR) << "Could not convert Ge Tensor because of unsupported data type: "
                  << static_cast<int>(ge_tensor->GetTensorDesc().GetDataType());
    return nullptr;
  }
  return GenerateMeTensor(ge_tensor, me_dims, type_id);
}

MeTensorPtr TransformUtil::ConvertGeTensor(const GeTensorPtr &ge_tensor, const TypeId &me_type) {
  MS_EXCEPTION_IF_NULL(ge_tensor);
  GeShape ge_shape = ge_tensor->GetTensorDesc().GetShape();
  vector<int64_t> me_dims = ConvertGeShape(ge_shape);

  if (me_type == MeDataType::kTypeUnknown) {
    MS_LOG(ERROR) << "Unsupported data type: " << static_cast<int>(me_type);
    return nullptr;
  }
  return GenerateMeTensor(ge_tensor, me_dims, me_type);
}

// if request_dims is empty, use ge tensor's shape,otherwise convert to request shape
MeTensorPtr TransformUtil::ConvertGeTensor(const GeTensorPtr ge_tensor, const ShapeVector &request_dims) {
  MS_EXCEPTION_IF_NULL(ge_tensor);
  GeShape ge_shape = ge_tensor->GetTensorDesc().GetShape();
  vector<int64_t> me_dims = ConvertGeShape(ge_shape, request_dims);
  MS_LOG(INFO) << "GE tensor type is " << static_cast<int>(ge_tensor->GetTensorDesc().GetDataType());
  // Create a tensor with wanted data type and shape
  TypeId type_id = ConvertGeDataType(ge_tensor->GetTensorDesc().GetDataType());
  if (type_id == MeDataType::kTypeUnknown) {
    MS_LOG(ERROR) << "Could not convert Ge Tensor because of unsupported data type: "
                  << static_cast<int>(ge_tensor->GetTensorDesc().GetDataType());
    return nullptr;
  }
  return GenerateMeTensor(ge_tensor, me_dims, type_id);
}

std::string TransformUtil::PrintGeTensor(const GeTensorPtr ge_tensor) {
  std::string ret;
  if (ge_tensor == nullptr) {
    MS_LOG(ERROR) << "Input ge tensor is nullptr";
    return ret;
  }

  MS_LOG(INFO) << "Ge Tensor data type is : " << static_cast<int>(ge_tensor->GetTensorDesc().GetDataType());
  switch (static_cast<int>(ge_tensor->GetTensorDesc().GetDataType())) {
    case GeDataType::DT_UINT32:
      ret = PrintVector(MakeVector<uint32_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_FLOAT:
      ret = PrintVector(MakeVector<float_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_INT32:
      ret = PrintVector(MakeVector<int32_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_DOUBLE:
      ret = PrintVector(MakeVector<double_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_INT64:
      ret = PrintVector(MakeVector<int64_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_UINT64:
      ret = PrintVector(MakeVector<uint64_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_INT16:
      ret = PrintVector(MakeVector<int16_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_UINT16:
      ret = PrintVector(MakeVector<uint16_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_DUAL_SUB_INT8:
    case GeDataType::DT_INT8:
      ret = PrintVector(MakeVector<int8_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_UINT8:
    case GeDataType::DT_DUAL_SUB_UINT8:
      ret = PrintVector(MakeVector<uint8_t>(ge_tensor->GetData(), ge_tensor->GetSize()));
      break;
    case GeDataType::DT_FLOAT16:
    case GeDataType::DT_BOOL:
    case GeDataType::DT_UNDEFINED:
    case GeDataType::DT_DUAL:
    default:
      MS_LOG(ERROR) << "Unsupported to print type:" << static_cast<int>(ge_tensor->GetTensorDesc().GetDataType())
                    << " ge tensor";
      break;
  }
  return ret;
}

std::string TransformUtil::NormOpName(const std::string &anf_name) {
  std::string str = anf_name.substr(anf_name.rfind("/") + 1);
  std::string ret;
  for (const auto &c : str) {
    if (std::isalnum(c) || c == '_' || c == '-') {
      ret += c;
    }
  }
  return ret;
}
}  // namespace transform
}  // namespace mindspore
