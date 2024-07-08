/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include <complex>

#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "transform/graph_ir/op_adapter_util.h"

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

namespace {
class MsTensorRel {
 public:
  explicit MsTensorRel(const MeTensorPtr &tensor) : tensor_(tensor) {}
  ~MsTensorRel() = default;
  void Rel() const { tensor_ = nullptr; }

 private:
  mutable MeTensorPtr tensor_;
};
}  // namespace

class TensorRefData : public tensor::TensorData {
 public:
  TensorRefData(void *data, ssize_t data_size, ssize_t itemsize, ssize_t ndim)
      : data_(data), data_size_(data_size), itemsize_(itemsize), ndim_(ndim) {}

  ~TensorRefData() override = default;

  // Total number of elements.
  ssize_t size() const override { return data_size_; }

  // Byte size of a single element.
  ssize_t itemsize() const override { return itemsize_; }

  // Total number of bytes.
  ssize_t nbytes() const override { return size() * itemsize(); }

  // Number of dimensions.
  ssize_t ndim() const override { return ndim_; }

  void *data() override { return data_; }
  const void *const_data() const override { return data_; }

  bool is_sub_data() const override { return false; }
  bool has_sub_data() const override { return false; }

  std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const override { return ""; }

 protected:
  void *data_ = nullptr;
  ssize_t data_size_ = 0;
  ssize_t itemsize_ = 0;
  ssize_t ndim_ = 0;
};

vector<int64_t> TransformUtil::ConvertIntToList(int64_t data, int size) {
  vector<int64_t> list{};
  if (size <= 0) {
    MS_LOG(WARNING) << "size <= 0";
    return list;
  }
  for (int i = 0; i < size; ++i) {
    list.emplace_back(data);
  }
  return list;
}

static std::map<MeDataType, GeDataType> datatype_trans_map = {
  {MeDataType::kNumberTypeFloat16, GeDataType::DT_FLOAT16},
  {MeDataType::kNumberTypeFloat32, GeDataType::DT_FLOAT},
  {MeDataType::kNumberTypeFloat64, GeDataType::DT_DOUBLE},
  {MeDataType::kNumberTypeBFloat16, GeDataType::DT_BF16},
  {MeDataType::kNumberTypeInt4, GeDataType::DT_INT4},
  {MeDataType::kNumberTypeInt8, GeDataType::DT_INT8},
  {MeDataType::kNumberTypeInt16, GeDataType::DT_INT16},
  {MeDataType::kNumberTypeInt32, GeDataType::DT_INT32},
  {MeDataType::kNumberTypeInt64, GeDataType::DT_INT64},
  {MeDataType::kNumberTypeUInt8, GeDataType::DT_UINT8},
  {MeDataType::kNumberTypeUInt16, GeDataType::DT_UINT16},
  {MeDataType::kNumberTypeUInt32, GeDataType::DT_UINT32},
  {MeDataType::kNumberTypeUInt64, GeDataType::DT_UINT64},
  {MeDataType::kNumberTypeBool, GeDataType::DT_BOOL},
  {MeDataType::kObjectTypeString, GeDataType::DT_STRING},
  {MeDataType::kNumberTypeFloat, GeDataType::DT_FLOAT},
  {MeDataType::kNumberTypeComplex64, GeDataType::DT_COMPLEX64},
  {MeDataType::kNumberTypeComplex128, GeDataType::DT_COMPLEX128}};

GeDataType TransformUtil::ConvertDataType(const MeDataType &type) {
  MS_LOG(DEBUG) << "Convert me data type: " << TypeIdLabel(type) << " to ge data type";
  if (datatype_trans_map.find(type) != datatype_trans_map.end()) {
    return datatype_trans_map[type];
  } else {
    return GeDataType::DT_UNDEFINED;
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

std::shared_ptr<GeTensorDesc> TransformUtil::GetGeTensorDesc(const ShapeVector &ori_shape, const MeDataType &me_type,
                                                             const std::string &ori_format,
                                                             const ShapeVector &dev_shape,
                                                             const std::string &dev_format) {
  // convert me shape to ge shape
  GeShape ori_ge_shape(ori_shape);
  if (ori_ge_shape.GetDimNum() == 0) {
    MS_LOG(DEBUG) << "The dims size of Ge tensor is zero";
  }
  // convert me format to ge format
  GeFormat ori_ge_format = ConvertFormat(ori_format, ori_shape.size());
  if (ori_ge_format == GeFormat::FORMAT_ND) {
    MS_LOG(DEBUG) << "Set ND data format";
  }
  // convert me datatype to ge datatype
  GeDataType data_type = ConvertDataType(me_type);
  if (data_type == GeDataType::DT_UNDEFINED) {
    MS_LOG(WARNING) << "undefined data type :" << me_type;
    return nullptr;
  }
  auto desc = std::make_shared<GeTensorDesc>();
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Create GeTensorDesc failed!";
    return nullptr;
  }
  // set ori shape and format.
  // note: if ori_shape and ori_format have been set. the set_shape and set_format will run as device info, otherwise
  // the set_shape and set_format will run as host info.
  if (!std::any_of(ori_shape.cbegin(), ori_shape.cend(), [](const auto &dim) { return dim < 0; })) {
    desc->SetOriginShape(ori_ge_shape);
    desc->SetOriginFormat(ori_ge_format);
  }
  desc->SetDataType(data_type);

  // set device shape and format, if value is empty, use ori shape and format replace.
  auto dev_ge_shape = dev_shape.empty() ? ori_ge_shape : GeShape(dev_shape);
  GeFormat dev_ge_format = dev_format.empty() ? ori_ge_format : ConvertFormat(dev_format, dev_ge_shape.GetDimNum());
  if (me_type == MeDataType::kNumberTypeInt4) {
    int64_t last_dim = dev_ge_shape.GetDimNum() - 1;
    dev_ge_shape.SetDim(last_dim, dev_ge_shape.GetDim(last_dim) * 2);
  }
  desc->SetShape(dev_ge_shape);
  desc->SetFormat(dev_ge_format);

  MS_LOG(DEBUG) << "SetRealDimCnt is :" << ori_shape.size();
  desc->SetRealDimCnt(SizeToInt(ori_shape.size()));
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
  py::gil_scoped_acquire gil;
  auto py_array = python_adapter::PyAdapterCallback::TensorToNumpy(*tensor);
  auto buf = py_array.request();
  auto data_ptr = static_cast<char *>(tensor->data().data());
  size_t single_char_offset = 4;

  if (buf.format.back() == 'w') {
    auto max_length = buf.format.substr(0, buf.format.length() - 1);
    int64_t max_length_long = 0;
    try {
      max_length_long = std::stol(max_length);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Invalid argument:" << e.what() << " when parse " << max_length;
    }
    auto string_max_length = LongToSize(max_length_long);
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
    int64_t length_long = 0;
    try {
      length_long = std::stol(buf.format.substr(0, buf.format.length() - 1));
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Invalid argument:" << e.what() << " when parse "
                        << buf.format.substr(0, buf.format.length() - 1);
    }
    auto string_length = LongToSize(length_long);
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

GeTensorPtr TransformUtil::ConvertTensor(const MeTensorPtr &tensor, const std::string &format, bool copy) {
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

  // get tensor buff size
  size_t data_buff_size = tensor->Size();
  if (data_buff_size == 0) {
    MS_LOG(INFO) << "The Me Tensor data buff size is 0.";
  }
  // create ge tensor
  auto desc = GetGeTensorDesc(tensor->shape_c(), tensor->data_type(), format);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Failed to get Tensor Desc";
    return nullptr;
  }
  GeTensorPtr tensor_ptr = make_shared<GeTensor>(*desc);
  if (tensor_ptr == nullptr) {
    MS_LOG(ERROR) << "Failed to convert Me Tensor to Ge Tensor!";
    return nullptr;
  }
  if (copy) {
    auto ret = tensor_ptr->SetData(static_cast<uint8_t *>(tensor->data_c()), data_buff_size);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(const uint8_t*, size), data size " << data_buff_size;
      return nullptr;
    }
  } else {
    MsTensorRel rel(tensor);
    auto ret = tensor_ptr->SetData(static_cast<uint8_t *>(tensor->data_c()), data_buff_size,
                                   [rel](uint8_t *) -> void { rel.Rel(); });
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << data_buff_size;
      return nullptr;
    }
  }
  MS_LOG(DEBUG) << "Convert Me Tensor to Ge Tensor success!";
  return tensor_ptr;
}

GeTensorPtr TransformUtil::ConvertScalar(const ValuePtr &val) {
  auto ge_tensor = ConvertAnyUtil(val, AnyTraits<ValueAny>());
  return make_shared<GeTensor>(ge_tensor);
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
    case GeDataType::DT_BF16:
      return MeDataType::kNumberTypeBFloat16;
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
                                            const TypeId &me_type, bool ref_mem) {
  MS_EXCEPTION_IF_NULL(ge_tensor);
  MS_EXCEPTION_IF_NULL(ge_tensor->GetData());
  if (ge_tensor->GetSize() == 0) {
    MS_LOG(ERROR) << "GE tensor data size is zero!";
    return nullptr;
  }

  if (ref_mem) {
    void *data = reinterpret_cast<void *>(const_cast<uint8_t *>(ge_tensor->GetData()));
    ssize_t data_size = static_cast<ssize_t>(SizeOf(me_dims));
    ssize_t itemsize = MeTensor(me_type, ShapeVector()).data().itemsize();
    ssize_t ndim = static_cast<ssize_t>(me_dims.size());
    auto ref_data = std::make_shared<TensorRefData>(data, data_size, itemsize, ndim);
    return make_shared<MeTensor>(me_type, me_dims, ref_data);
  } else {
    MeTensor me_tensor(me_type, me_dims);

    // Get the writable data pointer of the tensor and cast it to its data type.
    auto me_data_ptr = me_tensor.data_c();
    size_t me_data_size = static_cast<size_t>(me_tensor.data().nbytes());
    MS_EXCEPTION_IF_NULL(me_data_ptr);
    size_t length = ge_tensor->GetSize();
    if (me_data_size < length) {
      MS_LOG(ERROR) << "ME tensor data size[" << me_data_size << " bytes] is less than GE tensor [" << length
                    << " bytes]";
      return nullptr;
    }

    if (length < SECUREC_MEM_MAX_LEN) {
      int ret_code = memcpy_s(me_data_ptr, length, ge_tensor->GetData(), length);
      if (ret_code != EOK) {
        MS_LOG(ERROR) << "Memcpy_s from ge_tensor to me_tensor failed.";
        return nullptr;
      }
    } else {
      (void)memcpy(me_data_ptr, ge_tensor->GetData(), length);
    }

    return make_shared<MeTensor>(me_tensor);
  }
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
MeTensorPtr TransformUtil::ConvertGeTensor(const GeTensorPtr ge_tensor, const ShapeVector &request_dims, bool ref_mem) {
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
  return GenerateMeTensor(ge_tensor, me_dims, type_id, ref_mem);
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
