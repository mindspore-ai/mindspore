/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_TENSOR_INDEX_PY_H_
#define MINDSPORE_CCSRC_UTILS_TENSOR_INDEX_PY_H_

#include <tuple>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "ir/map_tensor.h"
#include "pybind_api/ir/tensor_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/pynative/base.h"

namespace py = pybind11;

namespace mindspore {
namespace tensor {
using tensor::TensorPy;
//
// Tensor index python adapter.
//
const int64_t kIndexMax = std::numeric_limits<int64_t>::max();

enum class TensorIndexType { None = 0, Ellipsis, Integer, Boolean, Slice, Tensor, List, Tuple, Array, Float };
enum class ValueTransferType {
  kUnknown,
  kTensorScatterUpdate,
  kExpandDims,
  kBroadCast,
  kCast,
  kSelect,
  kGather,
  kStrideSlice,
  kStrideSliceWithMask,
  kGatherND,
  kScatterNdUpdate,
  kReshape,
  kSelectView,
  kUnsqueeze,
  kCopyView,
  kScatterND,
  kNumberToTensor,
  kHandleSequenceValue,
  kByPass,
  kReSetItemByIndex,
  kCopySlice,
  kSetItemByBool,
  kEmptyTensor,
  kSetItemByEllipsis,
  kFormatIndexTensor,
  kGetitemByBoolTensor,
  kSetitemByBoolTensor,
  kJustReturn,
  kRaiseIndexError
};

enum class IndexOpType { GetItem = 0, SetItem };

class Slice final {
 public:
  Slice(const py::object &start_index, const py::object &stop_index, const py::object &step_index) {
    dim_size_ = kIndexMax;
    if (py::isinstance<Tensor>(step_index) || IsStubTensor(step_index)) {
      auto step_tensor = IsStubTensor(step_index) ? ConvertStubTensor(step_index) : step_index.cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(step_tensor);
      if (step_tensor->data_type() == kMetaTypeNone) {
        step_ = 1;
      } else {
        step_ = GetTensorData(step_tensor);
      }
    } else if (py::isinstance<py::none>(step_index)) {
      step_ = 1;
    } else if (py::isinstance<py::int_>(step_index)) {
      step_ = step_index.cast<int64_t>();
      if (step_ == 0) {
        MS_EXCEPTION(ValueError) << "For 'StridedSlice', 'strides' cannot contain 0";
      }
      if (step_ < -kIndexMax) {
        step_ = -kIndexMax;
      }
    }
    start_ = NormalizeIndex(start_index, step_, dim_size_);
    stop_ = NormalizeIndex(stop_index, -step_, dim_size_);
    stop_init_by_none_ = InitByNone(stop_index);
    start_init_by_none_ = InitByNone(start_index);
  }

  Slice(int64_t start_index, int64_t stop_index, int64_t step_index, int64_t dim_size, bool start_init_by_none,
        bool stop_init_by_none) {
    dim_size_ = dim_size;
    step_ = step_index;
    if (step_ == 0) {
      MS_EXCEPTION(ValueError) << "For 'StridedSlice', 'strides' cannot contain 0";
    }
    if (step_ < -kIndexMax) {
      step_ = -kIndexMax;
    }

    start_ = NormalizeIndex(start_index, dim_size_);
    stop_ = NormalizeIndex(stop_index, dim_size_);
    start_init_by_none_ = start_init_by_none;
    stop_init_by_none_ = stop_init_by_none;
  }

  // Empty slice (None:None:None) -> (0:DimSize:1)
  Slice() : Slice(0, kIndexMax, 1, kIndexMax, true, true) {}

  Slice(const Slice &slice, int64_t dim_size)
      : Slice(std::min(slice.start_, dim_size), std::min(slice.stop_, dim_size), slice.step_,
              std::min(slice.dim_size_, dim_size), slice.start_init_by_none_, slice.stop_init_by_none_) {}

  static inline int64_t GetTensorData(const TensorPtr &tensor) {
    MS_EXCEPTION_IF_NULL(tensor);
    const auto &device_address = tensor->device_address();
    if (device_address != nullptr) {
      tensor->data_sync();
    }
    if (!tensor->shape().empty()) {
      MS_EXCEPTION(TypeError) << "Only integer scalar tensors can be converted to a scalar index";
    }
    int64_t tensor_value = 0;
    if (tensor->data_type() == kNumberTypeInt32) {
      tensor_value = *static_cast<int32_t *>(tensor->data_c());
    } else if (tensor->data_type() == kNumberTypeInt64) {
      tensor_value = *static_cast<int64_t *>(tensor->data_c());
    }
    return tensor_value;
  }

  inline int64_t start() const { return start_; }

  inline bool start_init_by_none() const { return start_init_by_none_; }

  inline bool stop_init_by_none() const { return stop_init_by_none_; }

  inline int64_t stop() const { return stop_; }

  inline int64_t step() const { return step_; }
  inline int64_t dim_size() const { return dim_size_; }

 private:
  int64_t start_ = 0;
  int64_t stop_ = 0;
  int64_t step_ = 0;
  int64_t dim_size_ = 0;
  bool start_init_by_none_ = false;
  bool stop_init_by_none_ = false;

  static inline int64_t NormalizeIndex(int64_t index, int64_t dim_size) {
    int64_t new_index = index;
    if (dim_size == kIndexMax) {
      return new_index;
    }
    if (new_index < 0) {
      MS_EXCEPTION_IF_ZERO("DimsSize should not be zero", dim_size);
      return new_index < -dim_size ? 0 : (dim_size + (new_index % dim_size)) % dim_size;  // NOLINT
    }
    return new_index < dim_size ? new_index : dim_size;
  }

  static inline int64_t NormalizeIndex(const TensorPtr &index, int64_t step, int64_t dim_size) {
    MS_EXCEPTION_IF_NULL(index);
    if (index->data_type() == kMetaTypeNone) {
      return step > 0 ? 0 : dim_size;
    }
    int64_t new_index = GetTensorData(index);
    if (dim_size == kIndexMax) {
      return new_index;
    }
    if (new_index < 0) {
      MS_EXCEPTION_IF_ZERO("DimsSize should not be zero", dim_size);
      return new_index < -dim_size ? 0 : (dim_size + (new_index % dim_size)) % dim_size;  // NOLINT
    }
    return new_index < dim_size ? new_index : dim_size;
  }

  static inline bool InitByNone(const py::object &index) {
    if (py::isinstance<Tensor>(index)) {
      auto tensor_index = index.cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor_index);
      return tensor_index->data_type() == kMetaTypeNone;
    } else if (IsStubTensor(index)) {
      auto type_id = GetStubTensorInfo(index).second->type_id();
      return type_id == kMetaTypeNone;
    } else if (py::isinstance<py::none>(index)) {
      return true;
    }
    return false;
  }

  static inline int64_t NormalizeIndex(const py::object &index, int64_t step, int64_t dim_size) {
    int64_t normalized_index;
    if (py::isinstance<Tensor>(index) || IsStubTensor(index)) {
      auto tensor_index = IsStubTensor(index) ? ConvertStubTensor(index) : index.cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor_index);
      normalized_index = NormalizeIndex(tensor_index, step, dim_size);
    } else if (py::isinstance<py::int_>(index)) {
      normalized_index = NormalizeIndex(index.cast<int64_t>(), dim_size);
    } else if (py::isinstance<py::none>(index)) {
      normalized_index = step > 0 ? 0 : dim_size;
    } else {
      MS_LOG(EXCEPTION) << "Slice index type must be int, tensor or none.";
    }
    return normalized_index;
  }
  friend inline std::ostream &operator<<(std::ostream &out, const Slice &slice) {
    return out << "start: " << slice.start_ << ","
               << "stop: " << slice.stop_ << ","
               << "step: " << slice.step_;
  }
};

class TensorIndex final {
 public:
  explicit TensorIndex(const py::none &) : type_(TensorIndexType::None) {}

  explicit TensorIndex(const py::ellipsis &) : type_(TensorIndexType::Ellipsis) {}

  explicit TensorIndex(int64_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
  explicit TensorIndex(int integer) : TensorIndex((int64_t)integer) {}
  explicit TensorIndex(const py::int_ &integer) : TensorIndex(integer.cast<int64_t>()) {}

  explicit TensorIndex(bool boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}
  explicit TensorIndex(const py::bool_ &boolean) : TensorIndex(py::cast<bool>(boolean)) {}

  explicit TensorIndex(const Slice &slice) : slice_(slice), type_(TensorIndexType::Slice) {}
  explicit TensorIndex(const py::slice &py_slice)
      : TensorIndex(Slice(py_slice.attr("start"), py_slice.attr("stop"), py_slice.attr("step"))) {}

  explicit TensorIndex(TensorPtr tensor) : tensor_(std::move(tensor)), type_(TensorIndexType::Tensor) {}
  explicit TensorIndex(py::array py_array) : array_(std::move(py_array)), type_(TensorIndexType::Array) {}

  explicit TensorIndex(py::list py_list) : list_(std::move(py_list)), type_(TensorIndexType::List) {}
  explicit TensorIndex(py::tuple py_tuple) : tuple_(std::move(py_tuple)), type_(TensorIndexType::Tuple) {}

  explicit TensorIndex(float float_input) : float_(float_input), type_(TensorIndexType::Float) {}
  explicit TensorIndex(const py::float_ &float_input) : TensorIndex(float_input.cast<float>()) {}

  explicit TensorIndex(const py::handle &py_object) {
    if (py::isinstance<py::list>(py_object)) {
      this->list_ = py_object.cast<py::list>();
      this->type_ = TensorIndexType::List;
    } else if (py::isinstance<py::int_>(py_object) && !py::isinstance<py::bool_>(py_object)) {
      this->integer_ = py_object.cast<py::int_>();
      this->type_ = TensorIndexType::Integer;
    } else if (py::isinstance<py::float_>(py_object)) {
      this->float_ = py_object.cast<py::float_>();
      this->type_ = TensorIndexType::Float;
    } else if (py::isinstance<tensor::Tensor>(py_object)) {
      this->tensor_ = py_object.cast<tensor::TensorPtr>();
      this->type_ = TensorIndexType::Tensor;
    } else if (py::isinstance<py::tuple>(py_object)) {
      this->tuple_ = py_object.cast<py::tuple>();
      this->type_ = TensorIndexType::Tuple;
    } else if (py::isinstance<py::slice>(py_object)) {
      this->slice_ = TensorIndex(py_object.cast<py::slice>()).slice_;
      this->type_ = TensorIndexType::Slice;
    } else if (py::isinstance<py::ellipsis>(py_object)) {
      this->type_ = TensorIndexType::Ellipsis;
    } else if (py::isinstance<py::none>(py_object)) {
      this->type_ = TensorIndexType::None;
    } else if (py::isinstance<py::array>(py_object)) {
      this->array_ = py_object.cast<py::array>();
      this->type_ = TensorIndexType::Array;
    } else if (py::isinstance<py::bool_>(py_object)) {
      this->boolean_ = py_object.cast<py::bool_>();
      this->type_ = TensorIndexType::Boolean;
    } else if (IsStubTensor(py_object)) {
      this->tensor_ = ConvertStubTensor(py_object);
      this->type_ = TensorIndexType::Tensor;
    }
  }

  inline bool IsNone() const { return type_ == TensorIndexType::None; }

  inline bool IsEllipsis() const { return type_ == TensorIndexType::Ellipsis; }

  inline bool IsInteger() const { return type_ == TensorIndexType::Integer; }

  inline int64_t integer() const { return integer_; }

  inline bool IsBoolean() const { return type_ == TensorIndexType::Boolean; }

  inline bool boolean() const { return boolean_; }

  inline bool IsSlice() const { return type_ == TensorIndexType::Slice; }

  inline const Slice &slice() const { return slice_; }

  inline bool IsTensor() const { return type_ == TensorIndexType::Tensor; }

  inline const TensorPtr &tensor() const { return tensor_; }

  inline bool IsList() const { return type_ == TensorIndexType::List; }

  inline const py::list &list() const { return list_; }

  inline bool IsTuple() const { return type_ == TensorIndexType::Tuple; }

  inline const py::tuple &tuple() const { return tuple_; }

  inline bool IsSequence() const { return IsList() || IsTuple(); }

  inline const py::array &array() const { return array_; }

  inline bool IsArray() const { return type_ == TensorIndexType::Array; }

  inline const float &floating_point() const { return float_; }

  inline bool IsFloat() const { return type_ == TensorIndexType::Float; }

  inline const TensorIndexType &type() const { return type_; }

  static py::object GetItemByTensor(const ShapeVector &data_shape, const TensorPtr &index);
  static py::object GetItemByList(const ShapeVector &data_shape, const TensorIndex &tensor_index);
  static py::object GetItemByTuple(const ShapeVector &data_shape, const std::vector<TensorIndex> &tensor_indexes);
  static bool GetItemByTupleWithView(const ValuePtr &data_value, const ShapeVector &data_shape,
                                     const py::object &py_index, std::vector<int64_t> *data_transfer_types,
                                     std::vector<py::object> *data_transfer_args, const TypePtr &data_type);
  static py::object GetItemByBool(const ValuePtr &data_value, const ShapeVector &data_shape, bool index);
  static py::object GetItemByNumber(const ShapeVector &data_shape, int64_t index);
  static py::object GetItemByNumberWithView(const ValuePtr &data_value, const ShapeVector &data_shape, int64_t index);
  static py::object GetItemBySlice(const ValuePtr &data_value, const ShapeVector &data_shape,
                                   const TensorIndex &py_index);
  static py::object GetItemIndexSimpleIndex(const py::object &py_index, const ValuePtr &data_value,
                                            const ShapeVector &data_shape);
  static py::object GetItemIndexInfo(const py::object &data, const py::object &index, const py::bool_ &is_ascend);

  static py::object SetItemByNumber(const ShapeVector &data_shape, const TypePtr &data_type, bool is_parameter,
                                    const TensorIndex &tensor_index, const TensorIndexType &py_value_type);
  static py::object SetItemByNumberWithView(const ShapeVector &data_shape, const TypePtr &data_type, bool is_parameter,
                                            const TensorIndex &tensor_index, const TensorIndexType &py_value_type,
                                            const ValuePtr &data_value);
  static py::object SetItemByTensor(const ShapeVector &data_shape, bool is_parameter, const TensorIndex &tensor_index,
                                    const TensorIndexType &py_value_type);

  static py::object SetItemByTuple(const ShapeVector &data_shape, const TypePtr &data_type, const TensorIndex &py_index,
                                   const TensorIndexType &py_value_type);

  static py::object SetItemBySlice(const ShapeVector &data_shape, const TypePtr &data_type, const TensorIndex &py_index,
                                   const TensorIndexType &py_value_type, const ValuePtr &data_value);

  static py::object SetItemIndexInfo(const py::object &data, const py::object &index, const py::object &value,
                                     const py::bool_ &is_ascend);

  static py::object SetItemIndexByIndexType(const TensorIndex &index, const py::object &py_index,
                                            const ShapeVector &data_shape, const TypePtr &data_type,
                                            const TensorIndexType &value_type, bool is_parameter);
  static py::handle py_index_handle_;
  static py::handle py_value_handle_;
  static bool is_ascend_;
  static py::module np_module_;
  static IndexOpType index_op_type_;

 private:
  int64_t integer_ = 0;
  bool boolean_ = false;
  float float_ = 0.0;
  Slice slice_;
  TensorPtr tensor_;
  py::array array_;
  py::list list_;
  py::tuple tuple_;
  TensorIndexType type_;

  // ***********************************************utils*******************************************
  static void CheckGetItemIndex(const TensorIndexType &index_data_type);
  static void CheckSetItemIndex(const TensorIndexType &index_data_type, const TensorIndexType &value_data_type);
  template <typename T>
  static inline bool CheckTypeIsInstance(const T &type, const std::vector<T> &target_types) {
    return std::any_of(target_types.begin(), target_types.end(),
                       [&type](const auto &target_type) { return target_type == type; });
  }
  static inline void JudgeDataDim(int64_t data_dim, int64_t min_data_dim, int64_t max_data_dim) {
    if (data_dim < min_data_dim || data_dim > max_data_dim) {
      MS_EXCEPTION(ValueError) << "The input data's dim must in the range of [" << min_data_dim << ", " << max_data_dim
                               << "], but got '" << data_dim << "'.";
    }
  }
  template <typename T>
  static inline py::tuple VectorToPyTuple(const std::vector<T> &item_shape) {
    size_t tuple_size = item_shape.size();
    py::tuple out(tuple_size);
    for (size_t i = 0; i < tuple_size; i++) {
      out[i] = item_shape[i];
    }
    return out;
  }
  static ShapeVector BroadCastShape(const ShapeVector &x_shape, const ShapeVector &y_shape);
  static ShapeVector BroadCastShape(const std::vector<ShapeVector> &tensor_indexes_shapes) {
    if (tensor_indexes_shapes.empty()) {
      return {};
    }
    return std::accumulate(tensor_indexes_shapes.begin(), tensor_indexes_shapes.end(), tensor_indexes_shapes[0],
                           [](const auto &output_shape, const auto &tensor_indexes_shape) {
                             return BroadCastShape(output_shape, tensor_indexes_shape);
                           });
  }
  static std::vector<int64_t> SliceToVector(int64_t start, int64_t stop, int64_t step) {
    std::vector<int64_t> slice_ele_list_index;
    if (step > 0) {
      for (int64_t j = start; j < stop; j += step) {
        (void)slice_ele_list_index.emplace_back(j);
      }
      return slice_ele_list_index;
    }
    for (int64_t j = start; j > stop; j += step) {
      (void)slice_ele_list_index.emplace_back(j);
    }
    return slice_ele_list_index;
  }

  // This is the c++ version of sequence_to_index in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_constexpr_utils.py"
  // Transforms sequence to tensor.
  static inline TensorIndex SequenceToTensor(const TensorIndex &tensor_index, int64_t dim_size) {
    return tensor_index.type_ == TensorIndexType::List ? SequenceToTensor<py::list>(tensor_index.list_, dim_size)
                                                       : SequenceToTensor<py::tuple>(tensor_index.tuple_, dim_size);
  }
  template <typename T>
  static TensorIndex SequenceToTensor(const T &sequence, int64_t dim_size);
  static py::object Unpack(const py::object &x);
  static inline py::object CheckRange(const py::object &x, int64_t dim_size) {
    if (py::isinstance<py::int_>(x)) {
      auto temp_x = x.cast<int64_t>();
      if (temp_x >= dim_size || temp_x < -dim_size) {
        MS_EXCEPTION(IndexError) << "index " << temp_x << " out of bounds for dimension with size " << dim_size;
      }
      MS_EXCEPTION_IF_ZERO("dim_size", dim_size);
      return py::int_(CheckRange(temp_x, dim_size));
    }
    return x;
  }
  static inline int64_t CheckRange(int64_t x, int64_t dim_size) {
    MS_EXCEPTION_IF_ZERO("dim_size", dim_size);
    return (dim_size + (x % dim_size)) % dim_size;
  }

  static bool CheckScalarValue(const py::handle &value) {
    if (py::isinstance<Tensor>(value)) {
      TensorPtr data = value.cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(data);
      auto data_shape = data->shape();
      return data_shape.empty();
    }
    if (IsStubTensor(value)) {
      auto data_shape = GetStubTensorInfo(value).first;
      return data_shape.empty();
    }
    return CheckTypeIsInstance(TensorIndex(value).type(),
                               {TensorIndexType::Float, TensorIndexType::Integer, TensorIndexType::Boolean});
  }

  static py::object DeepList(const py::object &array_like, int64_t dim_size);
  static py::object DeepTensorToNdArray(const py::object &array_like);
  static py::array MakeNdArray(const py::object &a, int64_t dim_size);

  // This is the c++ version of _transform_ellipsis_to_slice in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_compile_utils.py"
  // Converts slice index into array
  static std::vector<TensorIndex> TransformEllipsisToSlice(const ShapeVector &data_shape,
                                                           const std::vector<TensorIndex> &indices);
  static std::tuple<ShapeVector, ShapeVector, ShapeVector, int64_t> GenerateIndexInfoFromTupleOfMixedTensors(
    const std::vector<int64_t> &tensor_positions, const std::vector<ShapeVector> &tensor_indexes_shapes,
    const ShapeVector &slice_shapes, const TensorIndex &py_fancy_position);
  // This is the c++ version of slice2indices in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_constexpr_utils.py"
  // Converts slice index into array
  static TensorIndex SliceToArray(const TensorIndex &tensor_index, const ShapeVector &shape);

  // This is the c++ version of convert_slice_to_tensor in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_compile_utils.py"
  // Converts slice index into array
  static TensorIndex SliceToArray(const TensorPtr &index, const ShapeVector &final_shape, size_t slice_cnt,
                                  const ShapeVector &broadcast_shape, const ShapeVector &slice_shape,
                                  int64_t fancy_position);

  static ShapeVector ComputeSliceShape(const ShapeVector &slice_shape, size_t broadcast_shape_len, size_t slice_cnt,
                                       int64_t fancy_position) {
    ShapeVector shape(slice_shape.size(), 1);
    if (slice_cnt >= shape.size()) {
      MS_EXCEPTION(IndexError) << "Index out of shape size.";
    }
    shape[slice_cnt] = slice_shape[slice_cnt];
    ShapeVector temp_shape(broadcast_shape_len, 1);
    (void)shape.insert(shape.begin() + fancy_position, temp_shape.begin(), temp_shape.end());
    return shape;
  }

  static ShapeVector ComputeMultiples(const ShapeVector &origin_shape, const ShapeVector &broadcast_shape) {
    int64_t len_gap = SizeToLong(broadcast_shape.size()) - SizeToLong(origin_shape.size());
    ShapeVector output_shape = broadcast_shape;
    (void)std::transform(broadcast_shape.begin() + len_gap, broadcast_shape.end(), origin_shape.begin(),
                         output_shape.begin() + len_gap, [](int64_t x, int64_t y) {
                           MS_EXCEPTION_IF_ZERO("dim of data shape", y);
                           return x / y;
                         });
    return output_shape;
  }

  static ShapeVector GeneratePaddingShape(const ShapeVector &shape, int64_t length) {
    if (SizeToLong(shape.size()) > length) {
      MS_EXCEPTION(ValueError) << "Can not pad " << shape << " to length " << length;
    }
    ShapeVector pad_shape(length - SizeToLong(shape.size()), 1);
    (void)pad_shape.insert(pad_shape.begin(), shape.begin(), shape.end());
    return pad_shape;
  }
  static py::object BroadCastTo(const ShapeVector &broadcast_shape, const py::object &item);

  // This is the c++ version of _transform_indexing_tensor in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_compile_utils.py"
  //  BroadCast tensor to the required
  static TensorIndex BroadCastTensor(const ShapeVector &broadcast_shape, const ShapeVector &final_shape,
                                     const ShapeVector &new_shape, const TensorPtr &item);
  static constexpr int64_t set_item_by_one_tensor = 0;
  static constexpr int64_t set_item_by_tuple_tensor = 1;
  static constexpr int64_t set_item_by_non_tensor = 2;
  static constexpr int64_t int32_bytes_number = 4;
  static std::tuple<int64_t, py::object, ShapeVector> GetValueTransferType(const TensorIndexType &py_value_type,
                                                                           int64_t op_type, const TypePtr &data_type,
                                                                           bool is_view);

  // This is the c++ version of format_tuple_indices in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_compile_utils.py"
  // Format tuple indices by unpacking high-dimension tuple and removing expand
  // dimension signs(Bool and None).
  static inline TensorIndex UnpackTuple(const TensorIndex &tensor_index) {
    return tensor_index.type_ == TensorIndexType::List ? UnpackTuple<py::list>(tensor_index.list_)
                                                       : UnpackTuple<py::tuple>(tensor_index.tuple_);
  }

  // Expand tuple TensorIndex to std::vector<TensorIndex>
  inline std::vector<TensorIndex> ExpandToVector() const {
    std::vector<TensorIndex> output;
    if (type_ == TensorIndexType::Tuple) {
      output.reserve(tuple_.size());
      for (auto const &e : tuple_) {
        (void)output.emplace_back(TensorIndex(e));
      }
    } else {
      output.reserve(list_.size());
      for (auto const &e : list_) {
        (void)output.emplace_back(TensorIndex(e));
      }
    }
    return output;
  }

  template <typename T>
  static TensorIndex UnpackTuple(const T &sequence);

  static inline bool UseCopySlice(const std::vector<TensorIndex> &indices, int64_t data_dims) {
    if (indices.size() >= 2 && data_dims >= 2) {
      bool valid = indices[0].IsInteger() && indices[1].IsSlice() && indices[1].slice().step() == 1;
      return valid && std::all_of(indices.begin() + 2, indices.end(), [](const TensorIndex &x) {
               return x.IsSlice() && x.slice().start_init_by_none() && x.slice().stop_init_by_none() &&
                      x.slice().step() == 1;
             });
    }
    return false;
  }

  // ***********************************************for get_item*******************************************
  // This is the c++ version of get_stride_info_from_tuple in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_constexpr_utils.py"
  // Get stride info from a tuple
  static py::tuple GenerateNonZeroIndex(const ShapeVector &data_shape, const TensorPtr &tensor_index, bool check_align);
  static std::vector<TensorPtr> GenerateNonZeroIndexTensorList(const ShapeVector &data_shape,
                                                               const TensorPtr &tensor_index, bool check_align);
  static std::tuple<std::vector<std::vector<int64_t>>, std::vector<int64_t>> GetStrideInfoFromTuple(
    const ShapeVector &data_shape, const std::vector<TensorIndex> &tuple_index);
  static bool TensorGetitemByTupleParseTensorIndex(const ShapeVector &data_shape, const TensorPtr &tensor_index,
                                                   std::vector<TensorPtr> *tuple_index_new,
                                                   std::vector<TensorPtr> *tensor_indexes,
                                                   std::vector<int64_t> *tensor_positions, bool check_align);
  static std::tuple<bool, ShapeVector, std::vector<TensorIndex>> GetExpandDimsInfo(
    const ShapeVector &data_shape, const std::vector<TensorIndex> &index);
  static py::object GenerateIndices(const std::vector<TensorPtr> &tuple_index_new,
                                    const std::vector<int64_t> &broadcast_shape,
                                    const std::vector<int64_t> &index_tensor_new_shape,
                                    const std::vector<int64_t> &final_shape,
                                    const std::vector<int64_t> &tensor_positions,
                                    const std::vector<int64_t> &slice_shapes, int64_t fancy_position);
  static py::object TensorGetitemByTuple(const ShapeVector &data_shape, const std::vector<TensorIndex> &tuple_index,
                                         std::vector<int64_t> *data_transfer_type,
                                         std::vector<py::object> *data_transfer_args);

  // ***********************************************for set_item*******************************************
  // This is the c++ version of format_list_indices in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_compile_utils.py"
  // Convert list indices to array or list indices based on its contents.
  static TensorIndex FormatList(const TensorIndex &tensor_index, int64_t length);
  static TensorPtr IntToTensor(int64_t i, const ShapeVector &shape);
  static py::object GenerateIndicesFromTupleOfTensor(const ShapeVector &data_shape,
                                                     const std::vector<TensorIndex> &tuple_index,
                                                     ShapeVector *output_index_shape, py::object *data_transfer_arg);
  static void RemNotExpandedDims(int64_t *idx_advanced, bool expand_true, int64_t tensor_index_ndim, int64_t rem_ndim,
                                 std::vector<bool> *not_expanded_dim);

  static inline ShapeVector FilterExpandedDims(const ShapeVector &shape, const std::vector<bool> &not_expanded_dim) {
    int64_t diff = SizeToLong(not_expanded_dim.size()) - SizeToLong(shape.size());
    if (diff < 0) {
      MS_EXCEPTION(ValueError) << "Input array must have the same size across all dimensions.";
    }
    std::vector<int64_t> res;
    size_t index = std::min(shape.size(), not_expanded_dim.size() - static_cast<size_t>(diff));
    for (size_t i = 0; i < index; i++) {
      if (not_expanded_dim[(i + static_cast<size_t>(diff))]) {
        (void)res.emplace_back(shape[i]);
      }
    }
    return res;
  }

  // This is the c++ version of format_index in
  // "mindspore/python/mindspore/ops/composite/multitype_ops/_compile_utils.py"
  // Converts advanced index into array
  static TensorIndex FormatIndex(const TensorIndex &idx, const ShapeVector &data_shape, size_t cur_dim,
                                 bool *need_format);
  static bool RemoveExpandedDimsParseTensorIndex(const ShapeVector &data_shape, const TensorPtr &index_out,
                                                 std::vector<TensorIndex> *indices_out,
                                                 std::vector<ShapeVector> *shapes, bool *has_sequence, size_t *cur_dim,
                                                 bool check_align);
  static std::pair<std::vector<TensorIndex>, ShapeVector> RemoveExpandedDims(
    const std::vector<TensorIndex> &indices, const ShapeVector &data_shape, const ShapeVector &value_shape,
    std::vector<int64_t> *value_transfer_type, std::vector<py::object> *value_transfer_args, int64_t *idx_advanced,
    bool *by_pass, std::vector<size_t> *format_index, std::vector<int64_t> *format_dim);
  static py::object GenerateIndicesFromTuple(const ShapeVector &data_shape, const std::vector<TensorIndex> &tuple_index,
                                             int64_t py_fancy_position, bool *by_pass, ShapeVector *output_index_shape,
                                             py::object *data_transfer_arg);
  static py::object ReSetitemByTensor(const std::vector<TensorIndex> &new_tuple_index,
                                      const std::vector<int64_t> &value_transfer_types,
                                      const std::vector<py::object> &value_transfer_args);
  static py::object SetitemByTupleWithTensor(const ShapeVector &data_shape, const std::vector<TensorIndex> &indices,
                                             const ShapeVector &value_shape, std::vector<int64_t> *value_transfer_type,
                                             std::vector<py::object> *value_transfer_args);

  static py::object SetitemBySliceWithTensor(const ShapeVector &data_shape, const TensorIndex &slice_index,
                                             std::vector<int64_t> *value_transfer_type,
                                             std::vector<py::object> *value_transfer_args, const ValuePtr &data_value,
                                             const TypePtr &data_type);

  static py::array SetItemByTensorByBool(const ShapeVector &data_shape, const TensorPtr &index, int64_t data_dims,
                                         std::vector<int64_t> *value_transfer_types,
                                         std::vector<py::object> *value_transfer_args,
                                         ValueTransferType *tensor_update_type);

  friend std::ostream &operator<<(std::ostream &stream, const TensorIndex &tensor_index);
  friend std::ostream &operator<<(std::ostream &stream, const std::vector<TensorIndex> &tensor_indices);
};
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_TENSOR_INDEX_PY_H_
