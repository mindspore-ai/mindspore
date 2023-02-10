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
#ifndef AICPU_CONTEXT_CPU_PROTO_ATTR_VALUE_IMPL_H
#define AICPU_CONTEXT_CPU_PROTO_ATTR_VALUE_IMPL_H
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cpu_kernel/inc/cpu_tensor.h"
#include "cpu_kernel/inc/cpu_tensor_shape.h"
#include "proto/cpu_attr.pb.h"

namespace aicpu {
class AttrValueImpl {
  friend class CpuKernelUtils;

 public:
  AttrValueImpl(
    aicpuops::AttrValue *attr, std::function<void(aicpuops::AttrValue *)> del_func = [](aicpuops::AttrValue *p) {})
      : attr_value_(attr, del_func) {}

  ~AttrValueImpl() = default;
  AttrValueImpl(const AttrValueImpl &) = delete;
  AttrValueImpl(AttrValueImpl &&) = delete;
  AttrValueImpl &operator=(const AttrValueImpl &) = delete;
  AttrValueImpl &operator=(AttrValueImpl &&) = delete;

  /*
   * get string value of attr.
   * @return string: string value of attr
   */
  std::string GetString() const;

  /*
   * get string list value of attr.
   * @return vector<std::string>: string list value of attr
   */
  std::vector<std::string> GetListString() const;

  /*
   * attr add string value to list.
   * @param string: string value need to add to list
   */
  void AddListString(const std::string &str);

  /*
   * get string list size of attr.
   * @return int32_t: string list size of attr
   */
  int32_t ListStringSize() const;

  /*
   * set string value to attr.
   * @param string: string value need to set to attr
   */
  void SetString(const std::string &byte);

  /*
   * set string list value to attr.
   * @param vector<std::string>: string list value need to set to attr
   */
  void SetListString(const std::vector<std::string> &bytes);

  /*
   * get int value of attr.
   * @return int64_t: int value of attr
   */
  int64_t GetInt() const;

  /*
   * get int list value of attr.
   * @return vector<int64_t>: int list value of attr
   */
  std::vector<int64_t> GetListInt() const;

  /*
   * get int list list value of attr.
   * @return vector<vector<int64_t>>: int list list value of attr
   */
  std::vector<std::vector<int64_t>> GetListListInt() const;

  /*
   * attr add int value to list.
   * @param i: int value need to add to list
   */
  void AddListInt(int64_t i);

  /*
   * get int list size of attr.
   * @return int32_t: int list size of attr
   */
  int32_t ListIntSize() const;

  /*
   * set int value to attr.
   * @param i: int value need to set to attr
   */
  void SetInt(int64_t i);

  /*
   * set int list value to attr.
   * @param vector<int64_t>: int list value need to set to attr
   */
  void SetListInt(const std::vector<int64_t> &list);

  /*
   * set int list list value to attr.
   * @param vector<vector<int64_t>>: int list list value need to set to attr
   */
  void SetListListInt(const std::vector<std::vector<int64_t>> &list);

  /*
   * get float value of attr.
   * @return float: float value of attr
   */
  float GetFloat() const;

  /*
   * get float list value of attr.
   * @return vector<float>: float list value of attr
   */
  std::vector<float> GetListFloat() const;

  /*
   * attr add float value to list.
   * @param f: float value need to add to list
   */
  void AddListFloat(float f);

  /*
   * get float list size of attr.
   * @return int32_t: float list size of attr
   */
  int32_t ListFloatSize() const;

  /*
   * set float value to attr.
   * @param f: float value need to set to attr
   */
  void SetFloat(float f);

  /*
   * set float list value to attr.
   * @param vector<float>: float list value need to set to attr
   */
  void SetListFloat(const std::vector<float> &list);

  /*
   * get bool value of attr.
   * @return bool: bool value of attr
   */
  bool GetBool() const;

  /*
   * get bool list value of attr.
   * @return vector<bool>: bool list value of attr
   */
  std::vector<bool> GetListBool() const;

  /*
   * attr add bool value to list.
   * @param b: bool value need to add to list
   */
  void AddListBool(bool b);

  /*
   * get bool list size of attr.
   * @return int32_t: bool list size of attr
   */
  int32_t ListBoolSize() const;

  /*
   * set bool value to attr.
   * @param b: bool value need to set to attr
   */
  void SetBool(bool b);

  /*
   * set bool list value to attr.
   * @param vector<bool>: bool list value need to set to attr
   */
  void SetListBool(const std::vector<bool> &list);

  /*
   * get data type value of attr.
   * @return DataType: data type value of attr
   */
  DataType GetDataType() const;

  /*
   * get data type list value of attr.
   * @return vector<int32_t>: data type list value of attr
   */
  std::vector<DataType> GetListDataType() const;

  /*
   * attr add data type value to list.
   * @param type: data type value need to add to list
   */
  void AddListDataType(DataType type);

  /*
   * get data type list size of attr.
   * @return int32_t: data type list size of attr
   */
  int32_t ListDataTypeSize() const;

  /*
   * set data type value to attr.
   * @param type: data type value need to set to attr
   */
  void SetDataType(DataType type);

  /*
   * set data type list value to attr.
   * @param vector<DataType>: data type list value need to set to attr
   */
  void SetListDataType(const std::vector<DataType> &list);

  /*
   * set tensor shape value to attr.
   * @param shape: tensor shape value need to set to attr
   * @return bool: true->success false->failed
   */
  bool SetTensorShape(const TensorShape *shape);

  /*
   * set tensor shape list value to attr.
   * @param vector<TensorShape>: tensor shape list value need to set to attr
   * @return uint32_t: success number
   */
  uint32_t SetListTensorShape(const std::vector<TensorShape *> &list);

  /*
   * attr add tensor shape value to list.
   * @return shared_ptr<TensorShape>: tensor shape value ptr added to list
   */
  std::shared_ptr<TensorShape> AddListTensorShape();

  /*
   * get tensor shape value of attr.
   * @return TensorShape: tensor shape value of attr
   */
  std::shared_ptr<TensorShape> GetTensorShape() const;

  /*
   * get tensor shape list value of attr.
   * @return vector<TensorShape>: tensor shape list value of attr
   */
  std::vector<TensorShape> GetListTensorShape() const;

  /*
   * get tensor shape list size of attr.
   * @return int32_t: tensor shape list size of attr
   */
  int32_t ListTensorShapeSize() const;

  /*
   * set tensor value to attr.
   * @param tensor: tensor value need to set to attr
   * @return bool: true->success false->failed
   */
  bool SetTensor(const Tensor *tensor);

  /*
   * set tensor list value to attr.
   * @param vector<Tensor>: tensor list value need to set to attr
   * @return uint32_t: success number
   */
  uint32_t SetListTensor(const std::vector<Tensor *> &list);

  /*
   * attr add tensor value to list.
   * @return shared_ptr<Tensor>: tensor value ptr added to list
   */
  std::shared_ptr<Tensor> AddListTensor();

  /*
   * get tensor value of attr.
   * @return Tensor: tensor value of attr
   */
  std::shared_ptr<Tensor> GetTensor() const;

  /*
   * get tensor list value of attr.
   * @return vector<Tensor>: tensor list value of attr
   */
  std::vector<Tensor> GetListTensor() const;

  /*
   * get tensor list size of attr.
   * @return int32_t: tensor list size of attr
   */
  int32_t ListTensorSize() const;

  /*
   * get attr proto.
   */
  aicpuops::AttrValue *GetProto() const;

 private:
  std::shared_ptr<aicpuops::AttrValue> attr_value_{nullptr};
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_CPU_PROTO_ATTR_VALUE_IMPL_H
