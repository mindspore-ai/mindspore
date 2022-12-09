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
#include "cpu_kernel/cpu_proto/attr_value_impl.h"
#include "cpu_kernel/inc/cpu_attr_value.h"

namespace aicpu {
AttrValue::AttrValue(AttrValueImpl *impl) : impl_(impl) {}

/*
 * get string value of attr.
 */
std::string AttrValue::GetString() const { return impl_->GetString(); }

/*
 * get string list size of attr.
 */
int32_t AttrValue::ListStringSize() const { return impl_->ListStringSize(); }

/*
 * get string list value of attr.
 */
std::vector<std::string> AttrValue::GetListString() const { return impl_->GetListString(); }

/*
 * set string list value to attr.
 */
void AttrValue::SetListString(const std::vector<std::string> &bytes) { impl_->SetListString(bytes); }

/*
 * set string value to attr.
 */
void AttrValue::SetString(const std::string &byte) { impl_->SetString(byte); }

/*
 * attr add string value to list.
 */
void AttrValue::AddListString(const std::string &str) { impl_->AddListString(str); }

/*
 * get int value of attr.
 */
int64_t AttrValue::GetInt() const { return impl_->GetInt(); }

/*
 * get int list value of attr.
 */
std::vector<int64_t> AttrValue::GetListInt() const { return impl_->GetListInt(); }

/*
 * get int list list value of attr.
 */
std::vector<std::vector<int64_t>> AttrValue::GetListListInt() const { return impl_->GetListListInt(); }

/*
 * attr add int value to list.
 */
void AttrValue::AddListInt(int64_t i) { impl_->AddListInt(i); }

/*
 * get int list size of attr.
 */
int32_t AttrValue::ListIntSize() const { return impl_->ListIntSize(); }

/*
 * set int value to attr.
 */
void AttrValue::SetInt(int64_t i) { impl_->SetInt(i); }

/*
 * set int list value to attr.
 */
void AttrValue::SetListInt(const std::vector<int64_t> &i) { impl_->SetListInt(i); }

/*
 * set int list list value to attr.
 */
void AttrValue::SetListListInt(const std::vector<std::vector<int64_t>> &i) { impl_->SetListListInt(i); }

/*
 * get float value of attr.
 */
float AttrValue::GetFloat() const { return impl_->GetFloat(); }

/*
 * get float list value of attr.
 */
std::vector<float> AttrValue::GetListFloat() const { return impl_->GetListFloat(); }

/*
 * attr add float value to list.
 */
void AttrValue::AddListFloat(float f) { impl_->AddListFloat(f); }

/*
 * set float value to attr.
 */
void AttrValue::SetFloat(float f) { impl_->SetFloat(f); }

/*
 * get float list size of attr.
 */
int32_t AttrValue::ListFloatSize() const { return impl_->ListFloatSize(); }

/*
 * set float list value to attr.
 */
void AttrValue::SetListFloat(const std::vector<float> &f) { impl_->SetListFloat(f); }

/*
 * get bool value of attr.
 */
bool AttrValue::GetBool() const { return impl_->GetBool(); }

/*
 * get bool list value of attr.
 */
std::vector<bool> AttrValue::GetListBool() const { return impl_->GetListBool(); }

/*
 * attr add bool value to list.
 */
void AttrValue::AddListBool(bool b) { impl_->AddListBool(b); }

/*
 * get bool list size of attr.
 */
int32_t AttrValue::ListBoolSize() const { return impl_->ListBoolSize(); }

/*
 * set bool value to attr.
 */
void AttrValue::SetBool(bool b) { impl_->SetBool(b); }

/*
 * set bool list value to attr.
 */
void AttrValue::SetListBool(const std::vector<bool> &b) { return impl_->SetListBool(b); }

/*
 * get data type value of attr.
 */
DataType AttrValue::GetDataType() const { return impl_->GetDataType(); }

/*
 * get data type list value of attr.
 */
std::vector<DataType> AttrValue::GetListDataType() const { return impl_->GetListDataType(); }

/*
 * attr add data type value to list.
 */
void AttrValue::AddListDataType(DataType type) { impl_->AddListDataType(type); }

/*
 * get data type list size of attr.
 */
int32_t AttrValue::ListDataTypeSize() const { return impl_->ListDataTypeSize(); }

/*
 * set data type value to attr.
 */
void AttrValue::SetDataType(DataType type) { impl_->SetDataType(type); }

/*
 * set data type list value to attr.
 */
void AttrValue::SetListDataType(const std::vector<DataType> &type) { impl_->SetListDataType(type); }

/*
 * set tensor shape value to attr.
 */
bool AttrValue::SetTensorShape(const TensorShape *shape) { return impl_->SetTensorShape(shape); }

/*
 * set tensor shape list value to attr.
 */
uint32_t AttrValue::SetListTensorShape(const std::vector<TensorShape *> &shape) {
  return impl_->SetListTensorShape(shape);
}

/*
 * attr add tensor shape value to list.
 */
std::shared_ptr<TensorShape> AttrValue::AddListTensorShape() { return impl_->AddListTensorShape(); }

/*
 * get tensor shape value of attr.
 */
std::shared_ptr<TensorShape> AttrValue::GetTensorShape() const { return impl_->GetTensorShape(); }

/*
 * get tensor shape list value of attr.
 */
std::vector<TensorShape> AttrValue::GetListTensorShape() const { return impl_->GetListTensorShape(); }

/*
 * get tensor shape list size of attr.
 */
int32_t AttrValue::ListTensorShapeSize() const { return impl_->ListTensorShapeSize(); }

/*
 * set tensor value to attr.
 */
bool AttrValue::SetTensor(const Tensor *tensor) { return impl_->SetTensor(tensor); }

/*
 * set tensor list value to attr.
 */
uint32_t AttrValue::SetListTensor(const std::vector<Tensor *> &tensor) { return impl_->SetListTensor(tensor); }

/*
 * attr add tensor value to list.
 */
std::shared_ptr<Tensor> AttrValue::AddListTensor() { return impl_->AddListTensor(); }

/*
 * get tensor value of attr.
 */
std::shared_ptr<Tensor> AttrValue::GetTensor() const { return impl_->GetTensor(); }

/*
 * get tensor list value of attr.
 */
std::vector<Tensor> AttrValue::GetListTensor() const { return impl_->GetListTensor(); }

/*
 * get tensor list size of attr.
 */
int32_t AttrValue::ListTensorSize() const { return impl_->ListTensorSize(); }
}  // namespace aicpu
