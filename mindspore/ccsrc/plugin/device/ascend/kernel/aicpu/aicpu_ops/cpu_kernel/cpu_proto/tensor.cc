/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/inc/cpu_tensor.h"
#include "cpu_kernel/cpu_proto/tensor_impl.h"

namespace aicpu {
Tensor::Tensor(TensorImpl *impl) : impl_(impl) {}

/*
 * get tensor shape value of tensor.
 */
std::shared_ptr<TensorShape> Tensor::GetTensorShape() const { return impl_->GetTensorShape(); }

/*
 * set tensor shape value to tensor.
 */
bool Tensor::SetTensorShape(const TensorShape *shape) { return impl_->SetTensorShape(shape); }

/*
 * get data type value of tensor.
 */
DataType Tensor::GetDataType() const { return impl_->GetDataType(); }

/*
 * set data type value to tensor.
 */
void Tensor::SetDataType(DataType type) { impl_->SetDataType(type); }

/*
 * get data ptr of tensor.
 */
void *Tensor::GetData() const { return impl_->GetData(); }

/*
 * set data ptr to tensor.
 */
void Tensor::SetData(void *addr) { impl_->SetData(addr); }

/*
 * get data size of tensor.
 */
uint64_t Tensor::GetDataSize() const { return impl_->GetDataSize(); }

/*
 * set data size to tensor.
 */
void Tensor::SetDataSize(uint64_t size) { impl_->SetDataSize(size); }

/*
 * calculate data size by tensor shape.
 */
int64_t Tensor::CalcDataSizeByShape() const { return impl_->CalcDataSizeByShape(); }

/*
 * get data elements number.
 */
int64_t Tensor::NumElements() const { return impl_->NumElements(); }
}  // namespace aicpu