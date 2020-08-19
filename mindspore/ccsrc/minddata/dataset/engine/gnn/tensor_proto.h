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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_TENSOR_PROTO_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_TENSOR_PROTO_H_

#include <deque>
#include <memory>
#include <vector>

#include "proto/gnn_tensor.pb.h"
#include "minddata/dataset/core/tensor.h"

namespace mindspore {
namespace dataset {

Status TensorToPb(const std::shared_ptr<Tensor> tensor, TensorPb *tensor_pb);

Status PbToTensor(const TensorPb *tensor_pb, std::shared_ptr<Tensor> *tensor);

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_TENSOR_PROTO_H_
