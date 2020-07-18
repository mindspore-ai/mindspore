/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_FBB_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_FBB_H_

/// This header contains some serialize and deserialize functions for tensor row using
/// Google Flatbuffer

#include <memory>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/cache/de_tensor_generated.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/slice.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// \brief Function to serialize TensorRow header used by CacheRowRequest
/// \param row TensorRow
/// \param fbb [in/out] fbb that contains the serialized data
/// \return Status object
Status SerializeTensorRowHeader(const TensorRow &row, std::shared_ptr<flatbuffers::FlatBufferBuilder> *fbb);

/// \brief A function used by BatchFetchRequest to deserialize a flat buffer back to a tensor row.
/// \param col_ts A serialized version of Tensor meta data
/// \param data Tensor data wrapped in a slice
/// \param out Tensor
/// \return Status object
Status RestoreOneTensor(const TensorMetaMsg *col_ts, const ReadableSlice &data, std::shared_ptr<Tensor> *out);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_FBB_H_
