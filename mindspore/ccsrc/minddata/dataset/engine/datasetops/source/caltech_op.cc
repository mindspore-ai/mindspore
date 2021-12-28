/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
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
#include "minddata/dataset/engine/datasetops/source/caltech_op.h"

#include <map>
#include <memory>
#include <set>
#include <utility>

namespace mindspore {
namespace dataset {
const std::set<std::string> kExts = {".jpg", ".JPEG"};
const std::map<std::string, int32_t> kClassIndex = {};
CaltechOp::CaltechOp(int32_t num_workers, const std::string &file_dir, int32_t queue_size, bool do_decode,
                     std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : ImageFolderOp(num_workers, file_dir, queue_size, false, do_decode, kExts, kClassIndex, std::move(data_schema),
                    std::move(sampler)) {}
}  // namespace dataset
}  // namespace mindspore
