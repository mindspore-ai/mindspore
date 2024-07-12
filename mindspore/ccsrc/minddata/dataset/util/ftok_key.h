/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_FTOK_KEY_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_FTOK_KEY_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
#include <sys/types.h>
#include <sys/ipc.h>
#endif

#include "minddata/dataset/api/python/python_mp.h"
#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/core/shared_memory_queue.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/map_op/map_job.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/wait_post.h"
#ifndef BUILD_LITE
#include "mindspore/core/utils/file_utils.h"
namespace platform = mindspore;
#else
#include "mindspore/lite/src/common/file_utils.h"
namespace platform = mindspore::lite;
#endif

namespace mindspore {
namespace dataset {
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
extern std::atomic<uint32_t> inc_id;
Status GetKey(key_t *key);
#endif
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_FTOK_KEY_H_
