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
#ifndef DATASET_CORE_CLIENT_H_
#define DATASET_CORE_CLIENT_H_

// client.h
// Include file for DE client functions

#include "dataset/core/constants.h"
#include "dataset/core/data_type.h"
#include "dataset/core/tensor.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/data_schema.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/engine/datasetops/batch_op.h"
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/engine/datasetops/device_queue_op.h"
#include "dataset/engine/datasetops/map_op.h"
#include "dataset/engine/datasetops/project_op.h"
#include "dataset/engine/datasetops/rename_op.h"
#include "dataset/engine/datasetops/repeat_op.h"
#include "dataset/engine/datasetops/shuffle_op.h"
#include "dataset/engine/datasetops/source/generator_op.h"
#include "dataset/engine/datasetops/source/mindrecord_op.h"
#include "dataset/engine/datasetops/source/storage_op.h"
#include "dataset/engine/datasetops/source/tf_reader_op.h"
#include "dataset/engine/datasetops/zip_op.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
// This is a one-time global initializer that needs to be called at the
// start of any minddata applications.
extern Status GlobalInit();
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_CORE_CLIENT_H_
