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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CLIENT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CLIENT_H_

// client.h
// Include file for DE client functions

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/dataset_iterator.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#endif

#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/datasetops/barrier_op.h"
#include "minddata/dataset/engine/datasetops/filter_op.h"
#include "minddata/dataset/engine/datasetops/source/generator_op.h"
#include "minddata/dataset/engine/datasetops/build_vocab_op.h"
#include "minddata/dataset/engine/datasetops/build_sentence_piece_vocab_op.h"
#endif

#include "minddata/dataset/engine/datasetops/batch_op.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/datasetops/project_op.h"
#include "minddata/dataset/engine/datasetops/rename_op.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/datasetops/take_op.h"
#include "minddata/dataset/engine/datasetops/zip_op.h"
#include "minddata/dataset/engine/datasetops/concat_op.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// This is a one-time global initializer that needs to be called at the
// start of any minddata applications.
extern Status GlobalInit();
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CLIENT_H_
