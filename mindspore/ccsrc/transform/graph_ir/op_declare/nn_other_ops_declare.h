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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_

#include "op_proto/inc/nn_other.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"

// InitPartitionMap
DECLARE_OP_ADAPTER(InitPartitionMap)
DECLARE_OP_USE_OUTPUT(InitPartitionMap)

// InitEmbeddingHashmap
DECLARE_OP_ADAPTER(InitEmbeddingHashmap)
DECLARE_OP_USE_OUTPUT(InitEmbeddingHashmap)

// EmbeddingTableImport
DECLARE_OP_ADAPTER(EmbeddingTableImport)
DECLARE_OP_USE_OUTPUT(EmbeddingTableImport)

// EmbeddingTableFind
DECLARE_OP_ADAPTER(EmbeddingTableFind)
DECLARE_OP_USE_OUTPUT(EmbeddingTableFind)

// EmbeddingTableFindAndInit
DECLARE_OP_ADAPTER(EmbeddingTableFindAndInit)
DECLARE_OP_USE_OUTPUT(EmbeddingTableFindAndInit)

// EmbeddingApplyFtrl
DECLARE_OP_ADAPTER(EmbeddingApplyFtrl)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyFtrl)

// EmbeddingApplyAdam
DECLARE_OP_ADAPTER(EmbeddingApplyAdam)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyAdam)

// EmbeddingApplyAdamW
DECLARE_OP_ADAPTER(EmbeddingApplyAdamW)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyAdamW)

// EmbeddingApplyAdaGrad
DECLARE_OP_ADAPTER(EmbeddingApplyAdaGrad)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyAdaGrad)

// EmbeddingComputeVarImport
DECLARE_OP_ADAPTER(EmbeddingComputeVarImport)
DECLARE_OP_USE_OUTPUT(EmbeddingComputeVarImport)

// EmbeddingComputeVarExport
DECLARE_OP_ADAPTER(EmbeddingComputeVarExport)
DECLARE_OP_USE_OUTPUT(EmbeddingComputeVarExport)

// EmbeddingTableExport
DECLARE_OP_ADAPTER(EmbeddingTableExport)
DECLARE_OP_USE_OUTPUT(EmbeddingTableExport)

// FakeRemoteLookupUniqued
DECLARE_OP_ADAPTER(FakeRemoteLookupUniqued)
DECLARE_OP_USE_OUTPUT(FakeRemoteLookupUniqued)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_
