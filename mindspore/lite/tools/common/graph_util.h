/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_GRAPH_UTIL_H
#define MINDSPORE_LITE_TOOLS_COMMON_GRAPH_UTIL_H

#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <limits>
#include <functional>
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/graph_util.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
using STATUS = int;
enum InsertPlace { kBefore, kAfter };

using NodeIter = std::vector<std::unique_ptr<schema::CNodeT>>::iterator;

using OpDefCopyer = std::function<std::unique_ptr<schema::CNodeT>(schema::CNodeT *)>;

OpDefCopyer GetSimpleOpCopyer();

int SetFuncGraphOutput(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &outputs);

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx, int inputIndexIdx = -1);

std::vector<size_t> GetInputNodeIdx(const schema::MetaGraphT &graphT, const schema::CNodeT &node,
                                    int inputIndexIdx = -1);

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const size_t &nodeIdx, int outputIndexIdx = -1);

std::vector<size_t> GetOutputNodeIdx(const schema::MetaGraphT &graphT, const schema::CNodeT &node,
                                     int output_index_idx = -1);

std::vector<size_t> GetLinkedPreIdx(const schema::MetaGraphT &graphT, const size_t &tensorIdx);

std::vector<size_t> GetLinkedPostIdx(const schema::MetaGraphT &graphT, const size_t &tensor_idx);

void ReplaceOutput(const uint32_t &old_index, const uint32_t &new_index, schema::MetaGraphT *graphT);

STATUS IsolateNode(schema::MetaGraphT *subGraph, schema::CNodeT *node);

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, size_t nodeIdx, bool removeTensor = true);

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, size_t subGraphIdx, size_t nodeIdx, bool removeTensor = true);

STATUS IsolateOneWayNode(schema::MetaGraphT *graphT, schema::CNodeT *node, bool removeTensor = true);

STATUS UpdateNodeIndex(schema::CNodeT *node, uint32_t deleteIdx);

STATUS RemoveTensor(schema::MetaGraphT *graphT, std::vector<uint32_t> toDeleteTensorIdxes, bool forceDelete = false);

STATUS AddTensor2Node(schema::MetaGraphT *graphT, uint32_t nodeIdx, std::unique_ptr<schema::TensorT> tensor,
                      InsertPlace place = kBefore);

STATUS ReplaceTensorOfNode(schema::MetaGraphT *graphT, uint32_t nodeIdx, uint32_t inTensorIdx,
                           std::unique_ptr<schema::TensorT> tensor);

int DoBitPack(const int &bit_num, schema::TensorT *tensor_input);

NodeIter InsertNode(schema::MetaGraphT *graphT, uint32_t existNodeIdx, InsertPlace place, size_t inoutIndex,
                    std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer = GetSimpleOpCopyer());

NodeIter InsertNode(schema::MetaGraphT *graphT, NodeIter existNodeIter, InsertPlace place, size_t inoutIndexIdx,
                    std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer = GetSimpleOpCopyer());

NodeIter InsertNodeBefore(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t inputIndexIdx,
                          std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                          const OpDefCopyer &opDefCopyer);

NodeIter InsertNodeAfter(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t outputIndexIdx,
                         std::unique_ptr<schema::CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                         const OpDefCopyer &opDefCopyery);

STATUS ValidateFileStr(const std::string &modelFile, const std::string &fileType);

STATUS SetSubgraphTensorIndices(schema::MetaGraphT *meta_graphT);

std::string GetModelName(const std::string &modelFile);

std::vector<int> GetTransposePerm(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &cnode);

std::string BoolVectorToString(const std::vector<bool> &bool_vec);

TypeId GetAbstractTensorDtype(const abstract::AbstractTensorPtr &tensor);

TypeId GetParameterDtype(const ParameterPtr &param_node);

STATUS UpdateFuncGraphInputsAndOutputsDtype(const FuncGraphPtr &func_graph);

template <typename T>
bool IndexingCompress(const std::set<T> &quant_data_set, const std::map<T, size_t> &unique_value_index_map,
                      size_t unique_value_bit, size_t unique_value_cnt, size_t pack_repetition_size_in_byte,
                      size_t bit_num, schema::TensorT *tensor) {
  auto quant_data_array = reinterpret_cast<T *>(tensor->data.data());
  std::vector<T> quant_data(quant_data_array, quant_data_array + tensor->data.size() / sizeof(T));

  std::vector<bool> bits(pack_repetition_size_in_byte * 8);
  size_t index = 0;
  // write unique_value_cnt: bit_num bit for unsigned
  for (size_t i = 0; i < bit_num; i++) {
    bits[index++] = (unique_value_cnt >> (bit_num - i - 1)) & (0x1);
  }
  // write the unique value set: each value has bit_num bit signed
  for (auto unique_value : quant_data_set) {
    for (size_t i = 0; i < bit_num; i++) {
      bits[index++] = ((unique_value + (1 << (bit_num - 1))) >> (bit_num - i - 1)) & (0x1);
    }
  }
  // write the index: each index has unique_value_bit unsigned
  for (auto quant_value : quant_data) {
    for (size_t i = 0; i < unique_value_bit; i++) {
      bits[index++] = (unique_value_index_map.at(quant_value) >> (unique_value_bit - i - 1)) & (0x1);
    }
  }
  if (index > pack_repetition_size_in_byte * 8) {
    MS_LOG(ERROR) << "unexpected index: " << index << " should not greater than " << pack_repetition_size_in_byte * 8;
    return false;
  }
  // update tensor data
  auto new_data_str = BoolVectorToString(bits);
  auto ret = memcpy_s(tensor->data.data(), tensor->data.size(), new_data_str.c_str(), new_data_str.size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error";
    return false;
  }
  tensor->data.resize(new_data_str.size());

  tensor->weightQunatCompressType = schema::WeightQunatCompressType_INDEXING;
  MS_LOG(DEBUG) << "set WeightQunatCompressType_INDEXING";
  return true;
}

template <typename T>
bool SparsityCompress(const std::set<T> &quant_data_set, const std::map<T, size_t> &unique_value_index_map,
                      size_t unique_value_bit, size_t unique_value_cnt, size_t pack_sparsity_size_in_byte,
                      size_t nz_cnt, size_t coor_best_bit, size_t bit_num, schema::TensorT *tensor) {
  auto quant_data_array = reinterpret_cast<T *>(tensor->data.data());
  std::vector<T> quant_data(quant_data_array, quant_data_array + tensor->data.size() / sizeof(T));
  auto &quant_params = tensor->quantParams;
  auto elem_cnt = quant_data.size();
  auto channel_cnt = quant_params.size();
  MS_CHECK_TRUE_MSG(channel_cnt != 0, false, "div zero.");
  auto elem_perchannel = elem_cnt / channel_cnt;

  std::vector<bool> bits(pack_sparsity_size_in_byte * 8);
  int index = 0;
  // coor_best_bit
  for (size_t i = 0; i < 8; i++) {
    bits[index++] = (coor_best_bit >> (8 - i - 1)) & 0x1;
  }
  // nz_cnt
  for (size_t i = 0; i < 32; i++) {
    bits[index++] = (nz_cnt >> (32 - i - 1)) & 0x1;
  }
  // unique_value cnt
  for (size_t i = 0; i < bit_num; i++) {
    bits[index++] = (unique_value_cnt >> (bit_num - i - 1)) & 0x1;
  }
  // unique_values
  for (auto unique_value : quant_data_set) {
    for (size_t i = 0; i < bit_num; i++) {
      bits[index++] = ((unique_value + (1 << (bit_num - 1))) >> (bit_num - i - 1)) & (0x1);
    }
  }
  // nz values indexing && get coor
  std::vector<size_t> coors(nz_cnt);
  size_t coors_index = 0;
  size_t prev_index = -1;
  for (size_t di = 0; di < elem_cnt; di++) {
    auto cur_channel = di / elem_perchannel;
    auto zp = quant_params[cur_channel]->zeroPoint;
    auto nz_value = quant_data[di];
    if (nz_value != zp || (di - prev_index) >= (size_t)(1 << coor_best_bit)) {
      MS_ASSERT(coors_index < nz_cnt);
      coors[coors_index++] = di - prev_index - 1;
      prev_index = di;
      for (size_t i = 0; i < unique_value_bit; i++) {
        bits[index++] = (unique_value_index_map.at(nz_value) >> (unique_value_bit - i - 1)) & (0x1);
      }
    }
  }
  // write coor
  for (auto coor : coors) {
    for (size_t i = 0; i < coor_best_bit; i++) {
      bits[index++] = (coor >> (coor_best_bit - i - 1)) & 0x1;
    }
  }
  if ((unsigned int)index > pack_sparsity_size_in_byte * 8) {
    MS_LOG(ERROR) << "unexpected index: " << index << " should not greater than " << pack_sparsity_size_in_byte * 8;
    return false;
  }
  auto new_data_str = BoolVectorToString(bits);
  auto ret = memcpy_s(tensor->data.data(), tensor->data.size(), new_data_str.c_str(), new_data_str.size());
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error";
    return false;
  }
  tensor->data.resize(new_data_str.size());

  tensor->weightQunatCompressType = schema::WeightQunatCompressType_SPARSE;
  MS_LOG(INFO) << "set WeightQunatCompressType_SPARSITY";
  return true;
}

template <typename T>
size_t CalCoorBestBit(const std::vector<T> &quant_data, size_t elem_cnt,
                      const std::vector<std::unique_ptr<schema::QuantParamT>> &quant_params, int unique_value_bit,
                      size_t *coor_best_bit) {
  size_t best_nn_cnt = 0;
  size_t min_len_in_bit = std::numeric_limits<size_t>::max();
  for (int bit = 2; bit <= 10; bit++) {
    // search
    size_t nn_cnt = 0;
    size_t prev_index = -1;
    auto channel_cnt = quant_params.size();
    auto elem_perchannel = elem_cnt / channel_cnt;
    for (size_t i = 0; i < elem_cnt; i++) {
      auto cur_channel = i / elem_perchannel;
      auto zp = quant_params[cur_channel]->zeroPoint;
      if (quant_data[i] != zp || (i - prev_index) >= (size_t)(1 << bit)) {
        nn_cnt++;
        prev_index = i;
      }
    }

    size_t len_in_bit = nn_cnt * bit + nn_cnt * unique_value_bit;
    if (len_in_bit < min_len_in_bit) {
      min_len_in_bit = len_in_bit;
      *coor_best_bit = bit;
      best_nn_cnt = nn_cnt;
    }
  }
  return best_nn_cnt;
}

template <typename T>
bool PackRepetition(size_t bit_num, schema::TensorT *tensor) {
  auto quant_data_array = reinterpret_cast<T *>(tensor->data.data());
  std::vector<T> quant_data(quant_data_array, quant_data_array + tensor->data.size() / sizeof(T));
  auto elem_cnt = quant_data.size();
  auto dims = tensor->dims;
  size_t elem_cnt_by_dims = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  if (elem_cnt != elem_cnt_by_dims) {
    MS_LOG(ERROR) << "elem_cnt: " << elem_cnt << " not equal elem_cnt_by_dims: " << elem_cnt_by_dims;
    return false;
  }

  auto &quant_params = tensor->quantParams;

  std::set<T> quant_data_set;
  for (auto quant_value : quant_data) {
    quant_data_set.insert(quant_value);
  }
  std::map<T, size_t> unique_value_index_map;
  auto index = 0;
  for (auto value : quant_data_set) {
    unique_value_index_map[value] = index++;
  }

  auto unique_value_cnt = quant_data_set.size();
  size_t unique_value_bit = ceil(log2(unique_value_cnt));
  auto pack_repetition_size_in_bit = bit_num + bit_num * unique_value_cnt + unique_value_bit * elem_cnt;
  size_t pack_repetition_size_in_byte = ceil(pack_repetition_size_in_bit / 8.0);
  size_t origin_size_in_byte = ceil(bit_num * elem_cnt / 8.0);

  size_t coor_best_bit = 0;
  auto nz_cnt = CalCoorBestBit<T>(quant_data, elem_cnt, quant_params, unique_value_bit, &coor_best_bit);
  // 1. coor_best_bit 2. nz_cnt 3. quant_data_set size 4. unique_values 5. unique_value indexing 6. nz values coord
  auto pack_sparsity_size_in_bit =
    1 * 8 + 4 * 8 + bit_num + bit_num * unique_value_cnt + unique_value_bit * nz_cnt + nz_cnt * coor_best_bit;
  size_t pack_sparsity_size_in_byte = ceil(pack_sparsity_size_in_bit / 8.0);
  MS_LOG(DEBUG) << "coor_best_bit: " << coor_best_bit << " ori: " << origin_size_in_byte
                << " indexing: " << pack_repetition_size_in_byte << " sparse: " << pack_sparsity_size_in_byte;
  auto min_byte_need = std::min({origin_size_in_byte, pack_repetition_size_in_byte, pack_sparsity_size_in_byte});
  if (min_byte_need == origin_size_in_byte) {
    return false;
  } else if (min_byte_need == pack_repetition_size_in_byte) {
    MS_LOG(DEBUG) << "from " << origin_size_in_byte << " to " << pack_repetition_size_in_byte;
    return IndexingCompress<T>(quant_data_set, unique_value_index_map, unique_value_bit, unique_value_cnt,
                               pack_repetition_size_in_byte, bit_num, tensor);
  } else if (min_byte_need == pack_sparsity_size_in_byte) {
    MS_LOG(DEBUG) << "from " << origin_size_in_byte << " to " << pack_sparsity_size_in_byte;
    return SparsityCompress<T>(quant_data_set, unique_value_index_map, unique_value_bit, unique_value_cnt,
                               pack_sparsity_size_in_byte, nz_cnt, coor_best_bit, bit_num, tensor);
  } else {
    MS_LOG(DEBUG) << "unexpected: " << min_byte_need << " not in {" << origin_size_in_byte << " "
                  << pack_repetition_size_in_byte << " " << pack_sparsity_size_in_byte << "}";
  }
  return false;
}

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_GRAPH_UTIL_H
