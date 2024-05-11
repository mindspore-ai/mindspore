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

#include "transform/graph_ir/op_declare/nn_other_ops_declare.h"
#include <vector>
#include <string>
namespace mindspore::transform {
// InitPartitionMap
INPUT_MAP(InitPartitionMap) = {{1, INPUT_DESC(ps_num)}, {2, INPUT_DESC(ps_ids)}};
INPUT_ATTR_MAP(InitPartitionMap) = {{3, ATTR_DESC(partition_num, AnyTraits<int64_t>())},
                                    {4, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                    {5, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
                                    {6, ATTR_DESC(_ps_num, AnyTraits<int64_t>())}};
ATTR_MAP(InitPartitionMap) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(InitPartitionMap) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(InitPartitionMap, kNameInitPartitionMap, ADPT_DESC(InitPartitionMap))

// InitEmbeddingHashmap
INPUT_MAP(InitEmbeddingHashmap) = {{1, INPUT_DESC(table_id)}};
INPUT_ATTR_MAP(InitEmbeddingHashmap) = {{2, ATTR_DESC(value_total_len, AnyTraits<int64_t>())},
                                        {3, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},
                                        {4, ATTR_DESC(bucket_size, AnyTraits<int64_t>())},
                                        {5, ATTR_DESC(dtype, AnyTraits<GEType>())},
                                        {6, ATTR_DESC(initializer_mode, AnyTraits<std::string>())},
                                        {7, ATTR_DESC(constant_value, AnyTraits<float>())},
                                        {8, ATTR_DESC(min, AnyTraits<float>())},
                                        {9, ATTR_DESC(max, AnyTraits<float>())},
                                        {10, ATTR_DESC(mu, AnyTraits<float>())},
                                        {11, ATTR_DESC(sigma, AnyTraits<float>())},
                                        {12, ATTR_DESC(seed, AnyTraits<int64_t>())},
                                        {13, ATTR_DESC(seed2, AnyTraits<int64_t>())},
                                        {14, ATTR_DESC(filter_mode, AnyTraits<std::string>())},
                                        {15, ATTR_DESC(optimizer_mode, AnyTraits<std::string>())},
                                        {16, ATTR_DESC(optimizer_params, AnyTraits<std::vector<float>>())},
                                        {17, ATTR_DESC(_table_id, AnyTraits<int64_t>())}};
ATTR_MAP(InitEmbeddingHashmap) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(InitEmbeddingHashmap) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(InitEmbeddingHashmap, kNameInitEmbeddingHashmap, ADPT_DESC(InitEmbeddingHashmap))

// EmbeddingTableFind
INPUT_MAP(EmbeddingTableFind) = {{1, INPUT_DESC(table_id)}, {2, INPUT_DESC(keys)}};
INPUT_ATTR_MAP(EmbeddingTableFind) = {
  {3, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},       {4, ATTR_DESC(default_value, AnyTraits<float>())},
  {5, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},      {6, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
  {7, ATTR_DESC(_use_counter_filter, AnyTraits<int64_t>())},
};
ATTR_MAP(EmbeddingTableFind) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
  {"_execute_times", ATTR_DESC(_execute_times, AnyTraits<int64_t>())}};
OUTPUT_MAP(EmbeddingTableFind) = {{0, OUTPUT_DESC(values)}};
REG_ADPT_DESC(EmbeddingTableFind, kNameEmbeddingTableFind, ADPT_DESC(EmbeddingTableFind))

// EmbeddingTableFindAndInit
INPUT_MAP(EmbeddingTableFindAndInit) = {{1, INPUT_DESC(table_id)}, {2, INPUT_DESC(keys)}};
INPUT_ATTR_MAP(EmbeddingTableFindAndInit) = {{4, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},
                                             {5, ATTR_DESC(value_total_len, AnyTraits<int64_t>())},
                                             {6, ATTR_DESC(initializer_mode, AnyTraits<GEInitializerMode>())},
                                             {7, ATTR_DESC(constant_value, AnyTraits<float>())},
                                             {8, ATTR_DESC(min, AnyTraits<float>())},
                                             {9, ATTR_DESC(max, AnyTraits<float>())},
                                             {10, ATTR_DESC(mu, AnyTraits<float>())},
                                             {11, ATTR_DESC(sigma, AnyTraits<float>())},
                                             {12, ATTR_DESC(seed, AnyTraits<int64_t>())},
                                             {13, ATTR_DESC(seed2, AnyTraits<int64_t>())},
                                             {14, ATTR_DESC(filter_mode, AnyTraits<GEFilterMode>())},
                                             {15, ATTR_DESC(filter_freq, AnyTraits<int64_t>())},
                                             {16, ATTR_DESC(default_key_or_value, AnyTraits<bool>())},
                                             {17, ATTR_DESC(default_key, AnyTraits<int64_t>())},
                                             {18, ATTR_DESC(default_value, AnyTraits<float>())},
                                             {19, ATTR_DESC(optimizer_mode, AnyTraits<GEOptimizerMode>())},
                                             {20, ATTR_DESC(optimizer_params, AnyTraits<std::vector<float>>())},
                                             {21, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                             {22, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
                                             {23, ATTR_DESC(_use_counter_filter, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingTableFindAndInit) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
  {"_execute_times", ATTR_DESC(_execute_times, AnyTraits<int64_t>())}};
OUTPUT_MAP(EmbeddingTableFindAndInit) = {{0, OUTPUT_DESC(values)}};
REG_ADPT_DESC(EmbeddingTableFindAndInit, kNameEmbeddingTableFindAndInit, ADPT_DESC(EmbeddingTableFindAndInit))

// EmbeddingApplyFtrl
INPUT_MAP(EmbeddingApplyFtrl) = {{1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(lr)},      {3, INPUT_DESC(lr_power)},
                                 {4, INPUT_DESC(lambda1)},    {5, INPUT_DESC(lambda2)}, {6, INPUT_DESC(grad)},
                                 {7, INPUT_DESC(keys)}};
INPUT_ATTR_MAP(EmbeddingApplyFtrl) = {{8, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},
                                      {9, ATTR_DESC(mask_zero, AnyTraits<bool>())},
                                      {10, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                      {11, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyFtrl) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingApplyFtrl) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyFtrl, kNameEmbeddingApplyFtrl, ADPT_DESC(EmbeddingApplyFtrl))

// EmbeddingApplyAdam
INPUT_MAP(EmbeddingApplyAdam) = {
  {1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(beta1_power)}, {3, INPUT_DESC(beta2_power)}, {4, INPUT_DESC(lr)},
  {5, INPUT_DESC(beta1)},      {6, INPUT_DESC(beta2)},       {7, INPUT_DESC(epsilon)},     {8, INPUT_DESC(grad)},
  {9, INPUT_DESC(keys)},       {10, INPUT_DESC(global_step)}};
INPUT_ATTR_MAP(EmbeddingApplyAdam) = {{11, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},
                                      {12, ATTR_DESC(mask_zero, AnyTraits<bool>())},
                                      {13, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                      {14, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyAdam) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingApplyAdam) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyAdam, kNameEmbeddingApplyAdam, ADPT_DESC(EmbeddingApplyAdam))

// EmbeddingApplyAdamW
INPUT_MAP(EmbeddingApplyAdamW) = {
  {1, INPUT_DESC(var_handle)},   {2, INPUT_DESC(beta1_power)}, {3, INPUT_DESC(beta2_power)},   {4, INPUT_DESC(lr)},
  {5, INPUT_DESC(weight_decay)}, {6, INPUT_DESC(beta1)},       {7, INPUT_DESC(beta2)},         {8, INPUT_DESC(epsilon)},
  {9, INPUT_DESC(grad)},         {10, INPUT_DESC(keys)},       {11, INPUT_DESC(max_grad_norm)}};
INPUT_ATTR_MAP(EmbeddingApplyAdamW) = {
  {12, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},  {13, ATTR_DESC(amsgrad, AnyTraits<bool>())},
  {14, ATTR_DESC(maximize, AnyTraits<bool>())},          {15, ATTR_DESC(mask_zero, AnyTraits<bool>())},
  {16, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())}, {17, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyAdamW) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingApplyAdamW) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyAdamW, kNameEmbeddingApplyAdamW, ADPT_DESC(EmbeddingApplyAdamW))

// EmbeddingApplyAdaGrad
INPUT_MAP(EmbeddingApplyAdaGrad) = {
  {1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(lr)},          {3, INPUT_DESC(grad)},
  {4, INPUT_DESC(keys)},       {5, INPUT_DESC(global_step)},
};
INPUT_ATTR_MAP(EmbeddingApplyAdaGrad) = {{6, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},
                                         {7, ATTR_DESC(mask_zero, AnyTraits<bool>())},
                                         {8, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                         {9, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyAdaGrad) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingApplyAdaGrad) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyAdaGrad, kNameEmbeddingApplyAdaGrad, ADPT_DESC(EmbeddingApplyAdaGrad))

// EmbeddingTableImport
INPUT_MAP(EmbeddingTableImport) = {{1, INPUT_DESC(file_path)}, {2, INPUT_DESC(ps_id)}, {3, INPUT_DESC(table_id)}};
ATTR_MAP(EmbeddingTableImport) = {
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
  {"value_total_len", ATTR_DESC(value_total_len, AnyTraits<std::vector<int64_t>>())},
  {"only_var_flag", ATTR_DESC(only_var_flag, AnyTraits<bool>())},
  {"file_type", ATTR_DESC(file_type, AnyTraits<std::string>())},
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingTableImport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingTableImport, kNameEmbeddingTableImport, ADPT_DESC(EmbeddingTableImport))

// EmbeddingTableExport
INPUT_MAP(EmbeddingTableExport) = {
  {1, INPUT_DESC(file_path)}, {2, INPUT_DESC(ps_id)}, {3, INPUT_DESC(table_id)}, {4, INPUT_DESC(global_step)}};
ATTR_MAP(EmbeddingTableExport) = {
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
  {"value_total_len", ATTR_DESC(value_total_len, AnyTraits<std::vector<int64_t>>())},
  {"export_mode", ATTR_DESC(export_mode, AnyTraits<std::string>())},
  {"only_var_flag", ATTR_DESC(only_var_flag, AnyTraits<bool>())},
  {"file_type", ATTR_DESC(file_type, AnyTraits<std::string>())},
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"filter_export_flag", ATTR_DESC(filter_export_flag, AnyTraits<bool>())},
  {"steps_to_live_list", ATTR_DESC(steps_to_live_list, AnyTraits<std::vector<int64_t>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingTableExport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingTableExport, kNameEmbeddingTableExport, ADPT_DESC(EmbeddingTableExport))

// EmbeddingComputeVarExport
INPUT_MAP(EmbeddingComputeVarExport) = {
  {1, INPUT_DESC(file_path)},
  {2, INPUT_DESC(ps_id)},
  {3, INPUT_DESC(table_id)},
};
ATTR_MAP(EmbeddingComputeVarExport) = {
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingComputeVarExport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingComputeVarExport, kNameEmbeddingComputeVarExport, ADPT_DESC(EmbeddingComputeVarExport))

// EmbeddingComputeVarImport
INPUT_MAP(EmbeddingComputeVarImport) = {
  {1, INPUT_DESC(file_path)},
  {2, INPUT_DESC(ps_id)},
  {3, INPUT_DESC(table_id)},
};
ATTR_MAP(EmbeddingComputeVarImport) = {
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingComputeVarImport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingComputeVarImport, kNameEmbeddingComputeVarImport, ADPT_DESC(EmbeddingComputeVarImport))

// FakeRemoteLookupUniqued
INPUT_MAP(FakeRemoteLookupUniqued) = {{1, INPUT_DESC(table_id)},
                                      {2, INPUT_DESC(keys)},
                                      {3, INPUT_DESC(actual_keys_num)},
                                      {4, INPUT_DESC(unique_indices)},
                                      {5, INPUT_DESC(key_count)}};
INPUT_ATTR_MAP(FakeRemoteLookupUniqued) = {{7, ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},
                                           {8, ATTR_DESC(value_total_len, AnyTraits<int64_t>())},
                                           {9, ATTR_DESC(initializer_mode, AnyTraits<GEInitializerMode>())},
                                           {10, ATTR_DESC(constant_value, AnyTraits<float>())},
                                           {11, ATTR_DESC(min, AnyTraits<float>())},
                                           {12, ATTR_DESC(max, AnyTraits<float>())},
                                           {13, ATTR_DESC(mu, AnyTraits<float>())},
                                           {14, ATTR_DESC(sigma, AnyTraits<float>())},
                                           {15, ATTR_DESC(seed, AnyTraits<int64_t>())},
                                           {16, ATTR_DESC(seed2, AnyTraits<int64_t>())},
                                           {17, ATTR_DESC(filter_mode, AnyTraits<GEFilterMode>())},
                                           {18, ATTR_DESC(filter_freq, AnyTraits<int64_t>())},
                                           {19, ATTR_DESC(default_key_or_value, AnyTraits<bool>())},
                                           {20, ATTR_DESC(default_key, AnyTraits<int64_t>())},
                                           {21, ATTR_DESC(default_value, AnyTraits<float>())},
                                           {22, ATTR_DESC(optimizer_mode, AnyTraits<GEOptimizerMode>())},
                                           {23, ATTR_DESC(optimizer_params, AnyTraits<std::vector<float>>())},
                                           {24, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                           {25, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
                                           {26, ATTR_DESC(_use_counter_filter, AnyTraits<int64_t>())}};
ATTR_MAP(FakeRemoteLookupUniqued) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
  {"_execute_times", ATTR_DESC(_execute_times, AnyTraits<int64_t>())}};
OUTPUT_MAP(FakeRemoteLookupUniqued) = {{0, OUTPUT_DESC(values)}};
REG_ADPT_DESC(FakeRemoteLookupUniqued, kNameFakeRemoteLookupUniqued, ADPT_DESC(FakeRemoteLookupUniqued))
}  // namespace mindspore::transform
