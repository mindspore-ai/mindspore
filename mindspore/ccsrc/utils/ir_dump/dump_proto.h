/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_IR_DUMP_PROTO_H_
#define MINDSPORE_CCSRC_UTILS_IR_DUMP_PROTO_H_

#include <map>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <fstream>
#include <string>
#include <set>
#include <list>
#include <unordered_map>
#include <vector>

#include "utils/hash_map.h"
#include "ir/tensor.h"
#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"
#include "utils/check_convert_utils.h"
#include "include/utils/visible.h"
#include "include/utils/common.h"
#include "utils/system/env.h"
#include "ir/functor.h"
#include "ops/scalar_graph_holder.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
using FloatPtr = std::shared_ptr<Float>;
using BFloatPtr = std::shared_ptr<BFloat>;
using IntPtr = std::shared_ptr<Int>;
using UIntPtr = std::shared_ptr<UInt>;
using ComplexPtr = std::shared_ptr<Complex>;
using ModelProtoPtr = std::shared_ptr<mind_ir::ModelProto>;
constexpr const size_t TOTAL_SAVE = 1024 * 1024 * 1024;
constexpr const int64_t OFFSET = 64;
constexpr const size_t PARA_ROUND = 1024;

// anf type to mindir type map
static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_type_map = {
  {kNumberTypeBool, mind_ir::TensorProto_DataType_BOOL},
  {kNumberTypeInt4, mind_ir::TensorProto_DataType_QINT4X2},
  {kNumberTypeInt8, mind_ir::TensorProto_DataType_INT8},
  {kNumberTypeInt16, mind_ir::TensorProto_DataType_INT16},
  {kNumberTypeInt, mind_ir::TensorProto_DataType_INT32},
  {kNumberTypeInt32, mind_ir::TensorProto_DataType_INT32},
  {kNumberTypeInt64, mind_ir::TensorProto_DataType_INT64},
  {kNumberTypeUInt8, mind_ir::TensorProto_DataType_UINT8},
  {kNumberTypeUInt16, mind_ir::TensorProto_DataType_UINT16},
  {kNumberTypeUInt32, mind_ir::TensorProto_DataType_UINT32},
  {kNumberTypeUInt64, mind_ir::TensorProto_DataType_UINT64},
  {kNumberTypeFloat16, mind_ir::TensorProto_DataType_FLOAT16},
  {kNumberTypeFloat, mind_ir::TensorProto_DataType_FLOAT},
  {kNumberTypeFloat32, mind_ir::TensorProto_DataType_FLOAT},
  {kNumberTypeFloat64, mind_ir::TensorProto_DataType_DOUBLE},
  {kNumberTypeBFloat16, mind_ir::TensorProto_DataType_BFLOAT16},
  {kObjectTypeString, mind_ir::TensorProto_DataType_STRING},
  {kNumberTypeComplex64, mind_ir::TensorProto_DataType_COMPLEX64},
  {kNumberTypeComplex128, mind_ir::TensorProto_DataType_COMPLEX128}};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_int_map = {
  {4, mind_ir::TensorProto_DataType_QINT4X2}, {8, mind_ir::TensorProto_DataType_INT8},
  {16, mind_ir::TensorProto_DataType_INT16},  {32, mind_ir::TensorProto_DataType_INT32},
  {64, mind_ir::TensorProto_DataType_INT64},
};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_uint_map = {
  {8, mind_ir::TensorProto_DataType_UINT8},
  {16, mind_ir::TensorProto_DataType_UINT16},
  {32, mind_ir::TensorProto_DataType_UINT32},
  {64, mind_ir::TensorProto_DataType_UINT64},
};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_float_map = {
  {16, mind_ir::TensorProto_DataType_FLOAT16},
  {32, mind_ir::TensorProto_DataType_FLOAT},
  {64, mind_ir::TensorProto_DataType_FLOAT64},
};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_bfloat_map = {
  {16, mind_ir::TensorProto_DataType_BFLOAT16},
};

static mindspore::HashMap<int, mind_ir::TensorProto_DataType> g_data_bits_complex_map = {
  {64, mind_ir::TensorProto_DataType_COMPLEX64},
  {128, mind_ir::TensorProto_DataType_COMPLEX128},
};

static std::set<std::string> g_export_attr_blacklist = {kAttrDump};

// Can build different builder according to format
class IrExportBuilder;
using IrExportBuilderPtr = std::shared_ptr<IrExportBuilder>;

class IrExporter {
 public:
  explicit IrExporter(IrExportBuilderPtr builder) : builder_(std::move(builder)) {}
  virtual ~IrExporter() = default;
  std::string GetDumpString(const FuncGraphPtr &func_graph);
  ModelProtoPtr GetDumpProto(const FuncGraphPtr &func_graph);
  ModelProtoPtr GetDumpProto(const FuncGraphPtr &root_graph, const std::vector<FuncGraphPtr> &child_graphs,
                             const std::vector<AnfNodePtr> &isolated_nodes);

 private:
  IrExportBuilderPtr builder_;
};
using IrExporterPtr = std::shared_ptr<IrExporter>;

class IrExportBuilder {
 public:
  explicit IrExportBuilder(const bool &incremental = false)
      : model_(std::make_shared<mind_ir::ModelProto>()), incremental_(incremental) {}
  ~IrExportBuilder() = default;
  std::string GetProtoString() const;
  void BuildModelInfo();
  bool BuildModel(const FuncGraphPtr &func_graph);
  bool BuildModel(const FuncGraphPtr &root_graph, const std::vector<FuncGraphPtr> &child_graphs,
                  const std::vector<AnfNodePtr> &isolated_nodes);
  ModelProtoPtr Model() { return model_; }
  bool BuildFuncGraph(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildFuncGraphAttrs(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildParameters(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildNodes(const FuncGraphPtr &func_graph, mind_ir::GraphProto *const graph_proto);
  bool BuildIsolatedNodes(const std::vector<AnfNodePtr> &isolated_nodes);
  bool BuildIsolatedCNode(const AnfNodePtr &node, std::set<AnfNodePtr> *visited);
  bool BuildOutput(const CNodePtr &node, mind_ir::GraphProto *const graph_proto);
  bool BuildCNode(const CNodePtr &node, mind_ir::GraphProto *const graph_proto);
  bool BuildValueNode(const ValueNodePtr &node, const std::string &node_name, mind_ir::GraphProto *const graph_proto);
  std::string BuildInputNode(const AnfNodePtr &node, mind_ir::GraphProto *const graph_proto);
  bool BuildCNodeAttr(const CNodePtr &node, mind_ir::NodeProto *const node_proto);
  bool SetValueInfoProto(const AnfNodePtr &node, mind_ir::ValueInfoProto *const value_proto);
  bool SetParamToTensorProto(const ParameterPtr &param, mind_ir::TensorProto *const tensor_proto);
  bool ConvertMapParameterToMapTensorProto(const ParameterPtr &map_parameter,
                                           mind_ir::MapTensorProto *const map_tensor_proto);
  bool ConvertAbstractMapTensorToAttrProto(const AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto);
  bool SetTensorProto(const AbstractBasePtr &abstract, mind_ir::TensorProto *const tensor_proto);
  bool SetCSRTensorToProto(const AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto);
  bool SetCOOTensorToProto(const AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto);
  bool SetAttributeProto(const AnfNodePtr &node, mind_ir::NodeProto *const node_proto);
  bool ExportSequence(const abstract::AbstractSequencePtr &abs, mind_ir::AttributeProto *const attr_proto);
  bool SetAbstractToNodeProto(const CNodePtr &node, mind_ir::NodeProto *const node_proto);
  bool SetAbstractToNodeProto(const abstract::AbstractBasePtr &abstract, mind_ir::AttributeProto *const attr_proto);
  bool SetValueToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetNamedValueToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetTypeToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetScalarToAttributeProto_ir(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetScalarToAttributeProtoForInt_ir(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetScalarToAttributeProto_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetScalarToAttributeProtoForInt_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto) const;
  bool SetTypeToAttributeProto_irs(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetTensorToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetSequenceToAttributeProto(const ValueSequencePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetDictToAttributeProto(const ValueDictionaryPtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetSeqElemToAttributeProto(const ValuePtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetQuantizationParamToAttrProto(const std::shared_ptr<QuantizationParam> &quantization_param,
                                       mind_ir::TensorProto_QuantParamProto *const quant_param_proto);
  bool SetFunctorToAttrProto(const FunctorPtr &value, mind_ir::AttributeProto *const attr_proto);
  bool SetScalarGraphHolderToAttrProto(const ops::ScalarGraphHolderPtr &value,
                                       mind_ir::AttributeProto *const attr_proto);
  bool SetTensorTypeToAttributeProto(const ValuePtr &value, mind_ir::TensorProto *tensor_proto);
  bool SetFuncGraphToAttrProto(const FuncGraphPtr &value, mind_ir::AttributeProto *const attr_proto);

  mind_ir::TensorProto_DataType GetMindirDataType(TypeId type_id) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsIntType(int bits) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsFloatType(int bits) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsBFloatType(int bits) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsUIntType(int bits) const;
  mind_ir::TensorProto_DataType GetMindirDataBitsComplexType(int bits) const;
  std::string GetNodeName(const AnfNodePtr &node) const;
  std::string GetUniqueNodeName(const AnfNodePtr &node);
  std::string GetOpTypeName(const AnfNodePtr &node);
  size_t GetUniqueID() { return ++unique_id_; }

 private:
  bool SetAbstractFuncToAttributeProto(const abstract::AbstractBasePtr &abstract,
                                       mind_ir::AttributeProto *const attr_proto);
  bool ExportWeight(const ParameterPtr &param, const std::string &param_name, mind_ir::GraphProto *const graph_proto);
  std::string GetPrimitiveUniqueName(const PrimitivePtr &primitive_ptr);
  bool BuildPrimitives();
  bool BuildPrimitivesByMap(std::map<PrimitivePtr, std::string> *primitives);

  ModelProtoPtr model_;
  mind_ir::NodeProto *last_node_{nullptr};
  std::list<FuncGraphPtr> todo_;
  std::map<AnfNodePtr, std::string> node_name_map_;
  std::map<PrimitivePtr, std::string> primitive_name_map_;
  std::set<std::string> nodeName_;
  size_t unique_id_{0};
  bool top_graph{true};
  std::map<FuncGraphPtr, mind_ir::GraphProto *> graph_protos_;
  bool incremental_{false};
  bool is_kernel_graph_{false};
};

COMMON_EXPORT std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph);

COMMON_EXPORT std::string GetBinaryProtoString(const FuncGraphPtr &func_graph, const bool &incremental = false);

COMMON_EXPORT ModelProtoPtr GenBinaryProto(const FuncGraphPtr &func_graph);

COMMON_EXPORT bool DumpBinaryProto(const FuncGraphPtr &func_graph, const std::string &file_path);
COMMON_EXPORT bool DumpBinaryProto(const FuncGraphPtr &root_graph, const std::vector<FuncGraphPtr> &child_graphs,
                                   const std::vector<AnfNodePtr> &isolated_nodes, const std::string &file_path);
COMMON_EXPORT void DumpIRProto(const FuncGraphPtr &func_graph, const std::string &suffix);

COMMON_EXPORT std::string GetFuncGraphProtoJsonString(const FuncGraphPtr &func_graph);

class COMMON_EXPORT MindIRExporter {
 public:
  MindIRExporter() {}
  explicit MindIRExporter(bool is_export_model) { is_export_model_ = is_export_model; }
  virtual ~MindIRExporter() {
    if (data_fs_ != nullptr) {
      data_fs_->close();
      delete data_fs_;
      data_fs_ = nullptr;
    }
  }

  bool ExportProto(const FuncGraphPtr &func_graph, const std::string &file_path,
                   const FuncGraphPtr &param_layout_fg = nullptr);
  bool IsSystemLittleEndidan() const;
  bool PreProcSaveTogether(const FuncGraphPtr &func_graph);
  bool SaveProtoToFile(mind_ir::ModelProto *model_proto, const std::string &output_file);

 private:
  bool ParserPath(const std::string &output_path);
  bool IfSaveTogether(bool *save_together);
  bool SaveMindIRTogether();
  bool SplitSave();
  bool UpdateParamCount(const FuncGraphPtr &func_graph);

 private:
  bool ParamDict(const FuncGraphPtr &func_graph);
  bool CreateParameterDir();
  std::shared_ptr<Parameter> GetFgParaAccordingToProtoName(const std::string &proto_name);
  bool ChangeParaDataFile(const std::string &file);
  std::string CreateExternalPath(const std::string &external_file);

 private:
  std::string model_name_;
  std::string save_path_;
  std::string save_model_path_;
  std::string dir_name_;
  std::string dir_path_;
  bool save_together_ = true;
  mind_ir::ModelProto model_proto_;
  std::unordered_map<std::string, ParameterPtr> param_dict_{};
  std::unordered_map<tensor::TensorPtr, mind_ir::TensorProto *> para_proto_dict_{};
  std::fstream *data_fs_ = nullptr;
  std::shared_ptr<system::FileSystem> fs_{};
  bool is_export_model_ = true;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_IR_DUMP_PROTO_H_
