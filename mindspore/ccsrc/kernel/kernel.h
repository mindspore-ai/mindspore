/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <optional>
#include <variant>
#include "nlohmann/json.hpp"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/base_operator.h"
#include "ir/tensor.h"
#include "abstract/dshape.h"
#include "utils/log_adapter.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/api/format.h"
#include "include/backend/visible.h"

#ifdef _MSC_VER
#undef OPAQUE
#endif

#ifdef OPAQUE
#undef OPAQUE
#endif

namespace mindspore {
enum KernelType : int {
  UNKNOWN_KERNEL_TYPE = 0,
  AKG_KERNEL,
  AICPU_KERNEL,
  RT_KERNEL,
  HCCL_KERNEL,
  TBE_KERNEL,
  HOST_KERNEL,
  CPU_KERNEL,
  GPU_KERNEL,
  BISHENG_KERNEL,
  ACL_KERNEL,
};
namespace kernel {
constexpr auto kPatternAntiQuant = "anti_quant";
constexpr auto kPatternDequant = "dequant";
constexpr auto kPatternDequantS16 = "dequant_s16";
constexpr auto kPatternQuant = "quant";
constexpr auto kPatternRequant = "requant";
constexpr auto kPatternRequant_s16 = "requant_s16";
constexpr auto kPatternBatchMatmul = "BatchMatmul";
constexpr auto kPatternBnGradReduce = "bn_grad_reduce";
constexpr auto kPatternBnReduce = "bn_reduce";
constexpr auto kPatternBnUpdate = "bn_update";
constexpr auto kPatternBnUpdate_grad = "bn_update_grad";
constexpr auto kPatternBroadcast = "Broadcast";
constexpr auto kPatternCommReduce = "CommReduce";
constexpr auto kPatternConfusiontranspose = "confusiontranspose";
constexpr auto kPatternConvolution = "Convolution";
constexpr auto kPatternConv2dBackpropFilter = "Conv2d_backprop_filter";
constexpr auto kPatternConv2dBackpropInput = "Conv2d_backprop_input";
constexpr auto kPatternConv3d = "Conv3d";
constexpr auto kPatternConv3dBackpropFilter = "Conv3d_backprop_filter";
constexpr auto kPatternConv3dBackpropInput = "Conv3d_backprop_input";
constexpr auto kPatternCosineEmbeddingLoss = "cosine_embedding_loss";
constexpr auto kPatternCubeLayerNorm = "cube_layer_norm";
constexpr auto kPatternDepthwiseConvolution = "DepthwiseConvolution";
constexpr auto kPatternDilation = "dilation";
constexpr auto kPatternDropOutDoMaskV3D = "DropOutDoMaskV3D";
constexpr auto kPatternElemWise = "ElemWise";
constexpr auto kPatternGEMM = "GEMM";
constexpr auto kPatternInplace = "Inplace";
constexpr auto kPatternL2Loss = "l2_loss";
constexpr auto kPatternL2Normalize = "l2_normalize";
constexpr auto kPatternL2lossMulAddn = "l2loss_mul_addn";
constexpr auto kPatternLayerNormGrad = "layer_norm_grad";
constexpr auto kPatternMatmul = "Matmul";
constexpr auto kPatternMatmulV2 = "Matmul_v2";
constexpr auto kPatternMaxPool = "MaxPool";
constexpr auto kPatternNorm = "Norm";
constexpr auto kPatternOpaque = "Opaque";
constexpr auto kPatternPool2d = "Pool2d";
constexpr auto kPatternPool3d = "Pool3d";
constexpr auto kPatternPureBroadcast = "PureBroadcast";
constexpr auto kPatternread_select = "read_select";
constexpr auto kPatternSegment = "Segment";
constexpr auto kPatternSoftmaxPattern = "softmax_pattern";
constexpr auto kPatternSoftmaxCrossEntropyWithLogits = "softmax_cross_entropy_with_logits";
constexpr auto kPatternStridedRead = "strided_read";
constexpr auto kPatternStridedWrite = "strided_write";
constexpr auto kPatternTransdata = "Transdata";
constexpr auto kPatternTranspose = "Transpose";
constexpr auto kPatternUnknown = "";
constexpr auto kPatternWriteSelect = "write_select";

// Backend processor
enum Processor {
  UNKNOWN = -1,
  AICORE = 0,
  AICPU,
  CUDA,
  CPU,
  BISHENG,
};

struct FlexArray {
  size_t len;
  char contents[];
};

struct GlobalWorkspace {
  size_t size;
  size_t type;
  bool is_overflow = false;
};

struct KernelJsonInfo {
  std::string bin_file_name;
  std::string bin_file_suffix;
  uint32_t block_dim;
  std::string kernel_name;
  std::string magic;
  std::vector<size_t> parameters;
  std::string sha256;
  std::vector<size_t> workspaces_type;
  std::vector<size_t> workspaces;
  GlobalWorkspace global_workspace;
  bool has_kernel_list = false;
  uint32_t op_para_size;
  int32_t KBHit;
  uint32_t mode_in_args_first_field;
  uint32_t batch_bind_only;
  uint32_t task_ration;
  std::string core_type;
  std::vector<std::vector<size_t>> args_remap;
  KernelJsonInfo() : block_dim(0), op_para_size(0) {}
};

class BACKEND_EXPORT KernelPack {
 public:
  KernelPack() : json_(nullptr), kernel_(nullptr) {}
  KernelPack(const KernelPack &) = default;
  KernelPack &operator=(const KernelPack &) = default;
  KernelJsonInfo kernel_json_info() const;
  bool LoadKernelMeta(const std::string &json_f);
  bool ReadFromJsonFile(const std::string &json_f, const std::string &processor);
  const FlexArray *GetJson() const { return json_; }
  const FlexArray *GetKernel() const { return kernel_; }
  ~KernelPack() {
    if (json_ != nullptr) {
      delete[] json_;
      json_ = nullptr;
    }
    if (kernel_ != nullptr) {
      delete[] kernel_;
      kernel_ = nullptr;
    }
  }

 private:
  bool ReadFromJsonFileHelper(std::ifstream &kernel_bin);
  void ParseKernelJson(const nlohmann::json &js);
  static void ParseKernelName(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBinFileName(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBinFileSuffix(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseMagic(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBlockDim(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseTaskRatio(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseCoreType(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseParameters(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseWorkSpace(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseOpParaSize(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseSHA256(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseKBHit(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseBatchBindOnly(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseKernelList(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseModeInArgsFirstField(const std::string &key, const nlohmann::json &js,
                                        KernelJsonInfo *kernel_json_info);
  static void ParseArgsRemap(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);
  static void ParseGlogbleWorkSpace(const std::string &key, const nlohmann::json &js, KernelJsonInfo *kernel_json_info);

  KernelJsonInfo kernel_json_info_;
  FlexArray *json_;
  FlexArray *kernel_;
};
using KernelPackPtr = std::shared_ptr<KernelPack>;

/**
 * @brief base class for autotensor kernel and cce kernel.
 */
struct Address {
  Address() : addr(nullptr), size(0) {}
  Address(void *address_addr, size_t address_size) : addr(address_addr), size(address_size) {}
  void *addr;
  size_t size;
};
using AddressPtr = std::shared_ptr<Address>;
using AddressPtrList = std::vector<AddressPtr>;
using StreamType = void *;
// The memory info of kernel launch.
struct KernelLaunchInfo {
  AddressPtrList inputs_;
  AddressPtrList outputs_;
  AddressPtrList workspaces_;
};
struct TensorInfo {
  mindspore::Format format;
  abstract::AbstractTensorPtr base_;             // Store data type and shape.
  std::vector<int64_t> device_shape_adaptively;  // deprecated field for dynamic shape
};
struct ScalarInfo {
  abstract::AbstractScalarPtr base_;
};
struct ListInfo {
  abstract::AbstractListPtr base_;
};
struct TupleInfo {
  abstract::AbstractTuplePtr base_;
};
using TensorInfoPtr = std::shared_ptr<TensorInfo>;
using BaseOperatorPtr = std::shared_ptr<ops::BaseOperator>;

class KernelAttr;
// we extend KernelTensor to represent tensor/map_tensor/list/tuple/scalar data type.
class BACKEND_EXPORT KernelTensor {
 public:
  KernelTensor() = default;
  ~KernelTensor() = default;

  bool IsDynamicShape() const;
  size_t GetSizeInBytes() const;
  AddressPtr GetData() const { return data_; }
  AddressPtr GetHostData() const { return host_data_; }
  TypeId GetDtype() const;
  mindspore::Format GetFormat() const {
    if (meta_type_ == kObjectTypeTensorType) {
      const TensorInfo &info = std::get<TensorInfo>(meta_);
      return info.format;
    }
    return Format::DEFAULT_FORMAT;
  }
  TypeId GetMetaType() const { return meta_type_; }
  std::variant<TensorInfo, ScalarInfo, TupleInfo, ListInfo> GetMeta() const { return meta_; }
  // If real type is not a list or tuple tensor, it will return kTypeUnknown.
  std::vector<TypeId> GetListOrTupleDtype() const;
  // If real type is not a single shape vector, it will return empty.
  ShapeVector GetShapeVector() const;
  // If real type is not a list or tuple shape vector, it will return empty.
  std::vector<ShapeVector> GetListOrTupleShapeVector() const;
  void SetData(const AddressPtr &data) { data_ = data; }
  void SetHostData(const AddressPtr &data) { host_data_ = data; }
  void SetDtype(const TypePtr &dtype);
  void SetFormat(mindspore::Format format) {
    TensorInfo &info = std::get<TensorInfo>(meta_);
    info.format = format;
  }
  void SetMetaType(const TypeId meta_type) { meta_type_ = meta_type; }
  void SetShapeVector(const ShapeVector &shape);

  // max shape is only used in compute-depended ops
  ShapeVector GetMaxShape() const;

  abstract::BaseShapePtr GetBaseShape() const;
  // If the shape need to be List or Tuple, `SetBaseShape` should be called.
  void SetBaseShape(const abstract::BaseShapePtr &base_shape);
  void SetTensorInfo(const TensorInfo &info) {
    meta_type_ = kObjectTypeTensorType;
    meta_ = info;
  }
  void SetScalarInfo(const ScalarInfo &info) {
    meta_type_ = kObjectTypeNumber;
    meta_ = info;
  }
  void SetTupleInfo(const TupleInfo &info) {
    meta_type_ = kObjectTypeTuple;
    meta_ = info;
  }
  void SetListInfo(const ListInfo &info) {
    meta_type_ = kObjectTypeList;
    meta_ = info;
  }
  // deprecated field for dynamic shape
  const ShapeVector &GetDeviceShapeAdaptively() const;
  void SetDeviceShapeAdaptively(const ShapeVector &device_shape_adaptively);

 private:
  TypeId meta_type_{kObjectTypeTensorType};
  // meta is a type-safe union of TensorInfo, ScalarInfo, TupleInfo, ListInfo.
  std::variant<TensorInfo, ScalarInfo, TupleInfo, ListInfo> meta_{TensorInfo()};
  AddressPtr data_{nullptr};       // Device data address.
  AddressPtr host_data_{nullptr};  // Host data address.
  string GetAbstractName() const;
};
using KernelTensorPtr = std::shared_ptr<KernelTensor>;

enum class KernelModType {
  Invalid = 0,
  KernelMod,
  GpuKernelMod,
  NativeGpuKernelMod,
  DeprecatedNativeGpuKernelMod,
  CpuKernelMod,
  NativeCpuKernelMod,
  DeprecatedNativeCpuKernelMod,
  HostKernelMod,
};

enum KernelErrorCode : int { KRET_OK = 0, KRET_RESIZE_FAILED = 1, KRET_UNKNOWN_SHAPE = 2, KRET_UNKNOWN_OUT_SHAPE = 3 };

class BACKEND_EXPORT KernelMod {
 public:
  KernelMod() {}
  virtual ~KernelMod() = default;
  virtual void SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }
  virtual void SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }
  virtual void SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }
  virtual const std::vector<size_t> &GetInputSizeList() const { return input_size_list_; }
  virtual const std::vector<size_t> &GetOutputSizeList() const { return output_size_list_; }
  virtual const std::vector<size_t> &GetWorkspaceSizeList() const { return workspace_size_list_; }
  virtual const std::vector<std::vector<int64_t>> &GetInputShapes() const { return input_shapes_; }
  virtual const std::vector<std::vector<int64_t>> &GetOutputShapes() const { return output_shapes_; }
  virtual bool Launch(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
                      const std::vector<AddressPtr> &workspace, void *stream_ptr);
  // deprecated
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr);
  virtual std::vector<size_t> GenParameters() { return {}; }
  virtual void ReleaseResource() {}
  // Initialization for the kernel mod.
  virtual bool Init(const BaseOperatorPtr & /* base_operator */, const std::vector<KernelTensorPtr> &inputs,
                    const std::vector<KernelTensorPtr> &outputs) {
    this->inputs_ = inputs;
    this->outputs_ = outputs;
    return true;
  }
  virtual std::vector<KernelAttr> GetOpSupport() = 0;
  // Resize() is for validating input/output shape and calculating the workspace size, framework will invoke this
  // routine after infer shape.
  virtual int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>());

  // Some kernels, e.g., Unique, can only get its output shape after its computing finished.
  virtual bool IsNeedRetrieveOutputShape() { return is_need_retrieve_output_shape_; }
  virtual std::vector<KernelTensorPtr> RetrieveOutputShape() {
    SyncData();
    return GetOutputs();
  }

  // Get input kernel tensor.
  virtual std::vector<KernelTensorPtr> GetInputKernelTensor() { return {}; }

  void set_unique_name(const std::string &unique_name) { unique_name_ = unique_name; }
  void set_fullname(const std::string &fullname) { fullname_ = fullname; }
  void set_is_monad(bool is_monad) { is_monad_ = is_monad; }
  void set_inputs_addr(const std::vector<AddressPtr> &addr) { inputs_addr_ = addr; }
  void set_workspaces_addr(const std::vector<AddressPtr> &addr) { workspaces_addr_ = addr; }
  void set_outputs_addr(const std::vector<AddressPtr> &addr) { outputs_addr_ = addr; }
  // User data is the extra data required when the kernel is launched, It will be set before launch by runtime.
  virtual void set_input_user_data(UserData *user_data, size_t input_index) {}
  virtual void set_output_user_data(UserData *user_data, size_t output_index) {}
  const std::vector<AddressPtr> &GetInputsAddr() const { return inputs_addr_; }
  const std::vector<AddressPtr> &GetWorkSpacesAddr() const { return workspaces_addr_; }
  const std::vector<AddressPtr> &GetOutputsAddr() const { return outputs_addr_; }
  void SetDevicedId(uint32_t device_id) { device_id_ = device_id; }
  virtual enum KernelModType GetKernelModType() const { return KernelModType::KernelMod; }
  bool Launch(const KernelLaunchInfo &kernel_launch_address, void *stream_ptr) {
    return Launch(kernel_launch_address.inputs_, kernel_launch_address.workspaces_, kernel_launch_address.outputs_,
                  stream_ptr);
  }
  int32_t task_id() const { return task_id_; }

 protected:
  virtual void SyncData() {}
  virtual std::vector<KernelTensorPtr> GetOutputs() { return {}; }
  bool IsValidShape(const ShapeVector &shape) const {
    if (std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; })) {
      return false;
    }
    return true;
  }

  std::string kernel_name_;
  std::string unique_name_;
  std::string fullname_;
  bool is_monad_{false};
  std::vector<size_t> input_size_list_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<size_t> output_size_list_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<size_t> workspace_size_list_;
  std::vector<KernelTensorPtr> inputs_;
  std::vector<KernelTensorPtr> outputs_;
  bool is_need_retrieve_output_shape_ = false;
  uint32_t device_id_ = 0;
  int32_t task_id_ = -1;

 private:
  std::vector<AddressPtr> inputs_addr_;
  std::vector<AddressPtr> workspaces_addr_;
  std::vector<KernelTensorPtr> workspace_;
  std::vector<AddressPtr> outputs_addr_;
};
using KernelModPtr = std::shared_ptr<KernelMod>;

template <typename T>
inline T *GetDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
  if (index >= addr_list.size()) {
    MS_LOG(ERROR) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    return nullptr;
  }

  if (addr_list[index] == nullptr) {
    MS_LOG(ERROR) << "The device address is nullptr, address index: " << index << ", and the length of 'addr_list' is "
                  << addr_list.size();
    return nullptr;
  }

  if (addr_list[index]->addr == nullptr) {
    MS_LOG(WARNING) << "The memory of device address is nullptr, address index: " << index
                    << ", and the length of 'addr_list' is " << addr_list.size();
    return nullptr;
  }

  // When the input is an empty tuple, the input size will be 0.
  if (addr_list[index]->size == 0) {
    MS_LOG(INFO) << "The size of device address is zero, address index: " << index
                 << ", and the length of 'addr_list' is " << addr_list.size();
    return nullptr;
  }
  return reinterpret_cast<T *>(addr_list[index]->addr);
}

BACKEND_EXPORT std::optional<std::vector<int64_t>> TryGetIntValueFromInputs(const std::vector<KernelTensorPtr> &inputs,
                                                                            const size_t input_index,
                                                                            const std::string &kernel_name,
                                                                            bool data_from_host);

inline bool TryGetIntValue(const std::vector<KernelTensorPtr> &inputs, const size_t input_index,
                           const std::string &kernel_name, int64_t *attr_value, bool data_from_host = true) {
  auto res = TryGetIntValueFromInputs(inputs, input_index, kernel_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  if (res.value().empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', value of the dynamic attr is empty!";
  }
  *attr_value = res.value()[0];
  return true;
}

inline bool TryGetIntValue(const std::vector<KernelTensorPtr> &inputs, const size_t input_index,
                           const std::string &kernel_name, std::vector<int64_t> *attr_value,
                           bool data_from_host = true) {
  auto res = TryGetIntValueFromInputs(inputs, input_index, kernel_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  *attr_value = res.value();
  return true;
}

BACKEND_EXPORT bool TryGetIntValue(const CNodePtr &kernel_node, const size_t input_index,
                                   std::vector<int64_t> *attr_value, bool data_from_host = true);
template <typename T>
inline bool CheckNullInput(const std::vector<T> &input_shape) {
  // If input_shape.size() == 0, it means a scalar input; If input_shape.size() != 0 and input_shape contains 0,
  // it means a null input. Just return a null output.
  if (input_shape.size() != 0) {
    if (std::any_of(input_shape.begin(), input_shape.end(), [](T i) { return i == 0; })) {
      return true;
    }
  }
  return false;
}
#define CHECK_NULL_INPUT(input_shape) mindspore::kernel::CheckNullInput(input_shape)

template <typename T>
inline std::string ConvertVectorToString(const std::vector<T> &value) {
  std::stringstream ss;
  ss << "(";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it == value.begin()) {
      ss << *it;
    } else {
      ss << ", " << *it;
    }
  }
  ss << ")";
  return ss.str();
}

#define CONVERT_VECTOR_TO_STRING(value) mindspore::kernel::ConvertVectorToString(value)

template <typename T>
inline bool CheckShapeNull(const std::vector<T> &shape, std::string kernel_name, std::string param_name) {
  if (CHECK_NULL_INPUT(shape)) {
    MS_LOG(WARNING) << "For '" << kernel_name << "', the shape of " << param_name << " cannot contain zero, but got "
                    << CONVERT_VECTOR_TO_STRING(shape);
    return true;
  }
  return false;
}

#define CHECK_SHAPE_NULL(shape, kernel_name, param_name) \
  mindspore::kernel::CheckShapeNull(shape, kernel_name, param_name)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
