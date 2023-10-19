//
// Created by jojo on 2023/10/18.
//

#include "py_boost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void PyBoostUtils::CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs) {
  auto create_tensor = [&outputs](const TypePtr &type, const ShapeVector &shape_vector) {
    auto output_tensor = std::make_shared<tensor::Tensor>(type->type_id(), shape_vector);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    (void)outputs->emplace_back(output_tensor);
    MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();
  };

  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractSequence>()) {
    auto seq = abstract->cast<abstract::AbstractSequencePtr>();
    auto elements = seq->elements();
    for (const auto &element : elements) {
      CreateOutputTensor(element, outputs);
    }
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
    auto shape = abstract_tensor->BuildShape();
    auto type = abstract_tensor->element()->BuildType();
    MS_LOG(DEBUG) << "get abstract tensor shape " << shape->ToString() << " type " << type->ToString();
    if (!shape->isa<abstract::Shape>()) {
      MS_LOG(EXCEPTION) << "AbstractTensor shape is valid " << shape->ToString();
    }
    auto shape_vector = shape->cast<abstract::ShapePtr>()->shape();
    create_tensor(type, shape_vector);
  } else if (abstract->isa<abstract::AbstractScalar>()) {
    auto scalar = abstract->cast<abstract::AbstractScalarPtr>();
    const auto &type = scalar->BuildType();
    MS_LOG(DEBUG) << "Create scalar tensor type " << type->ToString();
    create_tensor(type, {});
  } else {
    MS_LOG(EXCEPTION) << "Not support abstract " << abstract->ToString();
  }
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
