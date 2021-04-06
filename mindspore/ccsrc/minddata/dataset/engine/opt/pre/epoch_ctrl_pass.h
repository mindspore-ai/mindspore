/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef DATASET_ENGINE_OPT_PASS_PRE_EPOCH_INJECTION_PASS_H_
#define DATASET_ENGINE_OPT_PASS_PRE_EPOCH_INJECTION_PASS_H_

#include <memory>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

/// \class EpochInjectionPass epoch_ctrl_pass.h
/// \brief This is a pre pass that drives the injection of any nodes that could not be directly injected from the api
///     parsing.
class EpochCtrlPass : public IRTreePass {
  /// \class InjectionFinder
  /// \brief This is a nested node pass class who's job is to parse the tree and perform any identification logic for
  ///     operators that need to be injected.  It is run first by the main injection pass to find out what operators
  ///     it may need to inject.
  class InjectionFinder : public IRNodePass {
   public:
    /// \brief Constructor
    explicit InjectionFinder(std::shared_ptr<DatasetNode> node);

    /// \brief Destructor
    ~InjectionFinder() = default;

    /// \brief Performs finder work for BuildVocabNode that has special rules about epoch control injection.
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<RootNode> node, bool *const modified) override;

    /// \brief Performs finder work for BuildVocabNode that has special rules about epoch control injection.
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<BuildVocabNode> node, bool *const modified) override;

#ifndef ENABLE_ANDROID
    /// \brief Performs finder work for BuildSentenceVocabNode that has special rules about epoch control injection.
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<BuildSentenceVocabNode> node, bool *const modified) override;
#endif

    /// \brief Register the TransferNode for further action.
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status VisitAfter(std::shared_ptr<TransferNode> node, bool *const modified) override;

    /// \brief Getter
    std::shared_ptr<DatasetNode> injection_point() { return injection_point_; }

    /// \brief Getter
    int32_t num_epochs() { return num_epochs_; }

   private:
    std::shared_ptr<DatasetNode> injection_point_;
    int32_t num_epochs_;
  };

 public:
  /// \brief Constructor
  EpochCtrlPass();

  /// \brief Destructor
  ~EpochCtrlPass() = default;

  /// \brief Runs an injection pass to inject in operators needed at the pre pass stage
  /// \param[in, out] tree The tree to operate on.
  /// \param[in, out] Indicate of the tree was modified.
  /// \return Status The status code returned
  Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_PRE_EPOCH_INJECTION_PASS_H_
