#include "minddata/dataset/engine/ir/datasetops/source/libri_speech_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/libri_speech_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {
    
LibriSpeechNode::LibriSpeechNode(std::string dataset_dir, std::string usage, std::shared_ptr<SamplerObj> sampler,
                     std::shared_ptr<DatasetCache> cache)  
    : MappableSourceNode(std::move(cache)), dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler) {}
    
void LibriSpeechNode::Print(std::ostream &out) const { out << Name(); }
    
std::shared_ptr<DatasetNode> LibriSpeechNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<LibriSpeechNode>(dataset_dir_, usage_, sampler, cache_);
  return node;
}
    
Status LibriSpeechNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("LibriSpeechNode", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("LibriSpeechNode", sampler_));
  RETURN_IF_NOT_OK(ValidateStringValue("LibriSpeechNode", usage_, {"dev-clean", "dev-other", "test-clean","test-other","train-clean-100","train-clean-360","train-other-500"}));
  return Status::OK();
}
    
Status LibriSpeechNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT64), TensorImpl::kCv, 1)));  
  TensorShape scalar_rate = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("sample_rate", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar_rate)));
  TensorShape scalar_utterance = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("utterance", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar_utterance)));
  TensorShape scalar_speaker_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("speaker_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar_speaker_id)));
  TensorShape scalar_chapter_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("chapter_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar_chapter_id)));
  TensorShape scalar_utterance_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("utterance_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar_utterance_id)));
  



  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto op = std::make_shared<LibriSpeechOp>(usage_, num_workers_, dataset_dir_, connector_que_size_, std::move(schema),std::move(sampler_rt));
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}
    
// Get the shard id of node
Status LibriSpeechNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

    
// Get Dataset size
Status LibriSpeechNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(LibriSpeechOp::CountTotalRows(dataset_dir_, usage_, &num_rows));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

    
Status LibriSpeechNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}
   
} // namespace dataset
} // namespace mindspor