import numpy as np
import mindspore
import mindspore.communication.management as D
from mindspore import lazy_inline, context, nn, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.parallel import sync_pipeline_shared_parameters

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', jit_config={"jit_level": "O0"})
pipeline_stages = 4


class FC(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.w = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul = P.MatMul()

    def construct(self, x):
        return self.matmul(x, self.w)


class WordEmbedding(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.w = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul = P.MatMul()

    def construct(self, x):
        return self.matmul(x, self.w), self.w


class LMHead(nn.Cell):
    def __init__(self, shape):
        super().__init__()
        self.w = Parameter(Tensor(np.ones(shape), mindspore.float32), name="weight")
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

    def construct(self, x, w):
        x = self.matmul1(x, self.w)
        return self.matmul2(x, w), x + x


class Net(nn.Cell):
    @lazy_inline
    def __init__(self):
        super().__init__()
        shape = (8, 8)
        self.word_embedding = WordEmbedding(shape)
        self.decoder1 = FC(shape)
        self.decoder2 = FC(shape)
        self.lm_head = LMHead(shape)

        self.word_embedding.matmul.shard(((1, 1), (1, 1)))
        self.decoder1.matmul.shard(((1, 1), (1, 1)))
        self.decoder2.matmul.shard(((1, 1), (1, 1)))
        self.lm_head.matmul1.shard(((1, 1), (1, 1)))
        self.lm_head.matmul2.shard(((1, 1), (1, 1)))

        self.word_embedding.pipeline_stage = 0
        self.decoder1.pipeline_stage = 1
        self.decoder2.pipeline_stage = 2
        self.lm_head.pipeline_stage = 3

    def construct(self, x):
        x, w = self.word_embedding(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x, y = self.lm_head(x, w)
        return x, y


class NoLazyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        shape = (8, 8)
        self.word_embedding = WordEmbedding(shape)
        self.decoder1 = FC(shape)
        self.decoder2 = FC(shape)
        self.lm_head = LMHead(shape)

        self.word_embedding.matmul.shard(((1, 1), (1, 1)))
        self.decoder1.matmul.shard(((1, 1), (1, 1)))
        self.decoder2.matmul.shard(((1, 1), (1, 1)))
        self.lm_head.matmul1.shard(((1, 1), (1, 1)))
        self.lm_head.matmul2.shard(((1, 1), (1, 1)))

        self.word_embedding.pipeline_stage = 0
        self.decoder1.pipeline_stage = 1
        self.decoder2.pipeline_stage = 2
        self.lm_head.pipeline_stage = 3

    def construct(self, x):
        x, w = self.word_embedding(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x, y = self.lm_head(x, w)
        return x, y


class PipelineCellInference(nn.Cell):
    def __init__(self, network, micro_batch_num):
        super().__init__()
        self.network = network
        self.micro_batch_num = micro_batch_num
        self.concat = P.Concat()

    def construct(self, x):
        ret_x = ()
        ret_y = ()
        for i in range(self.micro_batch_num):
            micro_batch_size = x.shape[0] // self.micro_batch_num
            start = micro_batch_size * i
            end = micro_batch_size * (i + 1)

            micro_input = x[start:end]
            ret1, ret2 = self.network(micro_input)
            ret_x = ret_x + (ret1,)
            ret_y = ret_y + (ret2,)

        ret_x = self.concat(ret_x)
        ret_y = self.concat(ret_y)
        return ret_x, ret_y


def get_stage_id():
    rank_size = D.get_group_size()
    rank_id = D.get_rank()
    rank_per_stage = rank_size // pipeline_stages
    return rank_id // rank_per_stage


def test_pipeline_inference_basic():
    """
    Feature: Pipeline parallel inference
    Description: Micro batch split
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True,
                                      pipeline_stages=pipeline_stages)
    net = PipelineCellInference(Net(), micro_batch_num=2)
    net.set_train(False)

    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    ret = net(x)

    expect = [[np.zeros(shape, np.float32), np.zeros(shape, np.float32)],
              [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 4) * 2]]
    is_last_stage = get_stage_id() == pipeline_stages - 1
    assert np.allclose(ret[0].asnumpy(), expect[is_last_stage][0])
    assert np.allclose(ret[1].asnumpy(), expect[is_last_stage][1])


def test_pipeline_inference_no_lazy_inline_basic():
    """
    Feature: Pipeline parallel inference with no lazy inline
    Description: Micro batch split
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True,
                                      pipeline_stages=pipeline_stages)
    net = PipelineCellInference(NoLazyNet(), micro_batch_num=2)
    net.set_train(False)

    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    ret = net(x)

    expect = [[np.zeros(shape, np.float32), np.zeros(shape, np.float32)],
              [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 4) * 2]]
    is_last_stage = get_stage_id() == pipeline_stages - 1
    assert np.allclose(ret[0].asnumpy(), expect[is_last_stage][0])
    assert np.allclose(ret[1].asnumpy(), expect[is_last_stage][1])


def test_pipeline_inference_broadcast():
    """
    Feature: Pipeline parallel inference
    Description: Broadcast last stage result, multi-output.
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True,
                                      pipeline_stages=pipeline_stages, pipeline_result_broadcast=True)
    net = PipelineCellInference(Net(), micro_batch_num=4)
    net.set_train(False)

    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    ret = net(x)
    print(ret)

    expect = [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 4) * 2]
    assert np.allclose(ret[0].asnumpy(), expect[0])
    assert np.allclose(ret[1].asnumpy(), expect[1])


def test_pipeline_inference_single_micro_batch():
    """
    Feature: Pipeline parallel inference
    Description: Broadcast last stage result, without micro batch split
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True,
                                      pipeline_stages=pipeline_stages, pipeline_result_broadcast=True)
    net = PipelineCellInference(Net(), micro_batch_num=1)
    net.set_train(False)

    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    ret = net(x)

    print(ret)
    expect = [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 4) * 2]
    assert np.allclose(ret[0].asnumpy(), expect[0])
    assert np.allclose(ret[1].asnumpy(), expect[1])


def test_pipeline_inference_without_wrapper():
    """
    Feature: Pipeline parallel inference
    Description: Broadcast last stage result, without wrapper
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True,
                                      pipeline_stages=pipeline_stages, pipeline_result_broadcast=True)

    net = PipelineCellInference(Net(), micro_batch_num=1)
    net.set_train(False)

    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    ret = net(x)
    print(ret)
    expect = [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 4) * 2]
    assert np.allclose(ret[0].asnumpy(), expect[0])
    assert np.allclose(ret[1].asnumpy(), expect[1])


class PipelineCellInferenceSingleOutput(nn.Cell):
    def __init__(self, network, micro_batch_num):
        super().__init__()
        self.network = network
        self.micro_batch_num = micro_batch_num
        self.concat = P.Concat()

    def construct(self, x):
        ret = ()
        for i in range(self.micro_batch_num):
            micro_batch_size = x.shape[0] // self.micro_batch_num
            start = micro_batch_size * i
            end = micro_batch_size * (i + 1)

            micro_input = x[start:end]
            ret1, _ = self.network(micro_input)
            ret = ret + (ret1,)

        ret = self.concat(ret)
        return ret


def test_pipeline_inference_single_output():
    """
    Feature: Pipeline parallel inference
    Description: Micro batch split
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True,
                                      pipeline_stages=pipeline_stages, pipeline_result_broadcast=True)
    net = PipelineCellInferenceSingleOutput(Net(), micro_batch_num=2)
    net.set_train(False)

    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    ret = net(x)

    print(ret)
    expect = [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 5)]
    is_last_stage = get_stage_id() == pipeline_stages - 1
    assert np.allclose(ret.asnumpy(), expect[is_last_stage])


def test_pipeline_inference_shared_params():
    """
    Feature: Pipeline parallel inference
    Description: Shared parameters synchronize
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True,
                                      pipeline_stages=pipeline_stages, pipeline_result_broadcast=True)
    net = Net()

    if get_stage_id() == pipeline_stages - 1:
        shape, dtype = net.word_embedding.w.shape, net.word_embedding.w.dtype
        net.word_embedding.w.set_data(Tensor(np.zeros(shape), dtype))

    net = PipelineCellInference(net, micro_batch_num=1)
    net.set_train(False)
    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    # compile and synchronize
    net.compile(x)
    sync_pipeline_shared_parameters(net)

    ret = net(x)
    print(ret)
    expect = [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 4) * 2]
    assert np.allclose(ret[0].asnumpy(), expect[0])
    assert np.allclose(ret[1].asnumpy(), expect[1])


def test_pipeline_inference_data_parallel():
    """
    Feature: Pipeline parallel inference
    Description: Micro batch split
    Expectation: success
    """
    D.init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=False,
                                      pipeline_stages=pipeline_stages, pipeline_result_broadcast=True)
    net = PipelineCellInferenceSingleOutput(Net(), micro_batch_num=2)
    net.set_train(False)

    shape = (8, 8)
    x = Tensor(np.ones(shape), mindspore.float32)
    ret = net(x)

    print(ret)
    expect = [np.ones(shape, np.float32) * pow(8, 5), np.ones(shape, np.float32) * pow(8, 5)]
    is_last_stage = get_stage_id() == pipeline_stages - 1
    assert np.allclose(ret.asnumpy(), expect[is_last_stage])
