import mindspore_lite as mslite

from mindspore_lite.lib._c_lite_wrapper import StatusCode, Status


def init_stub():
    llm_engine = mslite.LLMEngine(mslite.LLMRole.Decoder, 0, batch_mode="manual")

    print("LLM Engine init end")
    llm_engine.inited_ = True
    cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
    print("start llm engine ")
    return llm_engine, cluster


def test_mocking_error_code(mocker):

    llm_engine, cluster = init_stub()

    def mock_link_clusters(self, clusters_inners, timeout):
        print("mock param_invalid_link status")
        ret = Status(StatusCode.kLiteParamInvalid)
        return ret, [Status()]

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.link_clusters', mock_link_clusters)

    try:
        ret, rets = llm_engine.link_clusters([cluster])
        assert ret.StatusCode() != mslite.LLMStatusCode.LLM_PARAM_INVALID.value
    except RuntimeError as e:
        print(e.statusCode)
        assert e.StatusCode() == mslite.LLMStatusCode.LLM_PARAM_INVALID, "link failed"

    def mock_failed_link_clusters(self, clusters_inners, timeout):
        print("mock success_link status")
        return Status(), (Status(StatusCode.kLiteLLMNotYetLink), Status(StatusCode.kLiteLLMLinkFailed),
                          Status(StatusCode.kLiteLLMClusterNumExceedLimit), Status(StatusCode.kLiteLLMProcessingLink))

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.link_clusters', mock_failed_link_clusters)

    ret, rets = llm_engine.link_clusters([cluster])
    assert ret.StatusCode() == mslite.LLMStatusCode.LLM_SUCCESS.value
    assert rets[0].StatusCode() == mslite.LLMStatusCode.LLM_NOT_YET_LINK.value, "link failed rets 0 error"
    assert rets[1].StatusCode() == mslite.LLMStatusCode.LLM_LINK_FAILED.value
    assert rets[2].StatusCode() == mslite.LLMStatusCode.LLM_CLUSTER_NUM_EXCEED_LIMIT.value
    assert rets[3].StatusCode() == mslite.LLMStatusCode.LLM_PROCESSING_LINK.value

    def mock_success_unlink_clusters(self, clusters_inners, timeout):
        print("mock unlink status")
        ret = Status()
        return ret, [Status(StatusCode.kLiteLLMNotYetLink), Status(StatusCode.kLiteLLMLinkFailed),
                     Status(StatusCode.kLiteLLMClusterNumExceedLimit), Status(StatusCode.kLiteLLMProcessingLink)]

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_success_unlink_clusters)

    ret, rets = llm_engine.unlink_clusters([cluster])
    assert ret.StatusCode() == mslite.LLMStatusCode.LLM_SUCCESS.value
    assert rets[0].StatusCode() == mslite.LLMStatusCode.LLM_NOT_YET_LINK.value
    assert rets[1].StatusCode() == mslite.LLMStatusCode.LLM_LINK_FAILED.value
    assert rets[2].StatusCode() == mslite.LLMStatusCode.LLM_CLUSTER_NUM_EXCEED_LIMIT.value
    assert rets[3].StatusCode() == mslite.LLMStatusCode.LLM_PROCESSING_LINK.value

    def mock_unlink_clusters_param_invalid(self, clusters_inners, timeout):
        print("mock unlink param invalid status")
        ret = Status(StatusCode.kLiteParamInvalid)
        return ret, [Status()]

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_unlink_clusters_param_invalid)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "unlink failed"
    except RuntimeError as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_PARAM_INVALID, "unlink failed"
