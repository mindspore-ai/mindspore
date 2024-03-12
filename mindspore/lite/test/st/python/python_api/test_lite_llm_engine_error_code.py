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

    def mock_param_invalid(self, clusters_inners, timeout):
        print("mock param_invalid error")
        ret = Status(StatusCode.kLiteParamInvalid)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_param_invalid)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_PARAM_INVALID failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_PARAM_INVALID, "LLM_PARAM_INVALID failed"

    def mock_handle_WaitProcessTimeOut(self, clusters_inners, timeout):
        print("mock TimeOut error")
        ret = Status(StatusCode.kLiteLLMWaitProcessTimeOut)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_WaitProcessTimeOut)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_WAIT_PROC_TIMEOUT, "LLM_WAIT_PROC_TIMEOUT failed"

    def mock_handle_KVCacheNotExist(self, clusters_inners, timeout):
        print("mock KVCacheNotExist Error")
        ret = Status(StatusCode.kLiteLLMKVCacheNotExist)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_KVCacheNotExist)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_KV_CACHE_NOT_EXIST failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_KV_CACHE_NOT_EXIST, "LLM_KV_CACHE_NOT_EXIST failed"

    def mock_handle_LLM_REPEAT_REQUEST(self, clusters_inners, timeout):
        print("mock LLM_REPEAT_REQUEST ERROR")
        ret = Status(StatusCode.kLiteLLMRepeatRequest)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_LLM_REPEAT_REQUEST)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_REPEAT_REQUEST failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_REPEAT_REQUEST, "LLM_REPEAT_REQUEST failed"

    def mock_handle_LLM_REQUEST_ALREADY_COMPLETED(self, clusters_inners, timeout):
        print("mock RequestAlreadyCompleted ERROR")
        ret = Status(StatusCode.kLiteLLMRequestAlreadyCompleted)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters',
                 mock_handle_LLM_REQUEST_ALREADY_COMPLETED)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_REQUEST_ALREADY_COMPLETED failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_REQUEST_ALREADY_COMPLETED, "REQUEST_ALREADY_COMPLETED failed"

    def mock_handle_LLM_ENGINE_FINALIZED(self, clusters_inners, timeout):
        print("mock LLM_ENGINE_FINALIZED ERROR")
        ret = Status(StatusCode.kLiteLLMEngineFinalized)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_LLM_ENGINE_FINALIZED)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_ENGINE_FINALIZED failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_ENGINE_FINALIZED, "LLM_ENGINE_FINALIZED failed"

    def mock_handle_LLM_NOT_YET_LINK(self, clusters_inners, timeout):
        print("mock LLM_NOT_YET_LINK ERROR")
        ret = Status(StatusCode.kLiteLLMNotYetLink)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_LLM_NOT_YET_LINK)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_NOT_YET_LINK failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_NOT_YET_LINK, "LLM_NOT_YET_LINK failed"

    def mock_handle_LLM_DEVICE_OUT_OF_MEMORY(self, clusters_inners, timeout):
        print("mock LLM_DEVICE_OUT_OF_MEMOR ERROR")
        ret = Status(StatusCode.kLiteLLMOutOfMemory)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters',
                 mock_handle_LLM_DEVICE_OUT_OF_MEMORY)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_DEVICE_OUT_OF_MEMORY failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_DEVICE_OUT_OF_MEMORY, "LLM_DEVICE_OUT_OF_MEMORY failed"

    def mock_handle_LLM_PREFIX_ALREADY_EXIST(self, clusters_inners, timeout):
        print("mock LLM_PREFIX_ALREADY_EXIST ERROR")
        ret = Status(StatusCode.kLiteLLMPrefixAlreadyExist)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters',
                 mock_handle_LLM_PREFIX_ALREADY_EXIST)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_PREFIX_ALREADY_EXIST failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_PREFIX_ALREADY_EXIST, "LLM_PREFIX_ALREADY_EXIST failed"

    def mock_handle_LLM_PREFIX_NOT_EXIST(self, clusters_inners, timeout):
        print("mock LLM_PREFIX_NOT_EXIST ERROR")
        ret = Status(StatusCode.kLiteLLMPrefixNotExist)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_LLM_PREFIX_NOT_EXIST)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_PREFIX_NOT_EXIST failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_PREFIX_NOT_EXIST, "LLM_PREFIX_NOT_EXIST failed"

    def mock_handle_LLM_SEQ_LEN_OVER_LIMIT(self, clusters_inners, timeout):
        print("mock LLM_SEQ_LEN_OVER_LIMIT ERROR")
        ret = Status(StatusCode.kLiteLLMSeqLenOverLimit)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_LLM_SEQ_LEN_OVER_LIMIT)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_SEQ_LEN_OVER_LIMIT failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_SEQ_LEN_OVER_LIMIT, "LLM_SEQ_LEN_OVER_LIMIT failed"

    def mock_handle_LLM_NO_FREE_BLOCK(self, clusters_inners, timeout):
        print("mock LLM_NO_FREE_BLOCK ERROR")
        ret = Status(StatusCode.kLiteLLMNoFreeBlock)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters', mock_handle_LLM_NO_FREE_BLOCK)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_NO_FREE_BLOCK failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_NO_FREE_BLOCK, "LLM_NO_FREE_BLOCK failed"

    def mock_handle_LLM_BLOCKS_OUT_OF_MEMORY(self, clusters_inners, timeout):
        print("mock LLM_BLOCKS_OUT_OF_MEMORY ERROR")
        ret = Status(StatusCode.kLiteLLMBlockOutOfMemory)
        return ret, []

    mocker.patch('mindspore_lite.lib._c_lite_wrapper.LLMEngine_.unlink_clusters',
                 mock_handle_LLM_BLOCKS_OUT_OF_MEMORY)
    try:
        ret, rets = llm_engine.unlink_clusters([cluster])
        assert ret.StatusCode != mslite.LLMStatusCode.LLM_SUCCESS.value, "LLM_BLOCKS_OUT_OF_MEMORY failed"
    except mslite.LLMException as e:
        print(e.statusCode)
        assert e.statusCode == mslite.LLMStatusCode.LLM_BLOCKS_OUT_OF_MEMORY, "LLM_BLOCKS_OUT_OF_MEMORY failed"
