package com.mindspore.flclient;

import static org.junit.Assert.assertEquals;

/**   
 * The callback class for result check.
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
class FLUTResultCheck implements IFLJobResultCallback {
    private int[] expectIterCode;
    private int expectFinishCode;

    public FLUTResultCheck(int finishCode, int[] iterCode) {
        expectFinishCode = finishCode;
        expectIterCode = iterCode;
    }

    /**
     * Called at the end of an iteration for Fl job result check
     *
     * @param modelName    the name of model
     * @param iterationSeq Iteration number
     * @param resultCode   Status Code
     */
    @Override
    public void onFlJobIterationFinished(String modelName, int iterationSeq, int resultCode) {
//            assertEquals(resultCode, 200);
    }

    /**
     * Called on completion for Fl job result check
     *
     * @param modelName      the name of model
     * @param iterationCount total Iteration numbers
     * @param resultCode     Status Code
     */
    @Override
    public void onFlJobFinished(String modelName, int iterationCount, int resultCode) {
        assertEquals(expectFinishCode, resultCode);
    }
}
