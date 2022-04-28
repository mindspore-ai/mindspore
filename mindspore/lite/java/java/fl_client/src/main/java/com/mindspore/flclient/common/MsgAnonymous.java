package com.mindspore.flclient.common;

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This Class is used to anonymous sensitive msg.
 * @author       : zhangzhaoju
 * @since  : 2022/4/17
 */
public class MsgAnonymous {
    private ArrayList<Anonymous> anonymouses;
    private boolean anonyed = false;
    private String result;

    MsgAnonymous() {
        anonymouses = new ArrayList<>();
        anonymouses.add(new UrlAnonymous());
        anonymouses.add(new PathAnonymous());
        anonymouses.add(new PathArrayAnonymous());
        anonymouses.add(new HostAnonymous());
        anonymouses.add(new IPAnonymous());
    }

    /**
     * For more performance, not support one msg matches multi Anonymous(like one msg content has both url and path)
     *
     * @param msg
     * @return
     */
    public String doAnonymous(String msg) {
        result = msg;
        for (Anonymous anonymous : anonymouses) {
            if (anonymous.doAnonymous(msg)) {
                break;
            }
        }
        return result;
    }

    protected interface Anonymous {
        public Boolean doAnonymous(String msg);
    }

    protected class UrlAnonymous implements Anonymous {
        public Boolean doAnonymous(String msg) {
            Pattern urlPattern = Pattern.compile("(http[s]?://)([a-zA-Z0-9.:]*)");
            Matcher urlMatcher = urlPattern.matcher(msg);
            if (!urlMatcher.find()) {
                return false;
            }
            result = urlMatcher.replaceAll("$1AnonymousDomain");
            return true;
        }
    }

    protected class PathAnonymous implements Anonymous {
        public Boolean doAnonymous(String msg) {
            Pattern pathPattern = Pattern.compile("([Pp][Aa][Tt][Hh][ ]?:[ ]?)(/[a-zA-Z0-9._]*)+");
            Matcher pathMatcher = pathPattern.matcher(msg);
            if (!pathMatcher.find()) {
                return false;
            }
            result = pathMatcher.replaceAll("$1/AnonymousPath$2");
            return true;
        }
    }

    protected class PathArrayAnonymous implements Anonymous {
        public Boolean doAnonymous(String msg) {
            Pattern pathArrayPattern = Pattern.compile("([Pp][Aa][Tt][Hh][ ]?:[ ]?)\\[(/[a-zA-Z0-9\\._]+[, ]*)+\\]");
            Matcher pathArrayMatcher = pathArrayPattern.matcher(msg);
            if (!pathArrayMatcher.find()) {
                return false;
            }
            result = pathArrayMatcher.replaceAll("$1/AnonymousPath$2");
            return true;
        }
    }

    protected class IPAnonymous implements Anonymous {
        public Boolean doAnonymous(String msg) {
            Pattern IPPattern = Pattern.compile("([/ ]+)([0-9]+.[0-9]+.[0-9]+.[0-9]+)([: ]+)([0-9]*)");
            Matcher IPMatcher = IPPattern.matcher(msg);
            if (!IPMatcher.find()) {
                return false;
            }
            result = IPMatcher.replaceAll("$1AnonymousIP");
            return true;
        }
    }

    protected class HostAnonymous implements Anonymous {
        public Boolean doAnonymous(String msg) {
            Pattern hosstPattern = Pattern.compile("(host[ :]+)([\"a-zA-Z0-9.]*)");
            Matcher hostMatcher = hosstPattern.matcher(msg);
            if (!hostMatcher.find()) {
                return false;
            }
            result = hostMatcher.replaceAll("$1Anonymous");
            return true;
        }
    }

}
