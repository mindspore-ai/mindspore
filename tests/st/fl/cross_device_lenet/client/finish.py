import argparse
import subprocess

parser = argparse.ArgumentParser(description="Finish client process")
# The parameter `--kill_tag` is used to search for the keyword to kill the client process.
parser.add_argument("--kill_tag", type=str, default="mindspore-lite-java-flclient")

args, _ = parser.parse_known_args()
kill_tag = args.kill_tag

cmd = "pid=`ps -ef|grep " + kill_tag
cmd += " |grep -v \"grep\" | grep -v \"finish\" |awk '{print $2}'` && "
cmd += "for id in $pid; do kill -9 $id && echo \"killed $id\"; done"

subprocess.call(['bash', '-c', cmd])
