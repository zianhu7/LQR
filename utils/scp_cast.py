import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", type=str, nargs='+')
    parser.add_argument("--files", type=str, nargs='+')
    args = parser.parse_args()
    files = ' '.join(args.files)
    for i in args.remote:
        fpath = " ubuntu@"+str(i)+":/home/ubuntu/"
        cmd = 'scp -i ethan.pem ' + files + fpath
        os.system(cmd)
