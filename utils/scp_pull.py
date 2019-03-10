import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_ids", type=str, nargs='+')
    #Must be congruent fpaths once in remote instances
    parser.add_argument("--remote_fp", type=str)
    parser.add_argument("--files", type=str, nargs='+')
    parser.add_argument("--local_fp", type=str, nargs='+')
    args = parser.parse_args()
    for i in range(len(args.remote_ids)):
        remote_i = args.remote_ids[i]
        local_fp = args.local_fp[0] + '/exp_' + str(i)
        os.system('mkdir ' + local_fp)
        for j in args.files:
            fp = " ubuntu@" + remote_i + ":" + str(args.remote_fp) + str(j)
            cmd = 'scp -i ethan.pem ' + fp + ' ' + local_fp
            os.system(cmd)


