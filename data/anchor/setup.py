import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--libvpx_dir', type=str, required=True)
    parser.add_argument('--binary_dir', type=str, required=True)
    args = parser.parse_args()

    print(args.libvpx_dir)

    # configure
    cmd = 'cd {} && ./nemo_server.sh'.format(args.libvpx_dir)
    os.system(cmd)

    # build
    cmd = 'cd {} && make'.format(args.libvpx_dir)
    os.system(cmd)

    # copy
    os.makedirs(args.binary_dir, exist_ok=True)
    name = 'vpxdec_nemo_ver2'
    src_binary = os.path.join(args.libvpx_dir, name)
    dest_binary = os.path.join(args.binary_dir, name)
    shutil.copy(src_binary, dest_binary)
