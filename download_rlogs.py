#!/usr/bin/env python3
"""从 comma 设备批量下载最近的 rlog 数据"""

import paramiko
import os
import stat

def main():
    """主函数：连接设备并下载最近 300 条路线的 rlog 文件"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('192.168.31.13', username='comma', key_filename=r'C:\Users\01\.ssh\id_ed25519')
    sftp = client.open_sftp()

    device_path = '/data/media/0/realdata/'
    output_dir = './data'
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有目录，按修改时间排序（最近的在前）
    entries = sftp.listdir_attr(device_path)
    dirs = [(e.filename, e.st_mtime) for e in entries if stat.S_ISDIR(e.st_mode) and e.filename != 'boot']
    dirs.sort(key=lambda x: x[1], reverse=True)
    routes = [d[0] for d in dirs[:300]]

    print(f'Found {len(routes)} recent routes')

    synced = 0
    skipped = 0
    errors = 0

    for i, route in enumerate(routes):
        remote_route = device_path + route
        try:
            files = sftp.listdir(remote_route)
        except Exception:
            errors += 1
            continue

        for fname in files:
            if not fname.startswith('rlog'):
                continue
            remote_file = remote_route + '/' + fname
            local_dir = os.path.join(output_dir, route)
            local_file = os.path.join(local_dir, fname)

            if os.path.exists(local_file):
                try:
                    remote_stat = sftp.stat(remote_file)
                    if os.path.getsize(local_file) == remote_stat.st_size:
                        skipped += 1
                        continue
                except Exception:
                    pass

            os.makedirs(local_dir, exist_ok=True)
            sftp.get(remote_file, local_file)
            synced += 1

        if (i + 1) % 50 == 0:
            print(f'Progress: {i+1}/{len(routes)} routes, synced={synced}, skipped={skipped}', flush=True)

    sftp.close()
    client.close()
    print(f'Done: synced={synced}, skipped={skipped}, errors={errors}')

if __name__ == '__main__':
    main()
