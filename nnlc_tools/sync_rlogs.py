#!/usr/bin/env python3
"""Sync rlog files from a comma device to a local directory."""

import argparse
import os
import posixpath
import shutil
import stat
import subprocess
import sys

DEFAULT_USER = "comma"
DEFAULT_DEVICE_PATH = "/data/media/0/realdata/"
RLOG_NAMES = {"rlog", "rlog.zst", "rlog.bz2"}


def sync_rsync(user, host, device_path, output_dir, dry_run=False):
    """Sync complete route/segment trees using rsync."""
    src = f"{user}@{host}:{device_path.rstrip('/')}/"
    cmd = [
        "rsync", "-avz", "--progress", "--partial",
        "--include=*/",
        "--include=rlog",
        "--include=rlog.zst",
        "--include=rlog.bz2",
        "--exclude=*",
        src, output_dir,
    ]
    if dry_run:
        cmd.insert(1, "--dry-run")
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def _iter_remote_files(sftp, root):
    """Yield (relative path, attributes) for every rlog below root."""
    stack = [(root.rstrip("/"), "")]
    while stack:
        current, relative = stack.pop()
        for entry in sftp.listdir_attr(current):
            remote_path = posixpath.join(current, entry.filename)
            rel_path = posixpath.join(relative, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                stack.append((remote_path, rel_path))
            elif entry.filename in RLOG_NAMES:
                yield rel_path, remote_path, entry


def sync_sftp(user, host, device_path, output_dir, dry_run=False):
    """Sync rlogs recursively with Paramiko when rsync is unavailable."""
    try:
        import paramiko
    except ImportError:
        print("ERROR: paramiko not installed. Install with: pip install paramiko")
        return False

    print(f"Connecting to {user}@{host} via SFTP...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        try:
            client.connect(host, username=user)
        except paramiko.AuthenticationException:
            connected = False
            for key_path in (
                os.path.expanduser("~/.ssh/id_ed25519"),
                os.path.expanduser("~/.ssh/id_rsa"),
            ):
                if os.path.exists(key_path):
                    try:
                        client.connect(host, username=user, key_filename=key_path)
                        connected = True
                        break
                    except paramiko.AuthenticationException:
                        continue
            if not connected:
                print("ERROR: Could not authenticate. Add your SSH key to the device.")
                return False

        sftp = client.open_sftp()
        synced = skipped = 0
        try:
            try:
                remote_files = list(_iter_remote_files(sftp, device_path))
            except (FileNotFoundError, OSError) as exc:
                print(f"ERROR: Remote path not found: {device_path} ({exc})")
                return False
            for relative, remote_file, remote_attr in remote_files:
                local_file = os.path.join(output_dir, *relative.split("/"))
                if os.path.exists(local_file) and os.path.getsize(local_file) == remote_attr.st_size:
                    skipped += 1
                    continue
                if dry_run:
                    print(f"  [dry-run] Would download: {remote_file}")
                    synced += 1
                    continue
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                print(f"  Downloading: {remote_file} -> {local_file}")
                sftp.get(remote_file, local_file)
                synced += 1
        finally:
            sftp.close()
    finally:
        client.close()

    action = "Would sync" if dry_run else "Synced"
    print(f"\n{action} {synced} files, skipped {skipped} (already present)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync rlog files from a comma device to a local directory.")
    parser.add_argument("-d", "--device", required=True, help="Device IP address")
    parser.add_argument("-o", "--output", required=True, help="Local output directory for rlogs")
    parser.add_argument("-u", "--user", default=DEFAULT_USER, help=f"SSH username (default: {DEFAULT_USER})")
    parser.add_argument("-p", "--path", default=DEFAULT_DEVICE_PATH, help=f"Device rlog path (default: {DEFAULT_DEVICE_PATH})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced without downloading")
    parser.add_argument("--no-rsync", action="store_true", help="Force SFTP mode")
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(output_dir, exist_ok=True)
    device_path = args.path.rstrip("/") + "/"
    if shutil.which("rsync") is not None and not args.no_rsync:
        success = sync_rsync(args.user, args.device, device_path, output_dir, args.dry_run)
    else:
        print("rsync not available, using SFTP fallback...")
        success = sync_sftp(args.user, args.device, device_path, output_dir, args.dry_run)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
