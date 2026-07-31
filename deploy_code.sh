#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_host="${1:-192.168.10.102}"
remote_user="${ROBOT_USER:-booster}"
remote_dir="${ROBOT_DIR:-/home/booster/obstacle_run/}"

echo "Deploying obstacle_run to ${remote_user}@${remote_host}:${remote_dir}"

ssh "${remote_user}@${remote_host}" "mkdir -p '${remote_dir}'"

rsync -av --info=flist2,name,progress \
  --force \
  --exclude '/build/' \
  --exclude '/install/' \
  --exclude '/log/' \
  --exclude '.git/' \
  --exclude '.git' \
  --exclude '/.deploy/' \
  --exclude '**/__pycache__/' \
  --exclude '*.pyc' \
  "${repo_dir}/" \
  "${remote_user}@${remote_host}:${remote_dir}/"

echo "Deployment ended"
echo "On the robot: cd ${remote_dir} && ./scripts/build.sh"