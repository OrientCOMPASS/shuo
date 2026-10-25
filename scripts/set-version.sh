#!/usr/bin/env bash
# 把版本号写进构建配置（CI 专用）：
#   src-tauri/tauri.conf.json
#   src-tauri/Cargo.toml
#   src-tauri/Cargo.lock（只改 name = "shuo" 的 [[package]]）
#
# 版本号的唯一来源是 tag；CI 在构建前把 tag 里的版本传给这个脚本。
# 本地不需要手改版本号，也不要提交这种改动。
#
# 用法：./scripts/set-version.sh 26.10.25
set -euo pipefail

cd "$(dirname "$0")/.."
v="${1:?用法: ./scripts/set-version.sh <版本>}"

python3 - "$v" <<'PY'
import json, pathlib, re, sys

v = sys.argv[1]

conf = pathlib.Path("src-tauri/tauri.conf.json")
d = json.loads(conf.read_text(encoding="utf-8"))
d["version"] = v
conf.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

cargo = pathlib.Path("src-tauri/Cargo.toml")
s = cargo.read_text(encoding="utf-8")
s, n = re.subn(r'(?m)^version = "[^"]+"', f'version = "{v}"', s, count=1)
if n != 1:
    raise SystemExit("Cargo.toml 里没找到顶层 version")
cargo.write_text(s, encoding="utf-8")

lock = pathlib.Path("src-tauri/Cargo.lock")
if lock.exists():
    ls = lock.read_text(encoding="utf-8")
    ls, n = re.subn(
        r'(\[\[package\]\]\nname = "shuo"\nversion = ")[^"]+(")',
        lambda m: m.group(1) + v + m.group(2),
        ls,
        count=1,
    )
    if n == 1:
        lock.write_text(ls, encoding="utf-8")
PY

echo "版本号已写入构建配置：$v"
