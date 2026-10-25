#!/usr/bin/env bash
# 说 · Shuo — 发版脚本
#
# 版本号规则：日期 YY.MM.DD（取本机日期），tag = v<版本>。
# 同一天第二次发版自动用 -2、-3…（semver 预发布段；deb 里等同修订号），避免撞 tag。
#
# 用法：./scripts/release.sh
#   1) 检查在 main、工作区干净、远端 tag 已同步
#   2) 把版本写进 src-tauri/tauri.conf.json、Cargo.toml、Cargo.lock
#   3) 提交 chore(release): <版本>
#   4) 打 tag v<版本>，推送 main 和 tag
#   5) GitHub Actions（.github/workflows/release.yml）自动构建
#      Windows NSIS + Linux deb/AppImage，并创建同名 Release
#
# 如果只想构建不想发版：GitHub Actions 页面手动 workflow_dispatch
# （Release 作业只在 tag 触发时运行）。
set -euo pipefail

cd "$(dirname "$0")/.."

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "请在 main 上发版（当前分支：$branch）" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "工作区不干净，先提交或还原：" >&2
  git status --short >&2
  exit 1
fi

# 远端 tag 可能是别的机器推的，先同步再算版本
git fetch --tags --quiet origin

base=$(date +%y.%m.%d)
version=$base
i=1
while git rev-parse -q --verify "refs/tags/v$version" >/dev/null; do
  i=$((i + 1))
  version="$base-$i"
done

python3 - "$version" <<'PY'
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
    # 只改 [[package]] name = "shuo" 这一块，别动依赖
    ls, n = re.subn(
        r'(\[\[package\]\]\nname = "shuo"\nversion = ")[^"]+(")',
        lambda m: m.group(1) + v + m.group(2),
        ls,
        count=1,
    )
    if n == 1:
        lock.write_text(ls, encoding="utf-8")
PY

git add src-tauri/tauri.conf.json src-tauri/Cargo.toml
if [ -f src-tauri/Cargo.lock ]; then
  git add src-tauri/Cargo.lock
fi

if git diff --cached --quiet; then
  echo "版本号已是 $version，不产生空提交"
else
  git commit -m "chore(release): $version"
fi

git tag -a "v$version" -m "说 · Shuo $version"
git push origin main
git push origin "v$version"

url=$(git remote get-url origin |
  sed -E 's#^git@github\.com:#https://github.com/#; s#^https://github\.com/#https://github.com/#; s#\.git$##')

echo
echo "已推送 v$version"
echo "  发版：$url/releases/tag/v$version"
echo "  构建：$url/actions/workflows/release.yml"
