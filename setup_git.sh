#!/bin/bash
# Git 仓库初始化和推送脚本

set -e  # 遇到错误立即退出

echo "================================"
echo "PTCUT Git 仓库初始化"
echo "================================"

# 进入项目目录
cd /home/lzh/myCode/PTCUT

# 1. 初始化 Git
echo ""
echo "[1/6] 初始化 Git 仓库..."
git init
echo "✓ Git 仓库初始化完成"

# 2. 配置用户信息（如果需要的话）
echo ""
echo "[2/6] 配置 Git 用户信息..."
read -p "请输入您的 Git 用户名: " git_username
read -p "请输入您的 Git 邮箱: " git_email
git config user.name "$git_username"
git config user.email "$git_email"
echo "✓ 用户信息配置完成"

# 3. 添加文件
echo ""
echo "[3/6] 添加文件到暂存区..."
git add .
echo "✓ 文件添加完成"

# 查看状态
echo ""
echo "将要提交的文件："
git status --short

# 4. 提交
echo ""
echo "[4/6] 创建首次提交..."
git commit -m "Initial commit: PTCUT project

- Added PTCUT model implementation
- Added training and testing scripts  
- Added documentation
"
echo "✓ 首次提交完成"

# 5. 添加远程仓库
echo ""
echo "[5/6] 添加远程仓库..."
echo "请先在 GitHub/GitLab 创建远程仓库"
read -p "请输入远程仓库 URL (例: https://github.com/username/PTCUT.git): " remote_url

if [ -z "$remote_url" ]; then
    echo "未输入远程 URL，跳过推送步骤"
    echo "稍后可以手动添加："
    echo "  git remote add origin <your-repo-url>"
    echo "  git push -u origin main"
    exit 0
fi

git remote add origin "$remote_url"
echo "✓ 远程仓库添加完成"

# 6. 推送到远程
echo ""
echo "[6/6] 推送到远程仓库..."
echo "注意：如果远程默认分支是 master，请手动修改命令"

# 检查当前分支名
current_branch=$(git branch --show-current)
echo "当前分支: $current_branch"

read -p "是否推送到远程? (y/n): " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    git push -u origin "$current_branch"
    echo "✓ 推送完成"
else
    echo "跳过推送，稍后可以手动执行："
    echo "  git push -u origin $current_branch"
fi

echo ""
echo "================================"
echo "✅ Git 仓库设置完成！"
echo "================================"
echo ""
echo "常用命令："
echo "  git status          # 查看状态"
echo "  git add .           # 添加所有更改"
echo "  git commit -m 'msg' # 提交更改"
echo "  git push            # 推送到远程"
echo "  git pull            # 拉取远程更新"
