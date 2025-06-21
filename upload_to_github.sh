#!/bin/bash

echo "🚀 GitHub 업로드 스크립트"
echo "========================="

# 사용자에게 GitHub 저장소 URL 입력 요청
echo "📝 GitHub에서 새 저장소를 생성한 후, 저장소 URL을 입력하세요:"
echo "예시: https://github.com/yourusername/analog-clock-reading-optimization.git"
read -p "저장소 URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "❌ 저장소 URL이 입력되지 않았습니다."
    exit 1
fi

echo "🔗 Remote origin 설정 중..."
git remote add origin "$REPO_URL"

echo "🌿 Main 브랜치로 변경 중..."
git branch -M main

echo "📤 GitHub에 업로드 중..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ 업로드 성공!"
    echo "🌟 저장소 확인: ${REPO_URL%.git}"
else
    echo "❌ 업로드 실패. 저장소 URL과 권한을 확인하세요."
fi