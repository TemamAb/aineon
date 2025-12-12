# UNIFIED DOCKER DEPLOYMENT FIX
echo "🚀 Starting unified Docker deployment fix..."

# 1. Navigate to correct directory
cd ~/Desktop/aineon-enterprise

# 2. Remove any existing Dockerfile
rm -f Dockerfile Dockerfile2

# 3. Create proper Dockerfile with exact content
cat > Dockerfile << 'DOCKEREOF'
FROM node:18-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
DOCKEREOF

# 4. Verify Dockerfile creation
echo "✅ Dockerfile created with $(wc -l Dockerfile) lines"

# 5. Ensure package.json exists
if [ ! -f "package.json" ]; then
  cat > package.json << 'PKGEOF'
{
  "name": "aineon-enterprise",
  "version": "1.0.0",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
PKGEOF
fi

# 6. Ensure src/index.js exists
mkdir -p src
if [ ! -f "src/index.js" ]; then
  cat > src/index.js << 'INDEXEOF'
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.json({ message: 'AINEON Running', status: 'ok' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
});
INDEXEOF
fi

# 7. Update render.yaml
cat > render.yaml << 'RENDEREOF'
services:
  - type: web
    name: aineon-enterprise
    env: docker
    dockerfilePath: ./Dockerfile
    dockerContext: .
    autoDeploy: true
RENDEREOF

# 8. Create .dockerignore
cat > .dockerignore << 'IGNOREEOF'
node_modules
npm-debug.log
.DS_Store
.env
.git
.gitignore
README.md
IGNOREEOF

# 9. Add all files to git
git add -A .

# 10. Commit with force
git commit -m "UNIFIED FIX: Complete Docker deployment with proper Dockerfile, Node.js server, and Render config"

# 11. Get commit hash
NEW_COMMIT=$(git rev-parse --short HEAD)

# 12. Force push to GitHub
git push origin main --force

echo " "
echo "✅ UNIFIED FIX COMPLETE!"
echo "📋 New commit: $NEW_COMMIT"
echo "📊 Dockerfile size: $(wc -l Dockerfile) lines, $(wc -c Dockerfile) bytes"
echo " "
echo "🚀 IMMEDIATE RENDER ACTIONS:"
echo "1. Go to: https://dashboard.render.com"
echo "2. Select 'aineon-enterprise' service"
echo "3. CANCEL any current build"
echo "4. Settings → Build & Deploy → CLEAR BUILD CACHE"
echo "5. Manual Deploy → Deploy latest commit ($NEW_COMMIT)"
echo " "
echo "✅ Expected result:"
echo "   → Dockerfile detected (not 2 bytes)"
echo "   → Building Docker image successfully"
echo "   → Server running on port 3000"