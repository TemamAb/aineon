FROM node:18-alpine
WORKDIR /app
COPY package.json package-lock.json ./
COPY hardhat.config.js ./
RUN npm install
COPY contracts/ ./contracts/
COPY scripts/ ./scripts/
COPY backend/ ./backend/
COPY artifacts/ ./artifacts/
RUN npx hardhat compile
EXPOSE 10000
CMD ["node", "backend/server.js"]
