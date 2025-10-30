const express = require('express');
const app = express();

app.use(express.static('.'));

app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ì∫Ä AINEON Arbitrage Dashboard</title>
        <style>
            body { 
                font-family: 'Courier New', monospace; 
                margin: 0; 
                padding: 20px; 
                background: #0a0a0a; 
                color: #00ff00; 
                line-height: 1.6;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                border: 1px solid #00ff00;
                padding: 20px;
                background: #111111;
            }
            .header { 
                text-align: center; 
                border-bottom: 2px solid #00ff00; 
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            .status-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px;
            }
            .status-card { 
                background: #1a1a1a; 
                padding: 20px; 
                border: 1px solid #333; 
                border-radius: 5px;
            }
            .port { 
                color: #ffff00; 
                font-weight: bold;
            }
            .endpoint { 
                color: #00ffff; 
                font-family: monospace;
                font-size: 0.9em;
            }
            .btn { 
                background: #00ff00; 
                color: #000; 
                border: none; 
                padding: 10px 20px; 
                margin: 5px; 
                cursor: pointer; 
                font-family: 'Courier New';
                font-weight: bold;
            }
            .btn:hover { 
                background: #00cc00; 
            }
            #result { 
                margin-top: 20px; 
                padding: 15px; 
                background: #000; 
                border: 1px solid #333;
                border-radius: 5px;
                max-height: 400px;
                overflow-y: auto;
            }
            .success { color: #00ff00; }
            .error { color: #ff0000; }
            .warning { color: #ffff00; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Ì∫Ä AINEON ARBITRAGE FLASH LOAN DASHBOARD</h1>
                <p>Advanced AI-Powered Multi-Chain Trading Platform</p>
            </div>

            <div class="status-grid">
                <div class="status-card">
                    <h2>Ì∂•Ô∏è Frontend Server</h2>
                    <p><strong>Port:</strong> <span class="port">4000</span></p>
                    <p><strong>Status:</strong> <span class="success">OPERATIONAL</span></p>
                    <p><strong>URL:</strong> <span class="endpoint">http://localhost:8080</span></p>
                </div>

                <div class="status-card">
                    <h2>Ì¥ß Backend API</h2>
                    <p><strong>Port:</strong> <span class="port">3000</span></p>
                    <p><strong>Status:</strong> <span id="backend-status">CHECKING...</span></p>
                    <p><strong>URL:</strong> <span class="endpoint">http://localhost:3000/api/health</span></p>
                </div>

                <div class="status-card">
                    <h2>Ì¥ó API Gateway</h2>
                    <p><strong>Port:</strong> <span class="port">5000</span></p>
                    <p><strong>Status:</strong> <span id="gateway-status">CHECKING...</span></p>
                    <p><strong>URL:</strong> <span class="endpoint">http://localhost:5000/api/health</span></p>
                </div>
            </div>

            <div class="status-card">
                <h2>ÌæØ Dashboard Controls</h2>
                <button class="btn" onclick="testHealth()">Test Health Check</button>
                <button class="btn" onclick="testModules()">Test Modules Status</button>
                <button class="btn" onclick="testTrading()">Test Trading Performance</button>
                <button class="btn" onclick="testBlockchain()">Test Blockchain Status</button>
                <button class="btn" onclick="clearResults()">Clear Results</button>
                
                <div id="result"></div>
            </div>

            <div class="status-card">
                <h2>Ì≥ä System Information</h2>
                <p><strong>Deployment:</strong> Docker Containerized</p>
                <p><strong>Frontend:</strong> Express.js + Static Serving</p>
                <p><strong>Backend:</strong> Node.js + REST API</p>
                <p><strong>Ports Allocated:</strong> 3000, 4000, 5000</p>
                <p><strong>Status:</strong> <span class="success">READY FOR INTEGRATION</span></p>
            </div>
        </div>

        <script>
            // Auto-check services on load
            window.addEventListener('load', function() {
                checkBackendStatus();
                checkGatewayStatus();
            });

            async function checkBackendStatus() {
                try {
                    const response = await fetch('http://localhost:3000/api/health');
                    const data = await response.json();
                    document.getElementById('backend-status').innerHTML = '<span class="success">OPERATIONAL</span>';
                } catch (error) {
                    document.getElementById('backend-status').innerHTML = '<span class="error">OFFLINE</span>';
                }
            }

            async function checkGatewayStatus() {
                try {
                    const response = await fetch('http://localhost:5000/api/health');
                    const data = await response.json();
                    document.getElementById('gateway-status').innerHTML = '<span class="success">OPERATIONAL</span>';
                } catch (error) {
                    document.getElementById('gateway-status').innerHTML = '<span class="error">OFFLINE</span>';
                }
            }

            async function testHealth() {
                showResult('Testing Backend Health...', 'warning');
                try {
                    const response = await fetch('http://localhost:3000/api/health');
                    const data = await response.json();
                    showResult('‚úÖ Health Check Successful:\n' + JSON.stringify(data, null, 2), 'success');
                } catch (error) {
                    showResult('‚ùå Health Check Failed: ' + error.message, 'error');
                }
            }

            async function testModules() {
                showResult('Testing Modules Status...', 'warning');
                try {
                    const response = await fetch('http://localhost:3000/api/modules/status');
                    const data = await response.json();
                    showResult('‚úÖ Modules Status:\n' + JSON.stringify(data, null, 2), 'success');
                } catch (error) {
                    showResult('‚ùå Modules Test Failed: ' + error.message, 'error');
                }
            }

            async function testTrading() {
                showResult('Testing Trading Performance...', 'warning');
                try {
                    const response = await fetch('http://localhost:3000/api/trading/performance');
                    const data = await response.json();
                    showResult('‚úÖ Trading Performance:\n' + JSON.stringify(data, null, 2), 'success');
                } catch (error) {
                    showResult('‚ùå Trading Test Failed: ' + error.message, 'error');
                }
            }

            async function testBlockchain() {
                showResult('Testing Blockchain Status...', 'warning');
                try {
                    const response = await fetch('http://localhost:3000/api/blockchain/status');
                    const data = await response.json();
                    showResult('‚úÖ Blockchain Status:\n' + JSON.stringify(data, null, 2), 'success');
                } catch (error) {
                    showResult('‚ùå Blockchain Test Failed: ' + error.message, 'error');
                }
            }

            function showResult(message, type) {
                const result = document.getElementById('result');
                result.innerHTML = '<pre class="' + type + '">' + message + '</pre>';
                result.scrollTop = result.scrollHeight;
            }

            function clearResults() {
                document.getElementById('result').innerHTML = '';
            }
        </script>
    </body>
    </html>
  `);
});

app.listen(4000, () => {
  console.log('Ì∂•Ô∏è AINEON Frontend Dashboard running on port 4000');
  console.log('Ì≥ä Access: http://localhost:8080');
});
