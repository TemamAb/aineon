const express = require('express');
const path = require('path');
const socketIo = require('socket.io');

const app = express();
const PORT = process.env.DASHBOARD_PORT || 3000;

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render('dashboard', {
        title: 'Ainexus Quantum Dashboard',
        flashCapacity: '$100,000,000',
        activeBots: 3,
        dailyProfit: 25000,
        systemStatus: 'OPERATIONAL'
    });
});

const server = app.listen(PORT, () => {
    console.log(`í³Š Dashboard running on port ${PORT}`);
});

// Socket.io for real-time updates
const io = socketIo(server);
io.on('connection', (socket) => {
    console.log('Dashboard client connected');
    
    // Send real-time profit updates
    setInterval(() => {
        socket.emit('profitUpdate', {
            timestamp: new Date(),
            profit: Math.random() * 1000 + 20000 // Mock data
        });
    }, 5000);
});
