import express from 'express';
import healthRouter from './api/health';

const app = express();
const PORT = process.env.PORT || 3000;

app.use('/api', healthRouter);
app.use(express.json());

app.listen(PORT, () => {
  console.log(`Arbitrage Flash Loan Engine running on port ${PORT}`);
});
