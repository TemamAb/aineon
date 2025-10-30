// Real-time Capital Allocation Normalization Engine
const sliders = {
  flashLoan: flashLoanCapitalSlider.value || 0,
  crossChain: crossChainCapitalSlider.value || 0,
  mevBots: mevBotsCapitalSlider.value || 0,
  reserve: reserveCapitalSlider.value || 0
};

const total = Object.values(sliders).reduce((sum, val) => sum + val, 0);

if (total > 100) {
  // Auto-normalize proportions
  const scaleFactor = 100 / total;
  return {
    flashLoan: Math.round(sliders.flashLoan * scaleFactor),
    crossChain: Math.round(sliders.crossChain * scaleFactor),
    mevBots: Math.round(sliders.mevBots * scaleFactor),
    reserve: Math.round(sliders.reserve * scaleFactor),
    normalized: true,
    message: `Auto-normalized from ${total}% to 100%`
  };
}

return {
  ...sliders,
  normalized: false,
  valid: total === 100,
  message: total === 100 ? 'Allocation valid' : `Allocation ${total}% - needs adjustment`
};
