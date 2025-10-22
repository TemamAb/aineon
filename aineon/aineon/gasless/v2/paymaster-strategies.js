// SPONSORED TRANSACTION MODELS
class PaymasterStrategies {
  constructor() {
    this.strategies = {
      'subscription': this.subscriptionModel,
      'profit-sharing': this.profitSharingModel,
      'ad-sponsored': this.adSponsoredModel,
      'protocol-subsidized': this.protocolSubsidizedModel
    };
  }

  async selectOptimalPaymaster(userOperation, userContext) {
    const strategies = await this.evaluateAllStrategies(userOperation, userContext);
    const optimal = strategies.reduce((best, current) => 
      current.score > best.score ? current : best
    );

    return {
      paymaster: optimal.paymaster,
      strategy: optimal.strategy,
      data: optimal.data,
      cost: optimal.estimatedCost,
      score: optimal.score
    };
  }

  async subscriptionModel(userOperation, user) {
    const subscription = await this.getUserSubscription(user);
    if (!subscription.active) return { score: 0 };
    
    const monthlyCost = subscription.fee;
    const opsThisMonth = await this.getUserOperationsThisMonth(user);
    const costPerOp = monthlyCost / Math.max(opsThisMonth, 1);
    
    return {
      strategy: 'subscription',
      paymaster: subscription.paymaster,
      data: this.encodeSubscriptionData(subscription),
      estimatedCost: costPerOp,
      score: 0.8 - (costPerOp / 100) // Lower cost = higher score
    };
  }

  async profitSharingModel(userOperation, user) {
    const expectedProfit = await this.estimateOperationProfit(userOperation);
    const share = expectedProfit * 0.1; // 10% profit share
    
    return {
      strategy: 'profit-sharing',
      paymaster: this.profitSharePaymaster,
      data: this.encodeProfitShareData(share),
      estimatedCost: share,
      score: 0.7 + (expectedProfit > 0 ? 0.2 : 0)
    };
  }

  async protocolSubsidizedModel(userOperation, user) {
    const protocol = await this.detectProtocol(userOperation);
    const subsidy = await this.getProtocolSubsidy(protocol);
    
    if (subsidy.available) {
      return {
        strategy: 'protocol-subsidized',
        paymaster: subsidy.paymaster,
        data: this.encodeSubsidyData(subsidy),
        estimatedCost: 0,
        score: 0.9
      };
    }
    
    return { score: 0 };
  }

  encodeSubscriptionData(subscription) {
    return ethers.utils.defaultAbiCoder.encode(
      ['uint256', 'uint256', 'bytes32'],
      [subscription.id, subscription.validUntil, subscription.signature]
    );
  }
}
module.exports = PaymasterStrategies;
