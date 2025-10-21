// market-sentiment.js - Market Sentiment Analysis & Regime Detection
// Reverse-engineered from Hugging Face, NLTK, VADER patterns

const tf = require('@tensorflow/tfjs-node');
const { SentimentAnalyzer } = require('natural');
const { VADER } = require('vader-sentiment');

class MarketSentiment {
    constructor(config) {
        this.config = config;
        this.sentimentModels = new Map();
        this.regimeModels = new Map();
        this.dataSources = new Map();
        this.sentimentHistory = [];
    }

    async initialize() {
        console.log('í³Š Initializing Market Sentiment Analysis...');
        
        await this.loadSentimentModels();
        await this.initializeDataSources();
        await this.warmUpAnalyzers();
        
        console.log('âœ… Market Sentiment Analysis initialized');
    }

    async analyzeMarketSentiment(marketData, newsData, socialData) {
        // Comprehensive market sentiment analysis
        const sentimentResults = await Promise.all([
            this.analyzePriceSentiment(marketData),
            this.analyzeNewsSentiment(newsData),
            this.analyzeSocialSentiment(socialData),
            this.analyzeOnChainSentiment(marketData),
            this.analyzeOptionsSentiment(marketData)
        ]);

        const aggregatedSentiment = this.aggregateSentimentScores(sentimentResults);
        const regimeDetection = await this.detectMarketRegime(aggregatedSentiment, marketData);
        
        return this.generateSentimentReport(aggregatedSentiment, regimeDetection);
    }

    async analyzePriceSentiment(marketData) {
        // Price-based sentiment analysis
        const priceSignals = await this.extractPriceSignals(marketData);
        const volumeSignals = await this.analyzeVolumeSentiment(marketData);
        const volatilitySignals = await this.analyzeVolatilitySentiment(marketData);
        
        return {
            type: 'price_sentiment',
            score: this.calculatePriceSentiment(priceSignals, volumeSignals, volatilitySignals),
            confidence: 0.85,
            components: { priceSignals, volumeSignals, volatilitySignals }
        };
    }

    async analyzeNewsSentiment(newsData) {
        // News and media sentiment analysis
        if (!newsData || newsData.length === 0) {
            return { type: 'news_sentiment', score: 0, confidence: 0, articles: 0 };
        }

        const articleSentiments = await Promise.all(
            newsData.map(article => this.analyzeArticleSentiment(article))
        );

        const validSentiments = articleSentiments.filter(s => s !== null);
        const averageScore = validSentiments.reduce((sum, s) => sum + s.score, 0) / validSentiments.length;
        
        return {
            type: 'news_sentiment',
            score: averageScore,
            confidence: Math.min(0.9, validSentiments.length / 10), // More articles = higher confidence
            articles: validSentiments.length,
            topStories: validSentiments.sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, 5)
        };
    }

    async analyzeSocialSentiment(socialData) {
        // Social media sentiment analysis
        if (!socialData || socialData.length === 0) {
            return { type: 'social_sentiment', score: 0, confidence: 0, posts: 0 };
        }

        const postSentiments = await Promise.all(
            socialData.map(post => this.analyzeSocialPost(post))
        );

        const validSentiments = postSentiments.filter(s => s !== null);
        const averageScore = validSentiments.reduce((sum, s) => sum + s.score, 0) / validSentiments.length;
        
        return {
            type: 'social_sentiment',
            score: averageScore,
            confidence: Math.min(0.8, validSentiments.length / 100), // Scale with post count
            posts: validSentiments.length,
            sources: this.analyzeSocialSources(validSentiments)
        };
    }

    async analyzeOnChainSentiment(marketData) {
        // On-chain data sentiment analysis
        const chainSignals = await this.extractOnChainSignals(marketData);
        const whaleActivity = await this.analyzeWhaleActivity(marketData);
        const exchangeFlows = await this.analyzeExchangeFlows(marketData);
        
        return {
            type: 'on_chain_sentiment',
            score: this.calculateOnChainSentiment(chainSignals, whaleActivity, exchangeFlows),
            confidence: 0.75,
            components: { chainSignals, whaleActivity, exchangeFlows }
        };
    }

    async analyzeOptionsSentiment(marketData) {
        // Options market sentiment analysis
        const putCallRatio = await this.analyzePutCallRatio(marketData);
        const volatilitySkew = await this.analyzeVolatilitySkew(marketData);
        const openInterest = await this.analyzeOpenInterest(marketData);
        
        return {
            type: 'options_sentiment',
            score: this.calculateOptionsSentiment(putCallRatio, volatilitySkew, openInterest),
            confidence: 0.7,
            components: { putCallRatio, volatilitySkew, openInterest }
        };
    }

    async analyzeArticleSentiment(article) {
        // Analyze sentiment of a single news article
        try {
            const text = `${article.title} ${article.content}`.toLowerCase();
            
            // Use multiple sentiment analysis methods
            const vaderScore = VADER.SentimentIntensityAnalyzer.polarity_scores(text);
            const naturalScore = this.analyzeWithNatural(text);
            
            // Combine scores
            const combinedScore = (vaderScore.compound + naturalScore) / 2;
            
            return {
                score: combinedScore,
                confidence: Math.abs(combinedScore),
                source: article.source,
                timestamp: article.timestamp
            };
        } catch (error) {
            console.warn('Article sentiment analysis failed:', error.message);
            return null;
        }
    }

    async analyzeSocialPost(post) {
        // Analyze sentiment of a social media post
        try {
            const text = post.content.toLowerCase();
            
            // Clean and preprocess text
            const cleanedText = this.cleanSocialText(text);
            
            // Analyze sentiment
            const sentiment = this.analyzeWithVADER(cleanedText);
            
            // Adjust for social media specific factors
            const adjustedScore = this.adjustSocialScore(sentiment, post);
            
            return {
                score: adjustedScore,
                confidence: this.calculateSocialConfidence(post),
                platform: post.platform,
                engagement: post.engagement
            };
        } catch (error) {
            console.warn('Social post analysis failed:', error.message);
            return null;
        }
    }

    analyzeWithVADER(text) {
        // VADER sentiment analysis
        const analyzer = new VADER.SentimentIntensityAnalyzer();
        const scores = analyzer.polarity_scores(text);
        return scores.compound; // Returns between -1 (negative) and +1 (positive)
    }

    analyzeWithNatural(text) {
        // Natural language processing sentiment analysis
        try {
            const analyzer = new SentimentAnalyzer('English');
            const analysis = analyzer.getSentiment(text.split(' '));
            return Math.tanh(analysis); // Normalize to -1 to 1
        } catch (error) {
            return 0;
        }
    }

    cleanSocialText(text) {
        // Clean social media text for analysis
        return text
            .replace(/[^\w\s]|_/g, '') // Remove punctuation
            .replace(/\s+/g, ' ') // Normalize whitespace
            .replace(/[0-9]/g, '') // Remove numbers
            .trim();
    }

    adjustSocialScore(sentiment, post) {
        // Adjust sentiment score based on social media factors
        let adjusted = sentiment;
        
        // Adjust for engagement (more engagement = stronger signal)
        if (post.engagement) {
            const engagementBoost = Math.min(0.3, post.engagement / 1000);
            adjusted += Math.sign(adjusted) * engagementBoost;
        }
        
        // Adjust for author influence
        if (post.authorInfluence) {
            adjusted *= (1 + post.authorInfluence * 0.5);
        }
        
        return Math.max(-1, Math.min(1, adjusted));
    }

    calculateSocialConfidence(post) {
        // Calculate confidence in social media analysis
        let confidence = 0.5; // Base confidence
        
        // Increase confidence with engagement
        if (post.engagement > 100) confidence += 0.2;
        if (post.engagement > 1000) confidence += 0.1;
        
        // Increase confidence with author credibility
        if (post.authorInfluence > 0.7) confidence += 0.2;
        
        return Math.min(0.9, confidence);
    }

    async detectMarketRegime(sentiment, marketData) {
        // Detect current market regime based on sentiment and market data
        const model = this.regimeModels.get('regime_detector');
        const features = this.extractRegimeFeatures(sentiment, marketData);
        
        const prediction = await model.predict(tf.tensor([features]));
        const result = await prediction.data();
        prediction.dispose();
        
        return this.interpretRegime(result, sentiment);
    }

    extractRegimeFeatures(sentiment, marketData) {
        // Extract features for regime detection
        return [
            sentiment.overallScore,
            marketData.volatility || 0,
            marketData.trendStrength || 0,
            sentiment.priceSentiment.score,
            sentiment.newsSentiment.score,
            sentiment.socialSentiment.score
        ];
    }

    interpretRegime(prediction, sentiment) {
        // Interpret regime prediction results
        const regimes = ['bull', 'bear', 'volatile', 'stable'];
        const maxIndex = prediction.indexOf(Math.max(...prediction));
        
        return {
            regime: regimes[maxIndex],
            confidence: prediction[maxIndex],
            sentimentBias: sentiment.overallScore,
            duration: this.estimateRegimeDuration(prediction)
        };
    }

    aggregateSentimentScores(sentimentResults) {
        // Aggregate sentiment scores from different sources
        const validResults = sentimentResults.filter(result => result.confidence > 0.1);
        
        if (validResults.length === 0) {
            return { overallScore: 0, confidence: 0, components: {} };
        }

        // Weighted average based on confidence
        let totalScore = 0;
        let totalWeight = 0;
        const components = {};

        for (const result of validResults) {
            const weight = result.confidence;
            totalScore += result.score * weight;
            totalWeight += weight;
            components[result.type] = result;
        }

        const overallScore = totalWeight > 0 ? totalScore / totalWeight : 0;
        const overallConfidence = totalWeight / validResults.length;

        return {
            overallScore,
            confidence: overallConfidence,
            components,
            bias: this.determineSentimentBias(overallScore)
        };
    }

    determineSentimentBias(score) {
        // Determine sentiment bias direction
        if (score > 0.1) return 'BULLISH';
        if (score < -0.1) return 'BEARISH';
        return 'NEUTRAL';
    }

    generateSentimentReport(sentiment, regime) {
        // Generate comprehensive sentiment report
        return {
            timestamp: Date.now(),
            overallSentiment: sentiment.overallScore,
            sentimentBias: sentiment.bias,
            confidence: sentiment.confidence,
            marketRegime: regime.regime,
            regimeConfidence: regime.confidence,
            componentAnalysis: sentiment.components,
            tradingImplications: this.deriveTradingImplications(sentiment, regime),
            riskAssessment: this.assessSentimentRisk(sentiment, regime)
        };
    }

    deriveTradingImplications(sentiment, regime) {
        // Derive trading implications from sentiment and regime
        const implications = [];
        
        if (sentiment.bias === 'BULLISH' && regime.regime === 'bull') {
            implications.push('Strong bullish momentum - consider long positions');
        } else if (sentiment.bias === 'BEARISH' && regime.regime === 'bear') {
            implications.push('Strong bearish momentum - consider short positions');
        } else if (regime.regime === 'volatile') {
            implications.push('High volatility - use smaller position sizes');
        }
        
        // Add more specific implications based on analysis
        
        return implications;
    }

    assessSentimentRisk(sentiment, regime) {
        // Assess risk based on sentiment analysis
        let riskLevel = 'LOW';
        let reasons = [];
        
        if (Math.abs(sentiment.overallScore) > 0.7) {
            riskLevel = 'HIGH';
            reasons.push('Extreme sentiment detected');
        }
        
        if (regime.regime === 'volatile' && sentiment.confidence < 0.5) {
            riskLevel = 'MEDIUM';
            reasons.push('High volatility with low sentiment confidence');
        }
        
        return { riskLevel, reasons };
    }

    async loadSentimentModels() {
        // Load sentiment analysis models
        const modelNames = [
            'news_sentiment',
            'social_sentiment',
            'regime_detector',
            'price_sentiment'
        ];

        for (const modelName of modelNames) {
            this.sentimentModels.set(modelName, await this.loadModel(modelName));
        }
    }

    async initializeDataSources() {
        // Initialize data sources for sentiment analysis
        this.dataSources.set('news', new NewsDataSource());
        this.dataSources.set('social', new SocialDataSource());
        this.dataSources.set('on_chain', new OnChainDataSource());
    }

    async warmUpAnalyzers() {
        // Warm up sentiment analyzers
        console.log('í´¥ Warming up sentiment analyzers...');
        
        const sampleData = this.generateSampleData();
        await this.analyzeMarketSentiment(sampleData, [], []);
        
        console.log('âœ… Sentiment analyzers warmed up');
    }

    generateSampleData() {
        // Generate sample market data for warming up
        return {
            volatility: 0.2,
            trendStrength: 0.1,
            price: 100,
            volume: 1000000
        };
    }

    // Mock implementations for analysis methods
    async extractPriceSignals(marketData) {
        return { momentum: 0.1, trend: 0.05, support: 95, resistance: 105 };
    }

    async analyzeVolumeSentiment(marketData) {
        return { score: 0.2, confidence: 0.7 };
    }

    async analyzeVolatilitySentiment(marketData) {
        return { score: -0.1, confidence: 0.8 };
    }

    calculatePriceSentiment(priceSignals, volumeSignals, volatilitySignals) {
        return (priceSignals.momentum * 0.4 + volumeSignals.score * 0.3 + volatilitySignals.score * 0.3);
    }

    async extractOnChainSignals(marketData) {
        return { activeAddresses: 0.6, transactionVolume: 0.7, networkGrowth: 0.5 };
    }

    async analyzeWhaleActivity(marketData) {
        return { score: 0.3, confidence: 0.6 };
    }

    async analyzeExchangeFlows(marketData) {
        return { score: -0.2, confidence: 0.7 };
    }

    calculateOnChainSentiment(chainSignals, whaleActivity, exchangeFlows) {
        return (chainSignals.activeAddresses * 0.3 + 
                chainSignals.transactionVolume * 0.3 +
                whaleActivity.score * 0.2 +
                exchangeFlows.score * 0.2);
    }

    async analyzePutCallRatio(marketData) {
        return { ratio: 0.8, score: -0.2 };
    }

    async analyzeVolatilitySkew(marketData) {
        return { skew: 0.1, score: 0.1 };
    }

    async analyzeOpenInterest(marketData) {
        return { interest: 0.6, score: 0.3 };
    }

    calculateOptionsSentiment(putCallRatio, volatilitySkew, openInterest) {
        return (putCallRatio.score * 0.4 + volatilitySkew.score * 0.3 + openInterest.score * 0.3);
    }

    estimateRegimeDuration(prediction) {
        // Estimate how long the regime might last
        const confidence = Math.max(...prediction);
        if (confidence > 0.8) return 'LONG_TERM';
        if (confidence > 0.6) return 'MEDIUM_TERM';
        return 'SHORT_TERM';
    }

    analyzeSocialSources(sentiments) {
        // Analyze sentiment by social media source
        const sources = {};
        sentiments.forEach(sentiment => {
            if (!sources[sentiment.platform]) {
                sources[sentiment.platform] = { total: 0, count: 0 };
            }
            sources[sentiment.platform].total += sentiment.score;
            sources[sentiment.platform].count += 1;
        });
        
        // Calculate averages
        Object.keys(sources).forEach(platform => {
            sources[platform].average = sources[platform].total / sources[platform].count;
        });
        
        return sources;
    }

    async loadModel(modelName) {
        // Mock model loading
        return {
            predict: async (tensor) => {
                const shape = tensor.shape;
                const values = new Array(shape[1] || 4).fill(0).map(() => Math.random());
                return tf.tensor([values]);
            }
        };
    }
}

// Mock data source classes
class NewsDataSource {
    async getLatestNews() {
        return [];
    }
}

class SocialDataSource {
    async getSocialPosts() {
        return [];
    }
}

class OnChainDataSource {
    async getOnChainData() {
        return {};
    }
}

module.exports = MarketSentiment;
