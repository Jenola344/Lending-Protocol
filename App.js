const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const mongoose = require('mongoose');
const redis = require('redis');
const Web3 = require('web3');
const { Server } = require('socket.io');
const http = require('http');
require('dotenv').config();

// Import routes
const marketRoutes = require('./routes/markets');
const userRoutes = require('./routes/users');
const liquidationRoutes = require('./routes/liquidations');
const analyticsRoutes = require('./routes/analytics');
const riskRoutes = require('./routes/risk');
const rewardsRoutes = require('./routes/rewards');

// Import services
const PriceService = require('./services/PriceService');
const RiskService = require('./services/RiskService');
const LiquidationService = require('./services/LiquidationService');
const MLService = require('./services/MLService');
const NotificationService = require('./services/NotificationService');

// Import middleware
const authMiddleware = require('./middleware/auth');
const validationMiddleware = require('./middleware/validation');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Initialize services
const priceService = new PriceService();
const riskService = new RiskService();
const liquidationService = new LiquidationService();
const mlService = new MLService();
const notificationService = new NotificationService(io);

// Redis client
const redisClient = redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

// Web3 setup
const web3 = new Web3(process.env.WEB3_PROVIDER_URL);

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(limiter);

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: process.env.npm_package_version || '1.0.0'
  });
});

// API Routes
app.use('/api/markets', marketRoutes);
app.use('/api/users', userRoutes);
app.use('/api/liquidations', liquidationRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api/risk', riskRoutes);
app.use('/api/rewards', rewardsRoutes);

// WebSocket handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  // Subscribe to price updates
  socket.on('subscribe_prices', (assets) => {
    socket.join('price_updates');
    // Send current prices
    assets.forEach(async (asset) => {
      const price = await priceService.getPrice(asset);
      socket.emit('price_update', { asset, price });
    });
  });

  // Subscribe to user updates  
  socket.on('subscribe_user', (userAddress) => {
    socket.join(`user_${userAddress}`);
    // Send current user data
    riskService.getUserRiskData(userAddress).then(data => {
      socket.emit('user_update', data);
    });
  });

  // Subscribe to liquidation alerts
  socket.on('subscribe_liquidations', () => {
    socket.join('liquidation_alerts');
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong!'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: 'The requested resource was not found'
  });
});

// Database connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/lending-protocol', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => {
  console.log('Connected to MongoDB');
})
.catch((err) => {
  console.error('MongoDB connection error:', err);
  process.exit(1);
});

// Redis connection
redisClient.connect()
.then(() => {
  console.log('Connected to Redis');
})
.catch((err) => {
  console.error('Redis connection error:', err);
});

// Initialize services
async function initializeServices() {
  try {
    await priceService.initialize();
    await riskService.initialize();
    await liquidationService.initialize();
    await mlService.initialize();
    
    console.log('All services initialized successfully');
    
    // Start background tasks
    startBackgroundTasks();
  } catch (error) {
    console.error('Failed to initialize services:', error);
    process.exit(1);
  }
}

// Background tasks
function startBackgroundTasks() {
  // Price updates every 30 seconds
  setInterval(async () => {
    try {
      const updatedPrices = await priceService.updateAllPrices();
      io.to('price_updates').emit('price_updates', updatedPrices);
    } catch (error) {
      console.error('Price update error:', error);
    }
  }, 30000);

  // Risk monitoring every 60 seconds
  setInterval(async () => {
    try {
      const riskAlerts = await riskService.monitorAllUsers();
      riskAlerts.forEach(alert => {
        io.to(`user_${alert.user}`).emit('risk_alert', alert);
        if (alert.severity === 'critical') {
          io.to('liquidation_alerts').emit('liquidation_alert', alert);
        }
      });
    } catch (error) {
      console.error('Risk monitoring error:', error);
    }
  }, 60000);

  // Liquidation monitoring every 30 seconds
  setInterval(async () => {
    try {
      const liquidationOpportunities = await liquidationService.findLiquidationOpportunities();
      if (liquidationOpportunities.length > 0) {
        io.to('liquidation_alerts').emit('liquidation_opportunities', liquidationOpportunities);
      }
    } catch (error) {
      console.error('Liquidation monitoring error:', error);
    }
  }, 30000);

  // ML model updates every hour
  setInterval(async () => {
    try {
      await mlService.updateModels();
      console.log('ML models updated');
    } catch (error) {
      console.error('ML model update error:', error);
    }
  }, 3600000);

  // Analytics aggregation every 5 minutes
  setInterval(async () => {
    try {
      await aggregateAnalytics();
    } catch (error) {
      console.error('Analytics aggregation error:', error);
    }
  }, 300000);
}

// Analytics aggregation
async function aggregateAnalytics() {
  const analytics = {
    totalValueLocked: await calculateTVL(),
    totalBorrows: await calculateTotalBorrows(),
    utilizationRates: await calculateUtilizationRates(),
    topYields: await calculateTopYields(),
    riskMetrics: await calculateRiskMetrics(),
    timestamp: new Date().toISOString()
  };

  // Cache analytics
  await redisClient.setEx('analytics:latest', 300, JSON.stringify(analytics));
  
  // Broadcast to connected clients
  io.emit('analytics_update', analytics);
}

// Helper functions for analytics
async function calculateTVL() {
  // Implementation would calculate total value locked across all markets
  return 1000000; // Placeholder
}

async function calculateTotalBorrows() {
  // Implementation would calculate total borrows across all markets
  return 500000; // Placeholder
}

async function calculateUtilizationRates() {
  // Implementation would calculate utilization rates for each market
  return {
    ETH: 0.75,
    USDC: 0.85,
    DAI: 0.70
  };
}

async function calculateTopYields() {
  // Implementation would calculate top yielding opportunities
  return [
    { asset: 'USDC', apy: 0.08, type: 'supply' },
    { asset: 'ETH', apy: 0.12, type: 'supply' },
    { asset: 'DAI', apy: 0.15, type: 'borrow' }
  ];
}

async function calculateRiskMetrics() {
  // Implementation would calculate system-wide risk metrics
  return {
    averageHealthFactor: 2.5,
    liquidationsAt24h: 12,
    systemUtilization: 0.78,
    volatilityIndex: 0.25
  };
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');
  
  // Close database connections
  await mongoose.connection.close();
  await redisClient.quit();
  
  // Close server
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  console.log('SIGINT received, shutting down gracefully');
  
  // Close database connections
  await mongoose.connection.close();
  await redisClient.quit();
  
  // Close server
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

// Start server
const PORT = process.env.PORT || 3001;
server.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  await initializeServices();
});

module.exports = { app, io, redisClient, web3 };
