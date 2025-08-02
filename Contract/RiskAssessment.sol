// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title RiskAssessment
 * @dev Real-time collateral valuation and health factor calculation
 * @author DeFi Lending Protocol Team
 */
contract RiskAssessment is Ownable {

    struct AssetConfig {
        address priceOracle;           // Chainlink price feed
        uint256 liquidationThreshold;  // Liquidation threshold (basis points)
        uint256 loanToValue;           // Maximum LTV ratio (basis points)
        uint256 liquidationBonus;      // Liquidation bonus (basis points)
        uint256 volatilityFactor;      // Volatility adjustment (Ray)
        uint256 correlationFactor;     // Asset correlation factor (Ray)
        bool isActive;
        bool isCollateral;
    }

    struct UserRiskData {
        uint256 totalCollateralValue;
        uint256 totalBorrowValue;
        uint256 healthFactor;
        uint256 liquidationThreshold;
        uint256 maxLoanToValue;
        uint256 availableBorrowValue;
        address[] collateralAssets;
        address[] borrowAssets;
        uint256 lastUpdateTimestamp;
    }

    struct MarketConditions {
        uint256 volatilityIndex;       // Market volatility (0-10000 basis points)
        uint256 correlationIndex;      // Asset correlation index
        uint256 liquidityStress;       // Liquidity stress factor
        uint256 lastUpdateTimestamp;
    }

    // Constants
    uint256 public constant RAY = 1e27;
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant HEALTH_FACTOR_LIQUIDATION_THRESHOLD = 1e18; // 1.0
    uint256 public constant MIN_HEALTH_FACTOR = 105e16; // 1.05
    uint256 public constant MAX_HEALTH_FACTOR = 115792089237316195423570985008687907853269984665640564039457e18;

    // Core contracts
    address public lendingPool;
    
    // Asset configurations
    mapping(address => AssetConfig) public assetConfigs;
    mapping(address => bool) public supportedAssets;
    address[] public assetsList;

    // Risk data
    mapping(address => UserRiskData) public userRiskData;
    MarketConditions public marketConditions;

    // ML Model parameters (simplified on-chain implementation)
    mapping(bytes32 => uint256) public mlModelWeights;
    uint256 public modelVersion;

    // Price deviation tracking
    mapping(address => uint256) public lastKnownPrices;
    mapping(address => uint256) public priceDeviationThreshold; // basis points

    // Events
    event AssetConfigUpdated(
        address indexed asset,
        address priceOracle,
        uint256 liquidationThreshold,
        uint256 loanToValue
    );

    event UserRiskUpdated(
        address indexed user,
        uint256 healthFactor,
        uint256 totalCollateralValue,
        uint256 totalBorrowValue
    );

    event LiquidationThresholdReached(
        address indexed user,
        uint256 healthFactor,
        uint256 timestamp
    );

    event MarketConditionsUpdated(
        uint256 volatilityIndex,
        uint256 correlationIndex,
        uint256 liquidityStress
    );

    event PriceDeviationAlert(
        address indexed asset,
        uint256 oldPrice,
        uint256 newPrice,
        uint256 deviation
    );

    modifier onlyLendingPool() {
        require(msg.sender == lendingPool, "Only lending pool");
        _;
    }

    modifier onlyValidAsset(address asset) {
        require(supportedAssets[asset], "Asset not supported");
        _;
    }

    constructor() {
        // Initialize market conditions
        marketConditions = MarketConditions({
            volatilityIndex: 1000, // 10% volatility
            correlationIndex: 5000, // 50% correlation
            liquidityStress: 0, // No stress
            lastUpdateTimestamp: block.timestamp
        });
        modelVersion = 1;
    }

    /**
     * @dev Set the lending pool address
     */
    function setLendingPool(address _lendingPool) external onlyOwner {
        require(_lendingPool != address(0), "Invalid lending pool address");
        lendingPool = _lendingPool;
    }

    /**
     * @dev Configure a new asset for risk assessment
     */
    function configureAsset(
        address asset,
        address priceOracle,
        uint256 liquidationThreshold,
        uint256 loanToValue,
        uint256 liquidationBonus,
        bool isCollateral
    ) external onlyOwner {
        require(asset != address(0), "Invalid asset address");
        require(priceOracle != address(0), "Invalid price oracle");
        require(liquidationThreshold <= BASIS_POINTS, "Invalid liquidation threshold");
        require(loanToValue <= liquidationThreshold, "LTV cannot exceed liquidation threshold");
        require(liquidationBonus <= 1000, "Invalid liquidation bonus"); // Max 10%

        AssetConfig storage config = assetConfigs[asset];
        config.priceOracle = priceOracle;
        config.liquidationThreshold = liquidationThreshold;
        config.loanToValue = loanToValue;
        config.liquidationBonus = liquidationBonus;
        config.volatilityFactor = RAY; // Default 1.0x
        config.correlationFactor = RAY; // Default 1.0x
        config.isActive = true;
        config.isCollateral = isCollateral;

        if (!supportedAssets[asset]) {
            supportedAssets[asset] = true;
            assetsList.push(asset);
        }

        // Initialize price tracking
        uint256 currentPrice = _getAssetPrice(asset);
        lastKnownPrices[asset] = currentPrice;
        priceDeviationThreshold[asset] = 500; // 5% default threshold

        emit AssetConfigUpdated(asset, priceOracle, liquidationThreshold, loanToValue);
    }

    /**
     * @dev Validate if a user can make a withdrawal
     */
    function validateWithdrawal(
        address user,
        address asset,
        uint256 amount
    ) external view onlyLendingPool returns (bool) {
        if (!assetConfigs[asset].isCollateral) {
            return true; // Non-collateral assets can be withdrawn freely
        }

        // Calculate health factor after withdrawal
        uint256 projectedHealthFactor = _calculateProjectedHealthFactor(
            user,
            asset,
            amount,
            0, // No additional borrow
            true // Is withdrawal
        );

        return projectedHealthFactor >= MIN_HEALTH_FACTOR;
    }

    /**
     * @dev Validate if a user can make a borrow
     */
    function validateBorrow(
        address user,
        address asset,
        uint256 amount
    ) external view onlyLendingPool returns (bool) {
        // Calculate health factor after borrow
        uint256 projectedHealthFactor = _calculateProjectedHealthFactor(
            user,
            asset,
            0, // No withdrawal
            amount,
            false // Is borrow
        );

        return projectedHealthFactor >= MIN_HEALTH_FACTOR;
    }

    /**
     * @dev Get comprehensive user account data
     */
    function getUserAccountData(address user)
        external
        view
        returns (
            uint256 totalCollateralETH,
            uint256 totalBorrowsETH,
            uint256 availableBorrowsETH,
            uint256 currentLiquidationThreshold,
            uint256 ltv,
            uint256 healthFactor
        )
    {
        (
            totalCollateralETH,
            totalBorrowsETH,
            availableBorrowsETH,
            currentLiquidationThreshold,
            ltv,
            healthFactor
        ) = _calculateUserAccountData(user);
    }

    /**
     * @dev Update user risk data (called by lending pool)
     */
    function updateUserRiskData(address user) external onlyLendingPool {
        (
            uint256 totalCollateralETH,
            uint256 totalBorrowsETH,
            uint256 availableBorrowsETH,
            uint256 currentLiquidationThreshold,
            uint256 ltv,
            uint256 healthFactor
        ) = _calculateUserAccountData(user);

        UserRiskData storage userData = userRiskData[user];
        userData.totalCollateralValue = totalCollateralETH;
        userData.totalBorrowValue = totalBorrowsETH;
        userData.healthFactor = healthFactor;
        userData.liquidationThreshold = currentLiquidationThreshold;
        userData.maxLoanToValue = ltv;
        userData.availableBorrowValue = availableBorrowsETH;
        userData.lastUpdateTimestamp = block.timestamp;

        emit UserRiskUpdated(user, healthFactor, totalCollateralETH, totalBorrowsETH);

        // Check if liquidation threshold is reached
        if (healthFactor <= HEALTH_FACTOR_LIQUIDATION_THRESHOLD) {
            emit LiquidationThresholdReached(user, healthFactor, block.timestamp);
        }
    }

    /**
     * @dev Calculate user account data with ML risk adjustments
     */
    function _calculateUserAccountData(address user)
        internal
        view
        returns (
            uint256 totalCollateralETH,
            uint256 totalBorrowsETH,
            uint256 availableBorrowsETH,
            uint256 currentLiquidationThreshold,
            uint256 ltv,
            uint256 healthFactor
        )
    {
        // This would interact with the lending pool to get user positions
        // Simplified implementation for demonstration
        
        uint256 weightedCollateral;
        uint256 weightedLiquidationThreshold;
        uint256 weightedLTV;

        // Calculate collateral values and weighted parameters
        for (uint256 i = 0; i < assetsList.length; i++) {
            address asset = assetsList[i];
            AssetConfig memory config = assetConfigs[asset];
            
            if (!config.isCollateral || !config.isActive) continue;

            // Get user balance (simplified - would come from lending pool)
            uint256 userBalance = _getUserAssetBalance(user, asset);
            if (userBalance == 0) continue;

            uint256 assetPrice = _getAssetPrice(asset);
            uint256 assetValueETH = (userBalance * assetPrice) / 1e18;

            // Apply volatility and correlation adjustments
            assetValueETH = _applyRiskAdjustments(asset, assetValueETH);

            totalCollateralETH += assetValueETH;
            weightedCollateral += assetValueETH;
            weightedLiquidationThreshold += (assetValueETH * config.liquidationThreshold);
            weightedLTV += (assetValueETH * config.loanToValue);
        }

        // Calculate borrow values
        for (uint256 i = 0; i < assetsList.length; i++) {
            address asset = assetsList[i];
            
            uint256 userBorrowBalance = _getUserBorrowBalance(user, asset);
            if (userBorrowBalance == 0) continue;

            uint256 assetPrice = _getAssetPrice(asset);
            uint256 borrowValueETH = (userBorrowBalance * assetPrice) / 1e18;
            totalBorrowsETH += borrowValueETH;
        }

        // Calculate weighted averages
        if (weightedCollateral > 0) {
            currentLiquidationThreshold = weightedLiquidationThreshold / weightedCollateral;
            ltv = weightedLTV / weightedCollateral;
        }

        // Calculate health factor
        if (totalBorrowsETH > 0) {
            healthFactor = (totalCollateralETH * currentLiquidationThreshold) / (totalBorrowsETH * BASIS_POINTS);
        } else {
            healthFactor = MAX_HEALTH_FACTOR;
        }

        // Calculate available borrows
        if (totalCollateralETH > 0 && ltv > 0) {
            uint256 maxBorrowETH = (totalCollateralETH * ltv) / BASIS_POINTS;
            availableBorrowsETH = maxBorrowETH > totalBorrowsETH ? maxBorrowETH - totalBorrowsETH : 0;
        }
    }

    /**
     * @dev Apply ML-based risk adjustments
     */
    function _applyRiskAdjustments(address asset, uint256 baseValue) 
        internal 
        view 
        returns (uint256) 
    {
        AssetConfig memory config = assetConfigs[asset];
        
        // Apply volatility adjustment
        uint256 volatilityAdjustment = (config.volatilityFactor * marketConditions.volatilityIndex) / (RAY * BASIS_POINTS);
        uint256 adjustedValue = (baseValue * (RAY - volatilityAdjustment)) / RAY;

        // Apply correlation adjustment
        uint256 correlationAdjustment = (config.correlationFactor * marketConditions.correlationIndex) / (RAY * BASIS_POINTS);
        adjustedValue = (adjustedValue * (RAY - correlationAdjustment)) / RAY;

        // Apply liquidity stress test
        if (marketConditions.liquidityStress > 0) {
            uint256 liquidityAdjustment = (marketConditions.liquidityStress * RAY) / BASIS_POINTS;
            adjustedValue = (adjustedValue * (RAY - liquidityAdjustment)) / RAY;
        }

        return adjustedValue;
    }

    /**
     * @dev Calculate projected health factor for validation
     */
    function _calculateProjectedHealthFactor(
        address user,
        address asset,
        uint256 withdrawAmount,
        uint256 borrowAmount,
        bool isWithdrawal
    ) internal view returns (uint256) {
        (
            uint256 totalCollateralETH,
            uint256 totalBorrowsETH,
            ,
            uint256 currentLiquidationThreshold,
            ,
        ) = _calculateUserAccountData(user);

        // Adjust for withdrawal/borrow
        if (isWithdrawal && withdrawAmount > 0) {
            uint256 assetPrice = _getAssetPrice(asset);
            uint256 withdrawValueETH = (withdrawAmount * assetPrice) / 1e18;
            totalCollateralETH = totalCollateralETH > withdrawValueETH ? 
                totalCollateralETH - withdrawValueETH : 0;
        }

        if (borrowAmount > 0) {
            uint256 assetPrice = _getAssetPrice(asset);
            uint256 borrowValueETH = (borrowAmount * assetPrice) / 1e18;
            totalBorrowsETH += borrowValueETH;
        }

        // Calculate new health factor
        if (totalBorrowsETH > 0) {
            return (totalCollateralETH * currentLiquidationThreshold) / (totalBorrowsETH * BASIS_POINTS);
        }
        
        return MAX_HEALTH_FACTOR;
    }

    /**
     * @dev Get asset price from Chainlink oracle
     */
    function _getAssetPrice(address asset) internal view