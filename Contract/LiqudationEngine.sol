// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "./RiskAssessment.sol";

/**
 * @title LiquidationEngine
 * @dev Automated liquidation system with auction mechanisms and protection features
 * @author DeFi Lending Protocol Team
 */
contract LiquidationEngine is ReentrancyGuard, Ownable {
    using SafeERC20 for IERC20;

    struct LiquidationData {
        address user;
        address collateralAsset;
        address debtAsset;
        uint256 collateralAmount;
        uint256 debtAmount;
        uint256 liquidationBonus;
        uint256 timestamp;
        bool isActive;
    }

    struct AuctionData {
        uint256 liquidationId;
        address liquidator;
        uint256 startPrice;
        uint256 currentPrice;
        uint256 startTime;
        uint256 endTime;
        uint256 minBidIncrement;
        bool isActive;
        bool isCompleted;
    }

    struct LiquidatorRewards {
        uint256 totalLiquidations;
        uint256 totalRewards;
        uint256 averageGasUsed;
        uint256 successRate;
        bool isWhitelisted;
    }

    struct ProtectionSettings {
        uint256 maxLiquidationPercent;  // Max % of debt that can be liquidated
        uint256 auctionDuration;        // Duration of Dutch auction
        uint256 priceDecayRate;         // Rate at which auction price decays
        uint256 minHealthFactorGap;     // Minimum gap below liquidation threshold
        bool isProtectionEnabled;
    }

    // Constants
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MAX_LIQUIDATION_PERCENT = 5000; // 50%
    uint256 public constant MIN_AUCTION_DURATION = 300; // 5 minutes
    uint256 public constant MAX_AUCTION_DURATION = 3600; // 1 hour

    // Core contracts
    address public lendingPool;
    RiskAssessment public riskAssessment;

    // State variables
    mapping(uint256 => LiquidationData) public liquidations;
    mapping(uint256 => AuctionData) public auctions;
    mapping(address => LiquidatorRewards) public liquidatorRewards;
    mapping(address => bool) public authorizedLiquidators;
    
    uint256 public nextLiquidationId;
    uint256 public totalLiquidations;
    uint256 public totalProtectedUsers;
    
    ProtectionSettings public protectionSettings;
    
    // Gas optimization
    uint256 public gasOptimizationLevel;
    mapping(address => uint256) public liquidatorGasRefunds;

    // Emergency settings
    bool public emergencyMode;
    uint256 public emergencyLiquidationDiscount; // Additional discount in emergency

    // Events
    event LiquidationExecuted(
        uint256 indexed liquidationId,
        address indexed user,
        address indexed liquidator,
        address collateralAsset,
        address debtAsset,
        uint256 collateralAmount,
        uint256 debtAmount,
        uint256 bonus
    );

    event AuctionStarted(
        uint256 indexed liquidationId,
        uint256 indexed auctionId,
        address indexed user,
        uint256 startPrice,
        uint256 endTime
    );

    event AuctionCompleted(
        uint256 indexed auctionId,
        address indexed winner,
        uint256 finalPrice,
        uint256 timestamp
    );

    event LiquidationProtectionTriggered(
        address indexed user,
        uint256 indexed liquidationId,
        uint256 protectionType
    );

    event LiquidatorWhitelisted(
        address indexed liquidator,
        bool isWhitelisted
    );

    event EmergencyModeToggled(
        bool isActive,
        uint256 timestamp
    );

    modifier onlyLendingPool() {
        require(msg.sender == lendingPool, "Only lending pool");
        _;
    }

    modifier onlyAuthorizedLiquidator() {
        require(authorizedLiquidators[msg.sender] || msg.sender == owner(), "Not authorized liquidator");
        _;
    }

    modifier notInEmergencyMode() {
        require(!emergencyMode, "Emergency mode active");
        _;
    }

    constructor(address _riskAssessment) {
        riskAssessment = RiskAssessment(_riskAssessment);
        
        // Initialize protection settings
        protectionSettings = ProtectionSettings({
            maxLiquidationPercent: 5000, // 50%
            auctionDuration: 1800, // 30 minutes
            priceDecayRate: 100, // 1% per minute
            minHealthFactorGap: 50e15, // 0.05 (5%)
            isProtectionEnabled: true
        });

        gasOptimizationLevel = 1;
        emergencyLiquidationDiscount = 500; // 5% additional discount
    }

    /**
     * @dev Set the lending pool address
     */
    function setLendingPool(address _lendingPool) external onlyOwner {
        require(_lendingPool != address(0), "Invalid lending pool");
        lendingPool = _lendingPool;
    }

    /**
     * @dev Execute liquidation for an unhealthy position
     */
    function liquidate(
        address user,
        address collateralAsset,
        address debtAsset,
        uint256 debtToCover,
        bool useAuction
    ) external nonReentrant onlyAuthorizedLiquidator notInEmergencyMode {
        // Validate liquidation conditions
        require(_isLiquidatable(user), "User not liquidatable");
        require(debtToCover > 0, "Invalid debt amount");

        // Get liquidation data from risk assessment
        (
            bool canBeLiquidated,
            uint256 maxLiquidationAmount,
            address[] memory collateralAssets,
            uint256[] memory collateralValues
        ) = riskAssessment.getLiquidationData(user);

        require(canBeLiquidated, "Cannot liquidate user");
        require(debtToCover <= maxLiquidationAmount, "Debt amount too high");

        // Calculate liquidation amounts
        (
            uint256 collateralAmount,
            uint256 liquidationBonus
        ) = _calculateLiquidationAmounts(user, collateralAsset, debtAsset, debtToCover);

        uint256 liquidationId = nextLiquidationId++;
        
        // Store liquidation data
        liquidations[liquidationId] = LiquidationData({
            user: user,
            collateralAsset: collateralAsset,
            debtAsset: debtAsset,
            collateralAmount: collateralAmount,
            debtAmount: debtToCover,
            liquidationBonus: liquidationBonus,
            timestamp: block.timestamp,
            isActive: true
        });

        if (useAuction && protectionSettings.isProtectionEnabled) {
            _startAuction(liquidationId, collateralAmount);
        } else {
            _executeLiquidation(liquidationId, msg.sender);
        }

        totalLiquidations++;
        _updateLiquidatorRewards(msg.sender, true);
    }

    /**
     * @dev Start Dutch auction for liquidation
     */
    function _startAuction(uint256 liquidationId, uint256 collateralAmount) internal {
        LiquidationData memory liquidationData = liquidations[liquidationId];
        
        // Calculate starting price (market price + bonus)
        uint256 collateralPrice = _getAssetPrice(liquidationData.collateralAsset);
        uint256 startPrice = (collateralAmount * collateralPrice * (BASIS_POINTS + liquidationData.liquidationBonus)) / (BASIS_POINTS * 1e18);

        uint256 auctionId = liquidationId; // Use same ID for simplicity
        
        auctions[auctionId] = AuctionData({
            liquidationId: liquidationId,
            liquidator: address(0),
            startPrice: startPrice,
            currentPrice: startPrice,
            startTime: block.timestamp,
            endTime: block.timestamp + protectionSettings.auctionDuration,
            minBidIncrement: startPrice / 100, // 1% minimum increment
            isActive: true,
            isCompleted: false
        });

        emit AuctionStarted(liquidationId, auctionId, liquidationData.user, startPrice, auctions[auctionId].endTime);
    }

    /**
     * @dev Bid on auction
     */
    function bidOnAuction(uint256 auctionId, uint256 bidAmount) 
        external 
        nonReentrant 
        onlyAuthorizedLiquidator 
    {
        AuctionData storage auction = auctions[auctionId];
        require(auction.isActive, "Auction not active");
        require(block.timestamp <= auction.endTime, "Auction ended");
        require(bidAmount >= auction.currentPrice + auction.minBidIncrement, "Bid too low");

        LiquidationData memory liquidationData = liquidations[auction.liquidationId];
        
        // Transfer bid amount from bidder
        IERC20(liquidationData.debtAsset).safeTransferFrom(msg.sender, address(this), bidAmount);

        // Refund previous bidder if any
        if (auction.liquidator != address(0)) {
            IERC20(liquidationData.debtAsset).safeTransfer(auction.liquidator, auction.currentPrice);
        }

        auction.liquidator = msg.sender;
        auction.currentPrice = bidAmount;

        // If bid is high enough, complete auction immediately
        if (bidAmount >= auction.startPrice) {
            _completeAuction(auctionId);
        }
    }

    /**
     * @dev Complete auction (can be called by anyone after end time)
     */
    function completeAuction(uint256 auctionId) external nonReentrant {
        AuctionData storage auction = auctions[auctionId];
        require(auction.isActive, "Auction not active");
        require(block.timestamp > auction.endTime, "Auction not ended");

        _completeAuction(auctionId);
    }

    /**
     * @dev Internal function to complete auction
     */
    function _completeAuction(uint256 auctionId) internal {
        AuctionData storage auction = auctions[auctionId];
        auction.isActive = false;
        auction.isCompleted = true;

        if (auction.liquidator != address(0)) {
            _executeLiquidation(auction.liquidationId, auction.liquidator);
            emit AuctionCompleted(auctionId, auction.liquidator, auction.currentPrice, block.timestamp);
        } else {
            // No bidders, liquidation protection triggered
            _triggerLiquidationProtection(auction.liquidationId);
        }
    }

    /**
     * @dev Execute the actual liquidation
     */
    function _executeLiquidation(uint256 liquidationId, address liquidator) internal {
        LiquidationData storage liquidationData = liquidations[liquidationId];
        require(liquidationData.isActive, "Liquidation not active");

        // Transfer collateral to liquidator
        IERC20(liquidationData.collateralAsset).safeTransfer(
            liquidator,
            liquidationData.collateralAmount
        );

        // Transfer debt payment to lending pool
        uint256 debtPayment = liquidationData.debtAmount;
        if (auctions[liquidationId].liquidator == liquidator) {
            // Use auction payment
            debtPayment = auctions[liquidationId].currentPrice;
        } else {
            // Direct liquidation - liquidator pays debt
            IERC20(liquidationData.debtAsset).safeTransferFrom(
                liquidator,
                address(this),
                debtPayment
            );
        }

        // Forward payment to lending pool
        IERC20(liquidationData.debtAsset).safeTransfer(lendingPool, debtPayment);

        liquidationData.isActive = false;

        emit LiquidationExecuted(
            liquidationId,
            liquidationData.user,
            liquidator,
            liquidationData.collateralAsset,
            liquidationData.debtAsset,
            liquidationData.collateralAmount,
            liquidationData.debtAmount,
            liquidationData.liquidationBonus
        );

        // Gas refund for liquidator
        if (liquidatorGasRefunds[liquidator] > 0) {
            _refundGas(liquidator);
        }
    }

    /**
     * @dev Trigger liquidation protection mechanisms
     */
    function _triggerLiquidationProtection(uint256 liquidationId) internal {
        // Implement protection mechanisms:
        // 1. Debt restructuring
        // 2. Grace period extension
        // 3. Partial liquidation delay
        
        totalProtectedUsers++;
        
        emit LiquidationProtectionTriggered(
            liquidations[liquidationId].user,
            liquidationId,
            1 // Protection type: Auction failed
        );
    }

    /**
     * @dev Emergency liquidation with additional discount
     */
    function emergencyLiquidate(
        address user,
        address collateralAsset,
        address debtAsset,
        uint256 debtToCover
    ) external nonReentrant onlyOwner {
        require(emergencyMode, "Not in emergency mode");
        require(_isLiquidatable(user), "User not liquidatable");

        (
            uint256 collateralAmount,
            uint256 liquidationBonus
        ) = _calculateLiquidationAmounts(user, collateralAsset, debtAsset, debtToCover);

        // Apply emergency discount
        liquidationBonus += emergencyLiquidationDiscount;

        uint256 liquidationId = nextLiquidationId++;
        
        liquidations[liquidationId] = LiquidationData({
            user: user,
            collateralAsset: collateralAsset,
            debtAsset: debtAsset,
            collateralAmount: collateralAmount,
            debtAmount: debtToCover,
            liquidationBonus: liquidationBonus,
            timestamp: block.timestamp,
            isActive: true
        });

        _executeLiquidation(liquidationId, msg.sender);
    }

    /**
     * @dev Check if user can be liquidated
     */
    function _isLiquidatable(address user) internal view returns (bool) {
        (bool atRisk, uint256 healthFactor) = riskAssessment.isUserAtRisk(user);
        return atRisk && healthFactor <= 1e18; // Health factor <= 1.0
    }

    /**
     * @dev Calculate liquidation amounts
     */
    function _calculateLiquidationAmounts(
        address user,
        address collateralAsset,
        address debtAsset,
        uint256 debtToCover
    ) internal view returns (uint256 collateralAmount, uint256 liquidationBonus) {
        // Get asset prices
        uint256 collateralPrice = _getAssetPrice(collateralAsset);
        uint256 debtPrice = _getAssetPrice(debtAsset);

        // Get liquidation bonus from risk assessment
        liquidationBonus = riskAssessment.getAssetConfig(collateralAsset).liquidationBonus;

        // Calculate collateral amount with bonus
        collateralAmount = (debtToCover * debtPrice * (BASIS_POINTS + liquidationBonus)) / (collateralPrice * BASIS_POINTS);

        // Apply emergency discount if in emergency mode
        if (emergencyMode) {
            liquidationBonus += emergencyLiquidationDiscount;
            collateralAmount = (collateralAmount * (BASIS_POINTS + emergencyLiquidationDiscount)) / BASIS_POINTS;
        }
    }

    /**
     * @dev Get asset price (simplified - would use price oracle)
     */
    function _getAssetPrice(address asset) internal view returns (uint256) {
        // This would integrate with price oracles
        // Simplified implementation for demonstration
        return 1e18; // $1 placeholder
    }

    /**
     * @dev Update liquidator rewards
     */
    function _updateLiquidatorRewards(address liquidator, bool successful) internal {
        LiquidatorRewards storage rewards = liquidatorRewards[liquidator];
        rewards.totalLiquidations++;
        if (successful) {
            rewards.totalRewards += 1; // Simplified reward calculation
            rewards.successRate = (rewards.successRate * (rewards.totalLiquidations - 1) + BASIS_POINTS) / rewards.totalLiquidations;
        } else {
            rewards.successRate = (rewards.successRate * (rewards.totalLiquidations - 1)) / rewards.totalLiquidations;
        }
    }

    /**
     * @dev Refund gas to liquidator
     */
    function _refundGas(address liquidator) internal {
        uint256 gasRefund = liquidatorGasRefunds[liquidator];
        if (gasRefund > 0 && address(this).balance >= gasRefund) {
            liquidatorGasRefunds[liquidator] = 0;
            payable(liquidator).transfer(gasRefund);
        }
    }

    /**
     * @dev Whitelist liquidator
     */
    function whitelistLiquidator(address liquidator, bool isWhitelisted) external onlyOwner {
        authorizedLiquidators[liquidator] = isWhitelisted;
        liquidatorRewards[liquidator].isWhitelisted = isWhitelisted;
        
        emit LiquidatorWhitelisted(liquidator, isWhitelisted);
    }

    /**
     * @dev Update protection settings
     */
    function updateProtectionSettings(
        uint256 maxLiquidationPercent,
        uint256 auctionDuration,
        uint256 priceDecayRate,
        uint256 minHealthFactorGap,
        bool isProtectionEnabled
    ) external onlyOwner {
        require(maxLiquidationPercent <= MAX_LIQUIDATION_PERCENT, "Invalid max liquidation percent");
        require(auctionDuration >= MIN_AUCTION_DURATION && auctionDuration <= MAX_AUCTION_DURATION, "Invalid auction duration");

        protectionSettings.maxLiquidationPercent = maxLiquidationPercent;
        protectionSettings.auctionDuration = auctionDuration;
        protectionSettings.priceDecayRate = priceDecayRate;
        protectionSettings.minHealthFactorGap = minHealthFactorGap;
        protectionSettings.isProtectionEnabled = isProtectionEnabled;
    }

    /**
     * @dev Toggle emergency mode
     */
    function toggleEmergencyMode() external onlyOwner {
        emergencyMode = !emergencyMode;
        emit EmergencyModeToggled(emergencyMode, block.timestamp);
    }

    /**
     * @dev Set gas refund for liquidator
     */
    function setGasRefund(address liquidator, uint256 amount) external onlyOwner {
        liquidatorGasRefunds[liquidator] = amount;
    }

    /**
     * @dev Get liquidation info
     */
    function getLiquidationInfo(uint256 liquidationId) 
        external 
        view 
        returns (LiquidationData memory) 
    {
        return liquidations[liquidationId];
    }

    /**
     * @dev Get auction info
     */
    function getAuctionInfo(uint256 auctionId) 
        external 
        view 
        returns (AuctionData memory) 
    {
        return auctions[auctionId];
    }

    /**
     * @dev Get liquidator stats
     */
    function getLiquidatorStats(address liquidator) 
        external 
        view 
        returns (LiquidatorRewards memory) 
    {
        return liquidatorRewards[liquidator];
    }

    /**
     * @dev Check if liquidation is available for user
     */
    function canLiquidate(address user) external view returns (bool, uint256, string memory) {
        if (!_isLiquidatable(user)) {
            return (false, 0, "User not liquidatable");
        }

        (
            bool canBeLiquidated,
            uint256 maxLiquidationAmount,
            ,
        ) = riskAssessment.getLiquidationData(user);

        if (!canBeLiquidated) {
            return (false, 0, "Risk assessment prevents liquidation");
        }

        return (true, maxLiquidationAmount, "Liquidation available");
    }

    /**
     * @dev Get current auction price (Dutch auction)
     */
    function getCurrentAuctionPrice(uint256 auctionId) external view returns (uint256) {
        AuctionData memory auction = auctions[auctionId];
        if (!auction.isActive || block.timestamp > auction.endTime) {
            return 0;
        }

        // Calculate price decay
        uint256 timeElapsed = block.timestamp - auction.startTime;
        uint256 priceDecay = (auction.startPrice * protectionSettings.priceDecayRate * timeElapsed) / (protectionSettings.auctionDuration * BASIS_POINTS);
        
        if (priceDecay >= auction.startPrice) {
            return auction.startPrice / 10; // Minimum 10% of start price
        }
        
        return auction.startPrice - priceDecay;
    }

    /**
     * @dev Batch liquidate multiple users
     */
    function batchLiquidate(
        address[] calldata users,
        address[] calldata collateralAssets,
        address[] calldata debtAssets,
        uint256[] calldata debtAmounts
    ) external nonReentrant onlyAuthorizedLiquidator {
        require(users.length == collateralAssets.length, "Array length mismatch");
        require(users.length == debtAssets.length, "Array length mismatch");
        require(users.length == debtAmounts.length, "Array length mismatch");
        require(users.length <= 10, "Too many liquidations"); // Prevent gas issues

        for (uint256 i = 0; i < users.length; i++) {
            if (_isLiquidatable(users[i])) {
                liquidate(users[i], collateralAssets[i], debtAssets[i], debtAmounts[i], false);
            }
        }
    }

    /**
     * @dev Get liquidation opportunities
     */
    function getLiquidationOpportunities(uint256 maxResults) 
        external 
        view 
        returns (
            address[] memory users,
            uint256[] memory healthFactors,
            uint256[] memory maxLiquidationAmounts
        ) 
    {
        // In a real implementation, this would scan user positions
        // Simplified for demonstration
        users = new address[](maxResults);
        healthFactors = new uint256[](maxResults);
        maxLiquidationAmounts = new uint256[](maxResults);
        
        // Would populate with actual liquidatable users
        return (users, healthFactors, maxLiquidationAmounts);
    }

    /**
     * @dev Simulate liquidation to estimate gas and rewards
     */
    function simulateLiquidation(
        address user,
        address collateralAsset,
        address debtAsset,
        uint256 debtToCover
    ) external view returns (
        bool canLiquidate,
        uint256 collateralAmount,
        uint256 liquidationBonus,
        uint256 estimatedGas,
        uint256 profitability
    ) {
        if (!_isLiquidatable(user)) {
            return (false, 0, 0, 0, 0);
        }

        (collateralAmount, liquidationBonus) = _calculateLiquidationAmounts(
            user, collateralAsset, debtAsset, debtToCover
        );

        canLiquidate = true;
        estimatedGas = 300000; // Estimated gas cost
        
        // Calculate profitability (simplified)
        uint256 collateralValue = collateralAmount * _getAssetPrice(collateralAsset) / 1e18;
        uint256 debtValue = debtToCover * _getAssetPrice(debtAsset) / 1e18;
        profitability = collateralValue > debtValue ? collateralValue - debtValue : 0;
    }

    /**
     * @dev Update emergency liquidation discount
     */
    function updateEmergencyDiscount(uint256 discount) external onlyOwner {
        require(discount <= 2000, "Discount too high"); // Max 20%
        emergencyLiquidationDiscount = discount;
    }

    /**
     * @dev Withdraw stuck tokens (emergency)
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(owner(), amount);
    }

    /**
     * @dev Receive ETH for gas refunds
     */
    receive() external payable {}

    /**
     * @dev Get contract statistics
     */
    function getContractStats() external view returns (
        uint256 totalLiquidationsCount,
        uint256 totalProtectedUsersCount,
        uint256 activeAuctions,
        uint256 averageAuctionDuration,
        bool isEmergencyMode
    ) {
        totalLiquidationsCount = totalLiquidations;
        totalProtectedUsersCount = totalProtectedUsers;
        isEmergencyMode = emergencyMode;
        
        // Count active auctions
        activeAuctions = 0;
        for (uint256 i = 0; i < nextLiquidationId; i++) {
            if (auctions[i].isActive) {
                activeAuctions++;
            }
        }
        
        // Calculate average auction duration (simplified)
        averageAuctionDuration = protectionSettings.auctionDuration;
    }
}