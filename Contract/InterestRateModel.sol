// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title InterestRateModel
 * @dev Dynamic interest rate calculation based on utilization and market conditions
 * @author DeFi Lending Protocol Team
 */
contract InterestRateModel is Ownable {
    
    struct AssetParams {
        uint256 baseVariableBorrowRate;    // Base variable borrow rate (Ray)
        uint256 variableRateSlope1;        // First slope of variable rate curve (Ray)
        uint256 variableRateSlope2;        // Second slope of variable rate curve (Ray)
        uint256 stableRateSlope1;          // First slope of stable rate curve (Ray)
        uint256 stableRateSlope2;          // Second slope of stable rate curve (Ray)
        uint256 baseStableRateOffset;      // Offset for stable rate (Ray)
        uint256 optimalUtilizationRate;    // Optimal utilization rate (Ray)
        uint256 maxExcessUtilizationRate;  // Max excess utilization rate (Ray)
        uint256 reserveFactor;             // Reserve factor (percentage in Ray)
        bool isActive;
    }

    // Ray math constants (10^27)
    uint256 public constant RAY = 1e27;
    uint256 public constant HALF_RAY = RAY / 2;
    
    // Maximum rates to prevent overflow
    uint256 public constant MAX_BORROW_RATE = 1000 * RAY / 100; // 1000%
    uint256 public constant MAX_UTILIZATION_RATE = RAY; // 100%
    
    // Default parameters
    uint256 public constant DEFAULT_OPTIMAL_UTILIZATION_RATE = 80 * RAY / 100; // 80%
    uint256 public constant DEFAULT_BASE_VARIABLE_BORROW_RATE = 2 * RAY / 100; // 2%
    uint256 public constant DEFAULT_VARIABLE_RATE_SLOPE1 = 4 * RAY / 100; // 4%
    uint256 public constant DEFAULT_VARIABLE_RATE_SLOPE2 = 75 * RAY / 100; // 75%
    uint256 public constant DEFAULT_STABLE_RATE_SLOPE1 = 5 * RAY / 100; // 5%
    uint256 public constant DEFAULT_STABLE_RATE_SLOPE2 = 75 * RAY / 100; // 75%
    uint256 public constant DEFAULT_BASE_STABLE_RATE_OFFSET = 2 * RAY / 100; // 2%

    // Asset-specific parameters
    mapping(address => AssetParams) public assetParams;
    mapping(address => bool) public supportedAssets;
    
    // Market conditions tracking
    mapping(address => uint256) public lastUpdateTimestamp;
    mapping(address => uint256) public avgUtilizationRate; // Moving average
    mapping(address => uint256) public volatilityFactor; // Market volatility adjustment
    
    // Events
    event AssetParamsUpdated(
        address indexed asset,
        uint256 baseVariableBorrowRate,
        uint256 variableRateSlope1,
        uint256 variableRateSlope2,
        uint256 optimalUtilizationRate
    );
    
    event RatesUpdated(
        address indexed asset,
        uint256 liquidityRate,
        uint256 variableBorrowRate,
        uint256 stableBorrowRate,
        uint256 utilizationRate
    );

    event VolatilityFactorUpdated(
        address indexed asset,
        uint256 oldFactor,
        uint256 newFactor
    );

    constructor() {}

    /**
     * @dev Initialize parameters for a new asset
     */
    function initializeAsset(
        address asset,
        uint256 baseVariableBorrowRate,
        uint256 variableRateSlope1,
        uint256 variableRateSlope2,
        uint256 stableRateSlope1,
        uint256 stableRateSlope2,
        uint256 baseStableRateOffset,
        uint256 optimalUtilizationRate,
        uint256 reserveFactor
    ) external onlyOwner {
        require(asset != address(0), "Invalid asset address");
        require(!supportedAssets[asset], "Asset already initialized");
        require(optimalUtilizationRate <= RAY, "Invalid optimal utilization rate");
        require(reserveFactor <= RAY, "Invalid reserve factor");

        AssetParams storage params = assetParams[asset];
        params.baseVariableBorrowRate = baseVariableBorrowRate;
        params.variableRateSlope1 = variableRateSlope1;
        params.variableRateSlope2 = variableRateSlope2;
        params.stableRateSlope1 = stableRateSlope1;
        params.stableRateSlope2 = stableRateSlope2;
        params.baseStableRateOffset = baseStableRateOffset;
        params.optimalUtilizationRate = optimalUtilizationRate;
        params.maxExcessUtilizationRate = RAY - optimalUtilizationRate;
        params.reserveFactor = reserveFactor;
        params.isActive = true;

        supportedAssets[asset] = true;
        lastUpdateTimestamp[asset] = block.timestamp;
        avgUtilizationRate[asset] = 0;
        volatilityFactor[asset] = RAY; // 1.0x (no adjustment)

        emit AssetParamsUpdated(
            asset,
            baseVariableBorrowRate,
            variableRateSlope1,
            variableRateSlope2,
            optimalUtilizationRate
        );
    }

    /**
     * @dev Initialize asset with default parameters
     */
    function initializeAssetWithDefaults(address asset) external onlyOwner {
        initializeAsset(
            asset,
            DEFAULT_BASE_VARIABLE_BORROW_RATE,
            DEFAULT_VARIABLE_RATE_SLOPE1,
            DEFAULT_VARIABLE_RATE_SLOPE2,
            DEFAULT_STABLE_RATE_SLOPE1,
            DEFAULT_STABLE_RATE_SLOPE2,
            DEFAULT_BASE_STABLE_RATE_OFFSET,
            DEFAULT_OPTIMAL_UTILIZATION_RATE,
            10 * RAY / 100 // 10% reserve factor
        );
    }

    /**
     * @dev Calculate current variable borrow rate
     */
    function getVariableBorrowRate(address asset) external view returns (uint256) {
        return _calculateVariableBorrowRate(asset, _getCurrentUtilizationRate(asset));
    }

    /**
     * @dev Calculate current stable borrow rate
     */
    function getStableBorrowRate(address asset) external view returns (uint256) {
        return _calculateStableBorrowRate(asset, _getCurrentUtilizationRate(asset));
    }

    /**
     * @dev Calculate current liquidity rate
     */
    function getLiquidityRate(address asset) external view returns (uint256) {
        uint256 utilizationRate = _getCurrentUtilizationRate(asset);
        uint256 variableBorrowRate = _calculateVariableBorrowRate(asset, utilizationRate);
        
        return _calculateLiquidityRate(asset, variableBorrowRate, utilizationRate);
    }

    /**
     * @dev Get comprehensive rate data for an asset
     */
    function getRateData(address asset)
        external
        view
        returns (
            uint256 liquidityRate,
            uint256 variableBorrowRate,
            uint256 stableBorrowRate,
            uint256 utilizationRate,
            uint256 optimalUtilizationRate
        )
    {
        require(supportedAssets[asset], "Asset not supported");
        
        utilizationRate = _getCurrentUtilizationRate(asset);
        variableBorrowRate = _calculateVariableBorrowRate(asset, utilizationRate);
        stableBorrowRate = _calculateStableBorrowRate(asset, utilizationRate);
        liquidityRate = _calculateLiquidityRate(asset, variableBorrowRate, utilizationRate);
        optimalUtilizationRate = assetParams[asset].optimalUtilizationRate;
    }

    /**
     * @dev Update volatility factor based on market conditions
     */
    function updateVolatilityFactor(address asset, uint256 newVolatilityFactor) external onlyOwner {
        require(supportedAssets[asset], "Asset not supported");
        require(newVolatilityFactor >= RAY / 2, "Volatility factor too low"); // Min 0.5x
        require(newVolatilityFactor <= 3 * RAY, "Volatility factor too high"); // Max 3.0x

        uint256 oldFactor = volatilityFactor[asset];
        volatilityFactor[asset] = newVolatilityFactor;

        emit VolatilityFactorUpdated(asset, oldFactor, newVolatilityFactor);
    }

    /**
     * @dev Update moving average utilization rate
     */
    function updateUtilizationMetrics(address asset) external {
        require(supportedAssets[asset], "Asset not supported");
        
        uint256 currentUtilization = _getCurrentUtilizationRate(asset);
        uint256 timeDelta = block.timestamp - lastUpdateTimestamp[asset];
        
        // Update moving average (exponential moving average with 1-hour decay)
        if (timeDelta > 0) {
            uint256 alpha = timeDelta * RAY / 3600; // 1 hour decay factor
            if (alpha > RAY) alpha = RAY;
            
            avgUtilizationRate[asset] = 
                (avgUtilizationRate[asset] * (RAY - alpha) + currentUtilization * alpha) / RAY;
            
            lastUpdateTimestamp[asset] = block.timestamp;
        }
    }

    /**
     * @dev Calculate variable borrow rate based on utilization
     */
    function _calculateVariableBorrowRate(address asset, uint256 utilizationRate) 
        internal 
        view 
        returns (uint256) 
    {
        AssetParams memory params = assetParams[asset];
        require(params.isActive, "Asset not active");

        uint256 rate;
        
        if (utilizationRate <= params.optimalUtilizationRate) {
            // Below optimal utilization: linear increase
            rate = params.baseVariableBorrowRate + 
                   (utilizationRate * params.variableRateSlope1 / params.optimalUtilizationRate);
        } else {
            // Above optimal utilization: steep increase
            uint256 excessUtilization = utilizationRate - params.optimalUtilizationRate;
            rate = params.baseVariableBorrowRate + 
                   params.variableRateSlope1 + 
                   (excessUtilization * params.variableRateSlope2 / params.maxExcessUtilizationRate);
        }

        // Apply volatility adjustment
        rate = (rate * volatilityFactor[asset]) / RAY;

        // Cap at maximum rate
        if (rate > MAX_BORROW_RATE) {
            rate = MAX_BORROW_RATE;
        }

        return rate;
    }

    /**
     * @dev Calculate stable borrow rate
     */
    function _calculateStableBorrowRate(address asset, uint256 utilizationRate) 
        internal 
        view 
        returns (uint256) 
    {
        AssetParams memory params = assetParams[asset];
        require(params.isActive, "Asset not active");

        uint256 variableRate = _calculateVariableBorrowRate(asset, utilizationRate);
        uint256 stableRateAddition;

        if (utilizationRate <= params.optimalUtilizationRate) {
            stableRateAddition = (utilizationRate * params.stableRateSlope1 / params.optimalUtilizationRate);
        } else {
            uint256 excessUtilization = utilizationRate - params.optimalUtilizationRate;
            stableRateAddition = params.stableRateSlope1 + 
                                (excessUtilization * params.stableRateSlope2 / params.maxExcessUtilizationRate);
        }

        uint256 rate = variableRate + params.baseStableRateOffset + stableRateAddition;

        // Cap at maximum rate
        if (rate > MAX_BORROW_RATE) {
            rate = MAX_BORROW_RATE;
        }

        return rate;
    }

    /**
     * @dev Calculate liquidity rate for lenders
     */
    function _calculateLiquidityRate(
        address asset,
        uint256 variableBorrowRate,
        uint256 utilizationRate
    ) internal view returns (uint256) {
        AssetParams memory params = assetParams[asset];
        
        // Liquidity rate = (Variable Borrow Rate * Utilization Rate * (1 - Reserve Factor))
        uint256 rate = (variableBorrowRate * utilizationRate * (RAY - params.reserveFactor)) / (RAY * RAY);
        
        return rate;
    }

    /**
     * @dev Get current utilization rate from lending pool
     */
    function _getCurrentUtilizationRate(address asset) internal view returns (uint256) {
        // This would interact with the LendingPool contract to get current utilization
        // For now, we'll use a simplified calculation
        
        try IERC20(asset).balanceOf(msg.sender) returns (uint256 totalLiquidity) {
            if (totalLiquidity == 0) return 0;
            
            // Simplified: assume some borrowed amount (in real implementation, 
            // this would come from LendingPool contract)
            uint256 totalBorrows = totalLiquidity / 4; // Assume 25% utilization
            
            return (totalBorrows * RAY) / (totalLiquidity + totalBorrows);
        } catch {
            return 0;
        }
    }

    /**
     * @dev Update asset parameters
     */
    function updateAssetParams(
        address asset,
        uint256 baseVariableBorrowRate,
        uint256 variableRateSlope1,
        uint256 variableRateSlope2,
        uint256 optimalUtilizationRate
    ) external onlyOwner {
        require(supportedAssets[asset], "Asset not supported");
        require(optimalUtilizationRate <= RAY, "Invalid optimal utilization rate");

        AssetParams storage params = assetParams[asset];
        params.baseVariableBorrowRate = baseVariableBorrowRate;
        params.variableRateSlope1 = variableRateSlope1;
        params.variableRateSlope2 = variableRateSlope2;
        params.optimalUtilizationRate = optimalUtilizationRate;
        params.maxExcessUtilizationRate = RAY - optimalUtilizationRate;

        emit AssetParamsUpdated(
            asset,
            baseVariableBorrowRate,
            variableRateSlope1,
            variableRateSlope2,
            optimalUtilizationRate
        );
    }

    /**
     * @dev Emergency function to deactivate an asset
     */
    function deactivateAsset(address asset) external onlyOwner {
        require(supportedAssets[asset], "Asset not supported");
        assetParams[asset].isActive = false;
    }

    /**
     * @dev Reactivate a deactivated asset
     */
    function reactivateAsset(address asset) external onlyOwner {
        require(supportedAssets[asset], "Asset not supported");
        assetParams[asset].isActive = true;
    }

    /**
     * @dev Get asset parameters
     */
    function getAssetParams(address asset) 
        external 
        view 
        returns (AssetParams memory) 
    {
        require(supportedAssets[asset], "Asset not supported");
        return assetParams[asset];
    }

    /**
     * @dev Check if asset is supported
     */
    function isAssetSupported(address asset) external view returns (bool) {
        return supportedAssets[asset];
    }

    /**
     * @dev Get optimal rates for given utilization
     */
    function getOptimalRates(address asset, uint256 targetUtilization)
        external
        view
        returns (
            uint256 variableBorrowRate,
            uint256 stableBorrowRate,
            uint256 liquidityRate
        )
    {
        require(supportedAssets[asset], "Asset not supported");
        require(targetUtilization <= RAY, "Invalid utilization rate");

        variableBorrowRate = _calculateVariableBorrowRate(asset, targetUtilization);
        stableBorrowRate = _calculateStableBorrowRate(asset, targetUtilization);
        liquidityRate = _calculateLiquidityRate(asset, variableBorrowRate, targetUtilization);
    }
}