// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "./InterestRateModel.sol";
import "./RiskAssessment.sol";
import "./LiquidationEngine.sol";
import "./RewardsDistributor.sol";

/**
 * @title LendingPool
 * @dev Core lending and borrowing functionality with advanced DeFi features
 * @author DeFi Lending Protocol Team
 */
contract LendingPool is ReentrancyGuard, Pausable, Ownable {
    using SafeERC20 for IERC20;

    struct Asset {
        address tokenAddress;
        uint256 totalSupply;
        uint256 totalBorrows;
        uint256 liquidityIndex;
        uint256 borrowIndex;
        uint256 lastUpdateTimestamp;
        uint256 liquidationThreshold; // Basis points (e.g., 8000 = 80%)
        uint256 liquidationBonus; // Basis points (e.g., 500 = 5%)
        uint256 reserveFactor; // Basis points
        bool isActive;
        bool isFrozen;
    }

    struct UserReserveData {
        uint256 scaledBalance;
        uint256 scaledBorrowBalance;
        uint256 stableBorrowRate;
        uint256 lastUpdateTimestamp;
        bool useAsCollateral;
    }

    struct FlashLoanData {
        address asset;
        uint256 amount;
        uint256 premium;
        address receiver;
        bytes params;
    }

    // Constants
    uint256 public constant MAX_STABLE_RATE_BORROW_SIZE_PERCENT = 2500; // 25%
    uint256 public constant FLASHLOAN_PREMIUM_TOTAL = 9; // 0.09%
    uint256 public constant MAX_NUMBER_RESERVES = 128;
    
    // Core contracts
    InterestRateModel public immutable interestRateModel;
    RiskAssessment public immutable riskAssessment;
    LiquidationEngine public immutable liquidationEngine;
    RewardsDistributor public immutable rewardsDistributor;

    // State variables
    mapping(address => Asset) public assets;
    mapping(address => mapping(address => UserReserveData)) public userReserveData;
    mapping(address => address[]) public userCollateralAssets;
    mapping(address => address[]) public userBorrowAssets;
    address[] public assetsList;
    
    // Flash loan state
    uint256 private _flashLoanLocked;
    
    // Credit delegation
    mapping(address => mapping(address => uint256)) public creditDelegation;

    // Events
    event Deposit(
        address indexed user,
        address indexed asset,
        uint256 amount,
        uint256 timestamp
    );
    
    event Withdraw(
        address indexed user,
        address indexed asset,
        uint256 amount,
        uint256 timestamp
    );
    
    event Borrow(
        address indexed user,
        address indexed asset,
        uint256 amount,
        uint256 borrowRate,
        uint256 timestamp
    );
    
    event Repay(
        address indexed user,
        address indexed asset,
        uint256 amount,
        uint256 timestamp
    );
    
    event FlashLoan(
        address indexed receiver,
        address indexed asset,
        uint256 amount,
        uint256 premium,
        uint256 timestamp
    );
    
    event CreditDelegated(
        address indexed delegator,
        address indexed delegatee,
        address indexed asset,
        uint256 amount
    );

    event AssetInitialized(
        address indexed asset,
        uint256 liquidationThreshold,
        uint256 liquidationBonus,
        uint256 reserveFactor
    );

    modifier onlyValidAsset(address asset) {
        require(assets[asset].isActive, "Asset not active");
        require(!assets[asset].isFrozen, "Asset is frozen");
        _;
    }

    modifier onlyFlashLoanReceiver() {
        require(_flashLoanLocked == 1, "Invalid flash loan state");
        _;
    }

    constructor(
        address _interestRateModel,
        address _riskAssessment,
        address _liquidationEngine,
        address _rewardsDistributor
    ) {
        interestRateModel = InterestRateModel(_interestRateModel);
        riskAssessment = RiskAssessment(_riskAssessment);
        liquidationEngine = LiquidationEngine(_liquidationEngine);
        rewardsDistributor = RewardsDistributor(_rewardsDistributor);
    }

    /**
     * @dev Initialize a new lending asset
     */
    function initializeAsset(
        address asset,
        uint256 liquidationThreshold,
        uint256 liquidationBonus,
        uint256 reserveFactor
    ) external onlyOwner {
        require(asset != address(0), "Invalid asset address");
        require(liquidationThreshold <= 10000, "Invalid liquidation threshold");
        require(liquidationBonus <= 1000, "Invalid liquidation bonus");
        require(reserveFactor <= 1000, "Invalid reserve factor");
        require(assetsList.length < MAX_NUMBER_RESERVES, "Too many reserves");

        Asset storage newAsset = assets[asset];
        require(!newAsset.isActive, "Asset already initialized");

        newAsset.tokenAddress = asset;
        newAsset.liquidationThreshold = liquidationThreshold;
        newAsset.liquidationBonus = liquidationBonus;
        newAsset.reserveFactor = reserveFactor;
        newAsset.isActive = true;
        newAsset.liquidityIndex = 1e27; // Ray math
        newAsset.borrowIndex = 1e27;
        newAsset.lastUpdateTimestamp = block.timestamp;

        assetsList.push(asset);

        emit AssetInitialized(asset, liquidationThreshold, liquidationBonus, reserveFactor);
    }

    /**
     * @dev Deposit assets to earn interest
     */
    function deposit(
        address asset,
        uint256 amount,
        bool useAsCollateral
    ) external nonReentrant whenNotPaused onlyValidAsset(asset) {
        require(amount > 0, "Amount must be greater than 0");

        _updateAssetState(asset);
        
        UserReserveData storage userData = userReserveData[msg.sender][asset];
        
        // Transfer tokens from user
        IERC20(asset).safeTransferFrom(msg.sender, address(this), amount);
        
        // Update user balance
        uint256 scaledAmount = amount * 1e27 / assets[asset].liquidityIndex;
        userData.scaledBalance += scaledAmount;
        userData.useAsCollateral = useAsCollateral;
        userData.lastUpdateTimestamp = block.timestamp;
        
        // Update asset state
        assets[asset].totalSupply += amount;
        
        // Add to user's collateral assets if using as collateral
        if (useAsCollateral) {
            _addUserCollateralAsset(msg.sender, asset);
        }
        
        // Distribute rewards
        rewardsDistributor.updateUserRewards(msg.sender, asset, userData.scaledBalance);
        
        emit Deposit(msg.sender, asset, amount, block.timestamp);
    }

    /**
     * @dev Withdraw deposited assets
     */
    function withdraw(
        address asset,
        uint256 amount
    ) external nonReentrant whenNotPaused onlyValidAsset(asset) {
        require(amount > 0, "Amount must be greater than 0");

        _updateAssetState(asset);
        
        UserReserveData storage userData = userReserveData[msg.sender][asset];
        uint256 userBalance = userData.scaledBalance * assets[asset].liquidityIndex / 1e27;
        
        require(userBalance >= amount, "Insufficient balance");
        
        // Check if withdrawal affects health factor
        if (userData.useAsCollateral) {
            require(
                riskAssessment.validateWithdrawal(msg.sender, asset, amount),
                "Withdrawal would exceed safe health factor"
            );
        }
        
        // Update user balance
        uint256 scaledAmount = amount * 1e27 / assets[asset].liquidityIndex;
        userData.scaledBalance -= scaledAmount;
        userData.lastUpdateTimestamp = block.timestamp;
        
        // Update asset state
        assets[asset].totalSupply -= amount;
        
        // Transfer tokens to user
        IERC20(asset).safeTransfer(msg.sender, amount);
        
        // Update rewards
        rewardsDistributor.updateUserRewards(msg.sender, asset, userData.scaledBalance);
        
        emit Withdraw(msg.sender, asset, amount, block.timestamp);
    }

    /**
     * @dev Borrow assets against collateral
     */
    function borrow(
        address asset,
        uint256 amount,
        uint256 interestRateMode // 1 = stable, 2 = variable
    ) external nonReentrant whenNotPaused onlyValidAsset(asset) {
        require(amount > 0, "Amount must be greater than 0");
        require(interestRateMode == 1 || interestRateMode == 2, "Invalid interest rate mode");

        _updateAssetState(asset);
        
        // Check borrowing capacity
        require(
            riskAssessment.validateBorrow(msg.sender, asset, amount),
            "Insufficient collateral for borrow"
        );
        
        UserReserveData storage userData = userReserveData[msg.sender][asset];
        
        uint256 borrowRate;
        if (interestRateMode == 1) {
            // Stable rate borrowing
            borrowRate = interestRateModel.getStableBorrowRate(asset);
            userData.stableBorrowRate = borrowRate;
        } else {
            // Variable rate borrowing
            borrowRate = interestRateModel.getVariableBorrowRate(asset);
        }
        
        // Update user borrow balance
        uint256 scaledAmount = amount * 1e27 / assets[asset].borrowIndex;
        userData.scaledBorrowBalance += scaledAmount;
        userData.lastUpdateTimestamp = block.timestamp;
        
        // Update asset state
        assets[asset].totalBorrows += amount;
        
        // Add to user's borrow assets
        _addUserBorrowAsset(msg.sender, asset);
        
        // Transfer tokens to user
        IERC20(asset).safeTransfer(msg.sender, amount);
        
        emit Borrow(msg.sender, asset, amount, borrowRate, block.timestamp);
    }

    /**
     * @dev Repay borrowed assets
     */
    function repay(
        address asset,
        uint256 amount
    ) external nonReentrant whenNotPaused onlyValidAsset(asset) {
        require(amount > 0, "Amount must be greater than 0");

        _updateAssetState(asset);
        
        UserReserveData storage userData = userReserveData[msg.sender][asset];
        uint256 userBorrowBalance = userData.scaledBorrowBalance * assets[asset].borrowIndex / 1e27;
        
        require(userBorrowBalance > 0, "No debt to repay");
        
        // Use minimum of amount and total debt
        uint256 repayAmount = amount > userBorrowBalance ? userBorrowBalance : amount;
        
        // Transfer tokens from user
        IERC20(asset).safeTransferFrom(msg.sender, address(this), repayAmount);
        
        // Update user borrow balance
        uint256 scaledAmount = repayAmount * 1e27 / assets[asset].borrowIndex;
        userData.scaledBorrowBalance -= scaledAmount;
        userData.lastUpdateTimestamp = block.timestamp;
        
        // Update asset state
        assets[asset].totalBorrows -= repayAmount;
        
        emit Repay(msg.sender, asset, repayAmount, block.timestamp);
    }

    /**
     * @dev Execute flash loan
     */
    function flashLoan(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params
    ) external nonReentrant whenNotPaused onlyValidAsset(asset) {
        require(amount > 0, "Amount must be greater than 0");
        require(amount <= IERC20(asset).balanceOf(address(this)), "Insufficient liquidity");

        uint256 premium = amount * FLASHLOAN_PREMIUM_TOTAL / 10000;
        uint256 balanceBefore = IERC20(asset).balanceOf(address(this));

        // Set flash loan state
        _flashLoanLocked = 1;

        // Transfer tokens to receiver
        IERC20(asset).safeTransfer(receiverAddress, amount);

        // Execute receiver logic
        IFlashLoanReceiver(receiverAddress).executeOperation(
            asset,
            amount,
            premium,
            msg.sender,
            params
        );

        // Reset flash loan state
        _flashLoanLocked = 0;

        // Check repayment
        uint256 balanceAfter = IERC20(asset).balanceOf(address(this));
        require(balanceAfter >= balanceBefore + premium, "Flash loan not repaid");

        emit FlashLoan(receiverAddress, asset, amount, premium, block.timestamp);
    }

    /**
     * @dev Delegate credit to another address
     */
    function delegateCredit(
        address delegatee,
        address asset,
        uint256 amount
    ) external onlyValidAsset(asset) {
        creditDelegation[msg.sender][delegatee] = amount;
        emit CreditDelegated(msg.sender, delegatee, asset, amount);
    }

    /**
     * @dev Borrow on behalf of another user (credit delegation)
     */
    function borrowOnBehalf(
        address user,
        address asset,
        uint256 amount,
        uint256 interestRateMode
    ) external nonReentrant whenNotPaused onlyValidAsset(asset) {
        require(creditDelegation[user][msg.sender] >= amount, "Insufficient credit delegation");
        
        // Reduce delegated credit
        creditDelegation[user][msg.sender] -= amount;
        
        // Execute borrow for the user
        _executeBorrowOnBehalf(user, asset, amount, interestRateMode);
    }

    /**
     * @dev Get user account data
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
        return riskAssessment.getUserAccountData(user);
    }

    /**
     * @dev Get asset data
     */
    function getAssetData(address asset)
        external
        view
        returns (
            uint256 totalSupply,
            uint256 totalBorrows,
            uint256 liquidityRate,
            uint256 variableBorrowRate,
            uint256 stableBorrowRate,
            uint256 utilizationRate
        )
    {
        Asset memory assetData = assets[asset];
        utilizationRate = assetData.totalSupply > 0 
            ? (assetData.totalBorrows * 1e27) / assetData.totalSupply 
            : 0;
        
        liquidityRate = interestRateModel.getLiquidityRate(asset);
        variableBorrowRate = interestRateModel.getVariableBorrowRate(asset);
        stableBorrowRate = interestRateModel.getStableBorrowRate(asset);
        
        return (
            assetData.totalSupply,
            assetData.totalBorrows,
            liquidityRate,
            variableBorrowRate,
            stableBorrowRate,
            utilizationRate
        );
    }

    // Internal functions
    function _updateAssetState(address asset) internal {
        Asset storage assetData = assets[asset];
        uint256 timeDelta = block.timestamp - assetData.lastUpdateTimestamp;
        
        if (timeDelta == 0) return;
        
        // Update indices based on interest rates
        uint256 liquidityRate = interestRateModel.getLiquidityRate(asset);
        uint256 variableBorrowRate = interestRateModel.getVariableBorrowRate(asset);
        
        // Compound interest calculation (simplified)
        uint256 liquidityIndexDelta = (liquidityRate * timeDelta) / (365 days);
        uint256 borrowIndexDelta = (variableBorrowRate * timeDelta) / (365 days);
        
        assetData.liquidityIndex += (assetData.liquidityIndex * liquidityIndexDelta) / 1e27;
        assetData.borrowIndex += (assetData.borrowIndex * borrowIndexDelta) / 1e27;
        assetData.lastUpdateTimestamp = block.timestamp;
    }

    function _addUserCollateralAsset(address user, address asset) internal {
        address[] storage userAssets = userCollateralAssets[user];
        for (uint256 i = 0; i < userAssets.length; i++) {
            if (userAssets[i] == asset) return;
        }
        userAssets.push(asset);
    }

    function _addUserBorrowAsset(address user, address asset) internal {
        address[] storage userAssets = userBorrowAssets[user];
        for (uint256 i = 0; i < userAssets.length; i++) {
            if (userAssets[i] == asset) return;
        }
        userAssets.push(asset);
    }

    function _executeBorrowOnBehalf(
        address user,
        address asset,
        uint256 amount,
        uint256 interestRateMode
    ) internal {
        // Implementation for borrowing on behalf of another user
        // Similar to regular borrow but for a different user
        _updateAssetState(asset);
        
        require(
            riskAssessment.validateBorrow(user, asset, amount),
            "Insufficient collateral for borrow"
        );
        
        UserReserveData storage userData = userReserveData[user][asset];
        
        uint256 borrowRate;
        if (interestRateMode == 1) {
            borrowRate = interestRateModel.getStableBorrowRate(asset);
            userData.stableBorrowRate = borrowRate;
        } else {
            borrowRate = interestRateModel.getVariableBorrowRate(asset);
        }
        
        uint256 scaledAmount = amount * 1e27 / assets[asset].borrowIndex;
        userData.scaledBorrowBalance += scaledAmount;
        userData.lastUpdateTimestamp = block.timestamp;
        
        assets[asset].totalBorrows += amount;
        _addUserBorrowAsset(user, asset);
        
        IERC20(asset).safeTransfer(msg.sender, amount);
        
        emit Borrow(user, asset, amount, borrowRate, block.timestamp);
    }

    // Emergency functions
    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    function freezeAsset(address asset) external onlyOwner {
        assets[asset].isFrozen = true;
    }

    function unfreezeAsset(address asset) external onlyOwner {
        assets[asset].isFrozen = false;
    }
}

interface IFlashLoanReceiver {
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external;
}