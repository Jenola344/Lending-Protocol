// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title RewardsDistributor
 * @dev Distribute incentive tokens to lenders and borrowers based on their activity
 * @author DeFi Lending Protocol Team
 */
contract RewardsDistributor is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    struct AssetRewards {
        uint256 supplyRewardRate;      // Rewards per second for suppliers
        uint256 borrowRewardRate;      // Rewards per second for borrowers
        uint256 totalSupplyRewards;    // Total rewards distributed to suppliers
        uint256 totalBorrowRewards;    // Total rewards distributed to borrowers
        uint256 lastUpdateTimestamp;   // Last time rewards were updated
        uint256 supplyIndex;           // Supply reward index
        uint256 borrowIndex;           // Borrow reward index
        bool isActive;                 // Whether rewards are active for this asset
    }

    struct UserRewards {
        uint256 supplyIndex;           // User's supply reward index
        uint256 borrowIndex;           // User's borrow reward index
        uint256 accruedRewards;        // Accrued but unclaimed rewards
        uint256 claimedRewards;        // Total claimed rewards
        uint256 lastClaimTimestamp;    // Last claim timestamp
        mapping(address => uint256) assetSupplyIndex;  // Per-asset supply index
        mapping(address => uint256) assetBorrowIndex;  // Per-asset borrow index
    }

    struct RewardCampaign {
        string name;
        address rewardToken;
        uint256 totalRewards;
        uint256 distributedRewards;
        uint256 startTime;
        uint256 endTime;
        uint256 minParticipationTime;  // Minimum time to be eligible
        bool isActive;
        mapping(address => bool) eligibleAssets;
    }

    struct StakingPool {
        address stakingToken;
        uint256 totalStaked;
        uint256 rewardRate;
        uint256 lastUpdateTime;
        uint256 rewardPerTokenStored;
        bool isActive;
        mapping(address => uint256) userStaked;
        mapping(address => uint256) userRewardPerTokenPaid;
        mapping(address => uint256) userRewards;
    }

    // Constants
    uint256 public constant PRECISION = 1e18;
    uint256 public constant MAX_REWARD_RATE = 1e18; // 100% per second (unrealistic but safe upper bound)

    // Core contracts
    address public lendingPool;
    IERC20 public rewardToken;

    // State variables
    mapping(address => AssetRewards) public assetRewards;
    mapping(address => UserRewards) public userRewards;
    mapping(uint256 => RewardCampaign) public rewardCampaigns;
    mapping(uint256 => StakingPool) public stakingPools;
    
    address[] public rewardAssets;
    uint256 public nextCampaignId;
    uint256 public nextStakingPoolId;
    uint256 public totalRewardsDistributed;
    
    // Boost mechanics
    mapping(address => uint256) public userBoostMultiplier; // Multiplier in basis points
    uint256 public maxBoostMultiplier = 25000; // 2.5x maximum boost
    
    // Governance rewards
    mapping(address => uint256) public governanceParticipation;
    uint256 public governanceRewardRate;
    
    // Flash loan rewards
    mapping(address => uint256) public flashLoanRewards;
    uint256 public flashLoanRewardRate;

    // Events
    event RewardsAccrued(
        address indexed user,
        address indexed asset,
        uint256 supplyRewards,
        uint256 borrowRewards
    );

    event RewardsClaimed(
        address indexed user,
        uint256 amount,
        uint256 timestamp
    );

    event AssetRewardsUpdated(
        address indexed asset,
        uint256 supplyRewardRate,
        uint256 borrowRewardRate
    );

    event CampaignCreated(
        uint256 indexed campaignId,
        string name,
        address rewardToken,
        uint256 totalRewards,
        uint256 startTime,
        uint256 endTime
    );

    event StakingPoolCreated(
        uint256 indexed poolId,
        address stakingToken,
        uint256 rewardRate
    );

    event Staked(
        uint256 indexed poolId,
        address indexed user,
        uint256 amount
    );

    event Unstaked(
        uint256 indexed poolId,
        address indexed user,
        uint256 amount
    );

    event BoostActivated(
        address indexed user,
        uint256 multiplier
    );

    modifier onlyLendingPool() {
        require(msg.sender == lendingPool, "Only lending pool");
        _;
    }

    constructor(address _rewardToken) {
        rewardToken = IERC20(_rewardToken);
        governanceRewardRate = 1e15; // 0.001 tokens per second
        flashLoanRewardRate = 1e14;  // 0.0001 tokens per flash loan
    }

    /**
     * @dev Set the lending pool address
     */
    function setLendingPool(address _lendingPool) external onlyOwner {
        require(_lendingPool != address(0), "Invalid lending pool");
        lendingPool = _lendingPool;
    }

    /**
     * @dev Initialize rewards for an asset
     */
    function initializeAssetRewards(
        address asset,
        uint256 supplyRewardRate,
        uint256 borrowRewardRate
    ) external onlyOwner {
        require(asset != address(0), "Invalid asset");
        require(supplyRewardRate <= MAX_REWARD_RATE, "Supply rate too high");
        require(borrowRewardRate <= MAX_REWARD_RATE, "Borrow rate too high");

        AssetRewards storage rewards = assetRewards[asset];
        if (!rewards.isActive) {
            rewardAssets.push(asset);
        }

        rewards.supplyRewardRate = supplyRewardRate;
        rewards.borrowRewardRate = borrowRewardRate;
        rewards.lastUpdateTimestamp = block.timestamp;
        rewards.supplyIndex = PRECISION;
        rewards.borrowIndex = PRECISION;
        rewards.isActive = true;

        emit AssetRewardsUpdated(asset, supplyRewardRate, borrowRewardRate);
    }

    /**
     * @dev Update user rewards (called by lending pool)
     */
    function updateUserRewards(
        address user,
        address asset,
        uint256 userSupplyBalance
    ) external onlyLendingPool {
        _updateAssetRewards(asset);
        _updateUserAssetRewards(user, asset, userSupplyBalance, 0);
    }

    /**
     * @dev Update user borrow rewards
     */
    function updateUserBorrowRewards(
        address user,
        address asset,
        uint256 userBorrowBalance
    ) external onlyLendingPool {
        _updateAssetRewards(asset);
        _updateUserAssetRewards(user, asset, 0, userBorrowBalance);
    }

    /**
     * @dev Update both supply and borrow rewards
     */
    function updateUserRewardsBoth(
        address user,
        address asset,
        uint256 userSupplyBalance,
        uint256 userBorrowBalance
    ) external onlyLendingPool {
        _updateAssetRewards(asset);
        _updateUserAssetRewards(user, asset, userSupplyBalance, userBorrowBalance);
    }

    /**
     * @dev Claim all accrued rewards
     */
    function claimRewards() external nonReentrant {
        _updateAllUserRewards(msg.sender);
        
        UserRewards storage user = userRewards[msg.sender];
        uint256 rewards = user.accruedRewards;
        require(rewards > 0, "No rewards to claim");

        user.accruedRewards = 0;
        user.claimedRewards += rewards;
        user.lastClaimTimestamp = block.timestamp;

        // Apply boost multiplier
        uint256 boostedRewards = _applyBoost(msg.sender, rewards);
        
        rewardToken.safeTransfer(msg.sender, boostedRewards);
        totalRewardsDistributed += boostedRewards;

        emit RewardsClaimed(msg.sender, boostedRewards, block.timestamp);
    }

    /**
     * @dev Claim rewards for specific assets
     */
    function claimAssetRewards(address[] calldata assets) external nonReentrant {
        uint256 totalRewards = 0;
        
        for (uint256 i = 0; i < assets.length; i++) {
            _updateAssetRewards(assets[i]);
            uint256 assetRewards = _calculateUserAssetRewards(msg.sender, assets[i]);
            totalRewards += assetRewards;
        }

        require(totalRewards > 0, "No rewards to claim");

        UserRewards storage user = userRewards[msg.sender];
        user.accruedRewards += totalRewards;
        
        uint256 boostedRewards = _applyBoost(msg.sender, user.accruedRewards);
        user.accruedRewards = 0;
        user.claimedRewards += boostedRewards;
        user.lastClaimTimestamp = block.timestamp;

        rewardToken.safeTransfer(msg.sender, boostedRewards);
        totalRewardsDistributed += boostedRewards;

        emit RewardsClaimed(msg.sender, boostedRewards, block.timestamp);
    }

    /**
     * @dev Create a new reward campaign
     */
    function createRewardCampaign(
        string calldata name,
        address campaignRewardToken,
        uint256 totalRewards,
        uint256 startTime,
        uint256 endTime,
        uint256 minParticipationTime,
        address[] calldata eligibleAssets
    ) external onlyOwner {
        require(bytes(name).length > 0, "Invalid campaign name");
        require(campaignRewardToken != address(0), "Invalid reward token");
        require(totalRewards > 0, "Invalid total rewards");
        require(endTime > startTime, "Invalid time range");
        require(startTime >= block.timestamp, "Start time in past");

        uint256 campaignId = nextCampaignId++;
        RewardCampaign storage campaign = rewardCampaigns[campaignId];
        
        campaign.name = name;
        campaign.rewardToken = campaignRewardToken;
        campaign.totalRewards = totalRewards;
        campaign.startTime = startTime;
        campaign.endTime = endTime;
        campaign.minParticipationTime = minParticipationTime;
        campaign.isActive = true;

        for (uint256 i = 0; i < eligibleAssets.length; i++) {
            campaign.eligibleAssets[eligibleAssets[i]] = true;
        }

        // Transfer campaign rewards to contract
        IERC20(campaignRewardToken).safeTransferFrom(msg.sender, address(this), totalRewards);

        emit CampaignCreated(campaignId, name, campaignRewardToken, totalRewards, startTime, endTime);
    }

    /**
     * @dev Create staking pool for additional rewards
     */
    function createStakingPool(
        address stakingToken,
        uint256 rewardRate
    ) external onlyOwner {
        require(stakingToken != address(0), "Invalid staking token");
        require(rewardRate > 0, "Invalid reward rate");

        uint256 poolId = nextStakingPoolId++;
        StakingPool storage pool = stakingPools[poolId];
        
        pool.stakingToken = stakingToken;
        pool.rewardRate = rewardRate;
        pool.lastUpdateTime = block.timestamp;
        pool.isActive = true;

        emit StakingPoolCreated(poolId, stakingToken, rewardRate);
    }

    /**
     * @dev Stake tokens in a pool
     */
    function stake(uint256 poolId, uint256 amount) external nonReentrant {
        StakingPool storage pool = stakingPools[poolId];
        require(pool.isActive, "Pool not active");
        require(amount > 0, "Invalid amount");

        _updateStakingPool(poolId);

        // Transfer staking tokens
        IERC20(pool.stakingToken).safeTransferFrom(msg.sender, address(this), amount);

        // Update user staking data
        pool.userStaked[msg.sender] += amount;
        pool.totalStaked += amount;
        pool.userRewardPerTokenPaid[msg.sender] = pool.rewardPerTokenStored;

        emit Staked(poolId, msg.sender, amount);
    }

    /**
     * @dev Unstake tokens from a pool
     */
    function unstake(uint256 poolId, uint256 amount) external nonReentrant {
        StakingPool storage pool = stakingPools[poolId];
        require(pool.userStaked[msg.sender] >= amount, "Insufficient staked amount");

        _updateStakingPool(poolId);

        // Calculate rewards
        uint256 reward = _calculateStakingRewards(poolId, msg.sender);
        if (reward > 0) {
            pool.userRewards[msg.sender] = 0;
            rewardToken.safeTransfer(msg.sender, reward);
        }

        // Update staking data
        pool.userStaked[msg.sender] -= amount;
        pool.totalStaked -= amount;
        pool.userRewardPerTokenPaid[msg.sender] = pool.rewardPerTokenStored;

        // Transfer staking tokens back
        IERC20(pool.stakingToken).safeTransfer(msg.sender, amount);

        emit Unstaked(poolId, msg.sender, amount);
    }

    /**
     * @dev Activate boost for user
     */
    function activateBoost(address user, uint256 multiplier) external onlyOwner {
        require(multiplier <= maxBoostMultiplier, "Multiplier too high");
        userBoostMultiplier[user] = multiplier;
        emit BoostActivated(user, multiplier);
    }

    /**
     * @dev Distribute governance rewards
     */
    function distributeGovernanceRewards(address user, uint256 participationScore) external onlyOwner {
        governanceParticipation[user] += participationScore;
        uint256 rewards = participationScore * governanceRewardRate;
        
        UserRewards storage userData = userRewards[user];
        userData.accruedRewards += rewards;
    }

    /**
     * @dev Distribute flash loan rewards
     */
    function distributeFlashLoanRewards(address user) external onlyLendingPool {
        flashLoanRewards[user] += flashLoanRewardRate;
        
        UserRewards storage userData = userRewards[user];
        userData.accruedRewards += flashLoanRewardRate;
    }

    /**
     * @dev Internal function to update asset rewards
     */
    function _updateAssetRewards(address asset) internal {
        AssetRewards storage rewards = assetRewards[asset];
        if (!rewards.isActive || rewards.lastUpdateTimestamp == block.timestamp) {
            return;
        }

        uint256 timeDelta = block.timestamp - rewards.lastUpdateTimestamp;
        
        // Update supply index
        if (rewards.supplyRewardRate > 0) {
            uint256 supplyRewardAccrued = rewards.supplyRewardRate * timeDelta;
            rewards.supplyIndex += supplyRewardAccrued;
            rewards.totalSupplyRewards += supplyRewardAccrued;
        }

        // Update borrow index
        if (rewards.borrowRewardRate > 0) {
            uint256 borrowRewardAccrued = rewards.borrowRewardRate * timeDelta;
            rewards.borrowIndex += borrowRewardAccrued;
            rewards.totalBorrowRewards += borrowRewardAccrued;
        }

        rewards.lastUpdateTimestamp = block.timestamp;
    }

    /**
     * @dev Update user rewards for a specific asset
     */
    function _updateUserAssetRewards(
        address user,
        address asset,
        uint256 userSupplyBalance,
        uint256 userBorrowBalance
    ) internal {
        UserRewards storage userData = userRewards[user];
        AssetRewards storage assetReward = assetRewards[asset];

        // Calculate supply rewards
        if (userSupplyBalance > 0) {
            uint256 supplyIndexDelta = assetReward.supplyIndex - userData.assetSupplyIndex[asset];
            uint256 supplyRewards = (userSupplyBalance * supplyIndexDelta) / PRECISION;
            userData.accruedRewards += supplyRewards;
        }
        userData.assetSupplyIndex[asset] = assetReward.supplyIndex;

        // Calculate borrow rewards
        if (userBorrowBalance > 0) {
            uint256 borrowIndexDelta = assetReward.borrowIndex - userData.assetBorrowIndex[asset];
            uint256 borrowRewards = (userBorrowBalance * borrowIndexDelta) / PRECISION;
            userData.accruedRewards += borrowRewards;
        }
        userData.assetBorrowIndex[asset] = assetReward.borrowIndex;

        emit RewardsAccrued(user, asset, 0, 0); // Would emit actual calculated rewards
    }

    /**
     * @dev Update all user rewards across all assets
     */
    function _updateAllUserRewards(address user) internal {
        for (uint256 i = 0; i < rewardAssets.length; i++) {
            address asset = rewardAssets[i];
            _updateAssetRewards(asset);
            // Would get actual user balances from lending pool
            _updateUserAssetRewards(user, asset, 0, 0);
        }
    }

    /**
     * @dev Calculate user rewards for a specific asset
     */
    function _calculateUserAssetRewards(address user, address asset) internal view returns (uint256) {
        // Simplified calculation - would use actual user balances
        return 0;
    }

    /**
     * @dev Apply boost multiplier to rewards
     */
    function _applyBoost(address user, uint256 baseRewards) internal view returns (uint256) {
        uint256 multiplier = userBoostMultiplier[user];
        if (multiplier == 0) {
            return baseRewards;
        }
        return (baseRewards * (10000 + multiplier)) / 10000;
    }

    /**
     * @dev Update staking pool rewards
     */
    function _updateStakingPool(uint256 poolId) internal {
        StakingPool storage pool = stakingPools[poolId];
        
        if (pool.totalStaked == 0) {
            pool.lastUpdateTime = block.timestamp;
            return;
        }

        uint256 timeDelta = block.timestamp - pool.lastUpdateTime;
        uint256 reward = timeDelta * pool.rewardRate;
        pool.rewardPerTokenStored += (reward * PRECISION) / pool.totalStaked;
        pool.lastUpdateTime = block.timestamp;
    }

    /**
     * @dev Calculate staking rewards for user
     */
    function _calculateStakingRewards(uint256 poolId, address user) internal view returns (uint256) {
        StakingPool storage pool = stakingPools[poolId];
        
        uint256 rewardPerToken = pool.rewardPerTokenStored;
        if (pool.totalStaked > 0) {
            uint256 timeDelta = block.timestamp - pool.lastUpdateTime;
            uint256 reward = timeDelta * pool.rewardRate;
            rewardPerToken += (reward * PRECISION) / pool.totalStaked;
        }

        return pool.userStaked[user] * (rewardPerToken - pool.userRewardPerTokenPaid[user]) / PRECISION + pool.userRewards[user];
    }

    // View functions
    function getUserRewardData(address user) external view returns (
        uint256 accruedRewards,
        uint256 claimedRewards,
        uint256 boostMultiplier,
        uint256 lastClaimTimestamp
    ) {
        UserRewards storage userData = userRewards[user];
        return (
            userData.accruedRewards,
            userData.claimedRewards,
            userBoostMultiplier[user],
            userData.lastClaimTimestamp
        );
    }

    function getAssetRewardData(address asset) external view returns (AssetRewards memory) {
        return assetRewards[asset];
    }

    function getCampaignData(uint256 campaignId) external view returns (
        string memory name,
        address rewardToken,
        uint256 totalRewards,
        uint256 distributedRewards,
        uint256 startTime,
        uint256 endTime,
        bool isActive
    ) {
        RewardCampaign storage campaign = rewardCampaigns[campaignId];
        return (
            campaign.name,
            campaign.rewardToken,
            campaign.totalRewards,
            campaign.distributedRewards,
            campaign.startTime,
            campaign.endTime,
            campaign.isActive
        );
    }

    function getStakingPoolData(uint256 poolId) external view returns (
        address stakingToken,
        uint256 totalStaked,
        uint256 rewardRate,
        bool isActive
    ) {
        StakingPool storage pool = stakingPools[poolId];
        return (
            pool.stakingToken,
            pool.totalStaked,
            pool.rewardRate,
            pool.isActive
        );
    }

    function getUserStakingData(uint256 poolId, address user) external view returns (
        uint256 stakedAmount,
        uint256 earnedRewards
    ) {
        StakingPool storage pool = stakingPools[poolId];
        return (
            pool.userStaked[user],
            _calculateStakingRewards(poolId, user)
        );
    }

    // Admin functions
    function updateRewardRates(
        address asset,
        uint256 supplyRewardRate,
        uint256 borrowRewardRate
    ) external onlyOwner {
        _updateAssetRewards(asset);
        
        AssetRewards storage rewards = assetRewards[asset];
        rewards.supplyRewardRate = supplyRewardRate;
        rewards.borrowRewardRate = borrowRewardRate;

        emit AssetRewardsUpdated(asset, supplyRewardRate, borrowRewardRate);
    }

    function setMaxBoostMultiplier(uint256 multiplier) external onlyOwner {
        require(multiplier <= 50000, "Multiplier too high"); // Max 5x
        maxBoostMultiplier = multiplier;
    }

    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(owner(), amount);
    }
}