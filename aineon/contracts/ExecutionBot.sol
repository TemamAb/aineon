// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ExecutionBot {
    function executeTrade(uint256 amount) external pure returns (uint256) {
        return (amount * 25) / 10000; // 0.25% profit
    }
}
