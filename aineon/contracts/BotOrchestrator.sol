// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract BotOrchestrator {
    address public scoutBot;
    address public executionBot;
    address public riskBot;
    
    event BotsOrchestrated(address scout, address executor, address risk);
    
    function orchestrateBots(address _scout, address _executor, address _risk) external {
        scoutBot = _scout;
        executionBot = _executor;
        riskBot = _risk;
        emit BotsOrchestrated(_scout, _executor, _risk);
    }
    
    function getBotStatus() external view returns (bool, bool, bool) {
        return (
            scoutBot != address(0),
            executionBot != address(0),
            riskBot != address(0)
        );
    }
}
