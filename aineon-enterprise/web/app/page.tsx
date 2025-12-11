"use client";
import React, { useEffect, useState } from 'react';
import { Activity, Zap, Cpu, Terminal, RefreshCw, Server, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';

// Types for our API data
interface EngineStatus {
    status: string;
    chain_id: number;
    ai_active: boolean;
    tier: string;
}

interface ProfitStats {
    total_pnl: number;
    active_trades: number;
    gas_saved: number;
}

interface Opportunity {
    pair: string;
    dex: string;
    profit: number;
    confidence: number;
}

export default function Home() {
    const [status, setStatus] = useState<EngineStatus | null>(null);
    const [profit, setProfit] = useState<ProfitStats | null>(null);
    const [opps, setOpps] = useState<Opportunity[]>([]);
    const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
    const [loading, setLoading] = useState(true);

    const fetchData = async () => {
        try {
            // Parallel fetch for efficiency
            const [statusRes, profitRes, oppsRes] = await Promise.all([
                fetch('/api/status'),
                fetch('/api/profit'),
                fetch('/api/opportunities')
            ]);

            if (statusRes.ok) setStatus(await statusRes.json());
            if (profitRes.ok) setProfit(await profitRes.json());
            if (oppsRes.ok) {
                const data = await oppsRes.json();
                setOpps(data.opportunities || []);
            }
            setLastUpdated(new Date());
        } catch (e) {
            console.error("Failed to fetch engine data", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 3000); // 3-second polling
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="min-h-screen bg-slate-950 text-white p-6 md:p-12 font-sans selection:bg-blue-500/30">
            {/* HEADER */}
            <header className="flex flex-col md:flex-row justify-between items-center mb-12 gap-4">
                <div>
                    <h1 className="text-4xl md:text-5xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-emerald-400">
                        AINEON <span className="text-white text-opacity-20 font-light">ENTERPRISE</span>
                    </h1>
                    <p className="text-slate-500 mt-2 text-sm font-mono flex items-center gap-2">
                        TIER 0.001% EXECUTION LAYER <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    </p>
                </div>

                <div className="flex flex-wrap gap-4 items-center">
                    <Badge
                        label={`CHAIN ID: ${status?.chain_id || '---'}`}
                        icon={<Server size={14} />}
                        color={status?.chain_id ? "blue" : "red"}
                    />
                    <Badge
                        label={status?.ai_active ? "AI NEURAL NET: ACTIVE" : "AI: INITIALIZING"}
                        icon={<Cpu size={14} />}
                        color={status?.ai_active ? "purple" : "yellow"}
                    />
                    <div className="text-xs text-slate-600 font-mono">
                        UPDATED: {lastUpdated.toLocaleTimeString()}
                    </div>
                </div>
            </header>

            {/* KPI GRID */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <MetricCard
                    title="Est. JIT Yield"
                    value={profit ? `$${profit.total_pnl.toLocaleString()}` : "---"}
                    subValue="+12.5% (24h)"
                    icon={<Zap className="text-yellow-400" size={24} />}
                    delay={0.1}
                />
                <MetricCard
                    title="Active Trades"
                    value={profit ? profit.active_trades.toString() : "---"}
                    subValue="High Frequency"
                    icon={<Activity className="text-blue-400" size={24} />}
                    delay={0.2}
                />
                <MetricCard
                    title="Gas Optimised"
                    value={profit ? `$${profit.gas_saved.toLocaleString()}` : "---"}
                    subValue="Pimlico Paymaster"
                    icon={<ShieldCheck className="text-emerald-400" size={24} />}
                    delay={0.3}
                />
                <MetricCard
                    title="System Latency"
                    value="12ms"
                    subValue="MEV Relay"
                    icon={<Terminal className="text-slate-400" size={24} />}
                    delay={0.4}
                />
            </div>

            {/* MAIN CONTENT SPLIT */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* LEFT: Live Feed */}
                <div className="lg:col-span-2">
                    <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <Activity size={20} className="text-blue-500" /> LIVE ARBITRAGE WATCH
                    </h2>
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden backdrop-blur-sm">
                        <div className="grid grid-cols-4 p-4 text-xs text-slate-500 font-mono uppercase tracking-wider border-b border-slate-800">
                            <div>Pair / DEX</div>
                            <div className="text-right">Est. Profit</div>
                            <div className="text-right">Confidence</div>
                            <div className="text-right">Action</div>
                        </div>

                        <div className="divide-y divide-slate-800/50">
                            {opps.length === 0 ? (
                                <div className="p-8 text-center text-slate-600 font-mono">
                                    SCANNING MEMPOOL...
                                </div>
                            ) : (
                                opps.map((op, i) => (
                                    <div key={i} className="grid grid-cols-4 p-4 items-center hover:bg-slate-800/30 transition-colors">
                                        <div>
                                            <div className="font-bold text-white">{op.pair}</div>
                                            <div className="text-xs text-slate-400">{op.dex}</div>
                                        </div>
                                        <div className="text-right font-mono text-emerald-400">
                                            +${op.profit.toFixed(2)}
                                        </div>
                                        <div className="text-right">
                                            <span className={`text-xs px-2 py-1 rounded-full ${op.confidence > 0.8 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-yellow-500/10 text-yellow-400'}`}>
                                                {(op.confidence * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                        <div className="text-right">
                                            <button className="text-xs bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded transition-colors">
                                                EXECUTE
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* RIGHT: System Visualizer */}
                <div className="lg:col-span-1">
                    <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <Cpu size={20} className="text-purple-500" /> SYSTEM STATUS
                    </h2>
                    <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 h-64 flex flex-col items-center justify-center text-center backdrop-blur-sm relative overflow-hidden group">
                        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 group-hover:opacity-100 transition-opacity opacity-50"></div>
                        <RefreshCw className={`w-12 h-12 text-slate-600 mb-4 ${loading ? 'animate-spin' : ''}`} />
                        <p className="text-slate-400 font-mono text-sm">
                            {status ? "ENGINE CONNECTED" : "CONNECTING TO CORE..."}
                        </p>
                        <div className="mt-4 w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 w-2/3 animate-pulse"></div>
                        </div>
                    </div>

                    <div className="mt-6 bg-gradient-to-r from-slate-900 to-slate-800 border border-slate-700 p-6 rounded-xl">
                        <h3 className="text-sm font-bold text-slate-300 mb-2">QUICK ACTIONS</h3>
                        <div className="grid grid-cols-2 gap-3">
                            <button className="bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/30 p-2 rounded text-xs font-bold transition-all">
                                START SCAN
                            </button>
                            <button className="bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 p-2 rounded text-xs font-bold transition-all">
                                HALT ENGINE
                            </button>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}

// Sub-components
function MetricCard({ title, value, subValue, icon, delay }: any) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay }}
            className="bg-slate-900/60 border border-slate-800 p-6 rounded-xl relative overflow-hidden group hover:border-slate-700 transition-all cursor-crosshair"
        >
            <div className="absolute top-0 right-0 p-4 opacity-50 group-hover:opacity-100 transition-opacity group-hover:scale-110 duration-300">
                {icon}
            </div>
            <div className="relative z-10">
                <span className="text-slate-400 text-xs uppercase tracking-widest font-bold">{title}</span>
                <div className="text-3xl font-black mt-2 tracking-tight text-white">{value}</div>
                <div className="text-xs text-slate-500 mt-1 font-mono">{subValue}</div>
            </div>
        </motion.div>
    );
}

function Badge({ label, color, icon }: any) {
    const colors: any = {
        blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
        purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
        emerald: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
        red: "bg-red-500/10 text-red-400 border-red-500/20",
        yellow: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
    }
    return (
        <span className={`pl-2 pr-3 py-1 rounded text-[10px] font-bold border uppercase flex items-center gap-2 backdrop-blur-md ${colors[color] || colors.blue}`}>
            {icon} {label}
        </span>
    );
}
