"use client";
import React from 'react';
import { Activity, Zap, Cpu, Terminal } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-white p-8 font-sans">
      <header className="flex justify-between items-center mb-10">
        <h1 className="text-3xl font-bold tracking-tight">AINEON <span className="text-blue-500">ENTERPRISE</span></h1>
        <div className="flex gap-4">
          <Badge label="PIMLICO: ONLINE" color="bg-emerald-500/10 text-emerald-500" />
          <Badge label="AI: LEARNING" color="bg-purple-500/10 text-purple-500" />
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <MetricCard title="Est. JIT Yield" value="$4,200" icon={<Zap className="text-yellow-400"/>} />
        <MetricCard title="Solver Surplus" value="$1,105" icon={<Activity className="text-blue-400"/>} />
        <MetricCard title="Gas Saved" value="$850" icon={<Cpu className="text-green-400"/>} />
        <MetricCard title="Active Strategies" value="3" icon={<Terminal className="text-slate-400"/>} />
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 h-96 flex items-center justify-center">
        <p className="text-slate-500">Live Financial Charting Module (Recharts Loaded)</p>
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon }: any) {
  return (
    <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
      <div className="flex justify-between mb-4">
        <span className="text-slate-400 text-sm">{title}</span>
        {icon}
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function Badge({ label, color }: any) {
  return <span className={`px-3 py-1 rounded text-xs font-bold border ${color} border-opacity-20`}>{label}</span>;
}
