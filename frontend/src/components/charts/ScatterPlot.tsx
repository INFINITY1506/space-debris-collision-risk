import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { DetailedThreatItem } from '../../api/client';

interface Props {
    threats: DetailedThreatItem[];
}

const COLORS: Record<string, string> = { HIGH: '#EF4444', MEDIUM: '#F59E0B', LOW: '#22C55E' };

export function ScatterPlot({ threats }: Props) {
    const data = threats.map(t => ({
        missDistance: t.miss_distance_km,
        velocity: t.relative_velocity_km_s,
        probability: t.collision_probability,
        name: t.debris_name,
        risk: t.risk_level,
        size: Math.max(30, Math.min(200, t.collision_probability * 20)),
    }));

    return (
        <div className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                Miss Distance vs Relative Velocity
            </div>
            <ResponsiveContainer width="100%" height={240}>
                <ScatterChart margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--brd)" />
                    <XAxis
                        dataKey="missDistance"
                        name="Miss Distance"
                        unit=" km"
                        tick={{ fontSize: 10, fill: 'var(--t4)' }}
                        stroke="var(--brd)"
                        label={{ value: 'Miss Distance (km)', position: 'insideBottom', offset: -2, style: { fontSize: 10, fill: 'var(--t4)' } }}
                    />
                    <YAxis
                        dataKey="velocity"
                        name="Velocity"
                        unit=" km/s"
                        tick={{ fontSize: 10, fill: 'var(--t4)' }}
                        stroke="var(--brd)"
                        width={55}
                        label={{ value: 'Velocity (km/s)', angle: -90, position: 'insideLeft', offset: 10, style: { fontSize: 10, fill: 'var(--t4)' } }}
                    />
                    <Tooltip
                        contentStyle={{ background: 'var(--bg-s)', border: '1px solid var(--brd)', borderRadius: 6, fontSize: '0.7rem' }}
                        formatter={(v: any, name: any) => [typeof v === 'number' ? v.toFixed(3) : v, name]}
                        labelFormatter={() => ''}
                    />
                    <Scatter data={data}>
                        {data.map((entry, i) => (
                            <Cell key={i} fill={COLORS[entry.risk] ?? '#22C55E'} fillOpacity={0.7} />
                        ))}
                    </Scatter>
                </ScatterChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 8 }}>
                {['HIGH', 'MEDIUM', 'LOW'].map(r => (
                    <div key={r} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: COLORS[r] }} />
                        <span style={{ fontSize: '.62rem', color: 'var(--t4)' }}>{r}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
