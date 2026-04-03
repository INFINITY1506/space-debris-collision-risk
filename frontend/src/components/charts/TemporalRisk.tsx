import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { TemporalRiskPoint } from '../../api/client';

interface Props {
    data: TemporalRiskPoint[];
}

export function TemporalRisk({ data }: Props) {
    const chartData = data.map(d => ({
        hour: d.hour,
        probability: d.collision_probability,
        missDistance: d.miss_distance_km,
        label: `${d.hour}h`,
    }));

    const maxProb = Math.max(...chartData.map(d => d.probability), 1e-12);

    return (
        <div className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                Temporal Risk Profile (168h)
            </div>
            <ResponsiveContainer width="100%" height={240}>
                <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--brd)" />
                    <XAxis
                        dataKey="hour"
                        tick={{ fontSize: 10, fill: 'var(--t4)' }}
                        tickFormatter={(v: number) => v % 24 === 0 ? `D${v / 24}` : ''}
                        stroke="var(--brd)"
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: 'var(--t4)' }}
                        tickFormatter={(v: number) => v < 0.001 ? v.toExponential(0) : v.toFixed(3)}
                        stroke="var(--brd)"
                        width={55}
                        domain={[0, maxProb * 1.2]}
                    />
                    <Tooltip
                        contentStyle={{ background: 'var(--bg-s)', border: '1px solid var(--brd)', borderRadius: 6, fontSize: '0.7rem' }}
                        labelStyle={{ color: 'var(--t2)', fontFamily: 'JetBrains Mono, monospace' }}
                        formatter={(v: any) => [Number(v).toExponential(4), 'P(collision)']}
                        labelFormatter={(v: any) => `Hour ${v}`}
                    />
                    {/* Day markers */}
                    {[24, 48, 72, 96, 120, 144].map(h => (
                        <ReferenceLine key={h} x={h} stroke="var(--brd)" strokeDasharray="3 3" />
                    ))}
                    <Line
                        type="monotone"
                        dataKey="probability"
                        stroke="var(--r-high)"
                        strokeWidth={1.5}
                        dot={false}
                        activeDot={{ r: 3, fill: 'var(--r-high)' }}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
