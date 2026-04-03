import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { DetailedThreatItem } from '../../api/client';

interface Props {
    threats: DetailedThreatItem[];
}

export function UncertaintyChart({ threats }: Props) {
    const data = threats.map(t => ({
        name: t.debris_name.length > 12 ? t.debris_name.slice(0, 12) + '...' : t.debris_name,
        epistemic: t.epistemic_uncertainty,
        aleatoric: Math.max(0, t.uncertainty_pct / 100 - t.epistemic_uncertainty),
        total: t.uncertainty_pct / 100,
    }));

    return (
        <div className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: '.72rem', fontWeight: 600, color: 'var(--t3)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '.05em' }}>
                Uncertainty Decomposition
            </div>
            <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--brd)" />
                    <XAxis
                        dataKey="name"
                        tick={{ fontSize: 9, fill: 'var(--t4)' }}
                        stroke="var(--brd)"
                        angle={-30}
                        textAnchor="end"
                        height={50}
                    />
                    <YAxis
                        tick={{ fontSize: 10, fill: 'var(--t4)' }}
                        stroke="var(--brd)"
                        width={50}
                        tickFormatter={(v: number) => v < 0.01 ? v.toExponential(0) : v.toFixed(2)}
                    />
                    <Tooltip
                        contentStyle={{ background: 'var(--bg-s)', border: '1px solid var(--brd)', borderRadius: 6, fontSize: '0.7rem' }}
                        formatter={(v: any, name: any) => [Number(v).toExponential(3), name]}
                    />
                    <Legend wrapperStyle={{ fontSize: '0.65rem', color: 'var(--t4)' }} />
                    <Bar dataKey="epistemic" stackId="a" fill="#3B82F6" name="Epistemic" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="aleatoric" stackId="a" fill="#8B5CF6" name="Aleatoric" radius={[2, 2, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
